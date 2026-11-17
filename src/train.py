"""
Training & Evaluation Pipeline for the Unified 5-Upgrade Architecture
(and Single-Window Baseline Reference) on PrimeVul.
Supports Kaggle multi-GPU DataParallel, BF16/FP16 mixed precision, differential
learning rates, Siamese consistency warmup, and validation PR-AUC/F1 checkpointing.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from src import config
from src.dataset import PrimeVulGraphSiameseDataset
from src.metrics import evaluate_predictions, find_best_threshold
from src.model import HierarchicalGraphSiameseVulnDetector, compute_multitask_siamese_loss


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = config.DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs inference over a DataLoader and returns:
      - func_probs:  (N,) positive vulnerability probabilities P(y=1 | x)
      - token_probs: (N, K, L) per-token bug localization probabilities
    """
    model.eval()
    all_func_probs = []
    all_token_probs = []
    amp_dtype = torch.bfloat16 if config.USE_BF16 else torch.float16

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        chunk_mask = batch["chunk_mask"].to(device, non_blocking=True)
        edge_types = batch["edge_types"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                chunk_mask=chunk_mask,
                edge_types=edge_types,
            )
            f_probs = torch.softmax(outputs["func_logits"].float(), dim=-1)[:, 1]
            t_probs = torch.sigmoid(outputs["token_logits"].float())

        all_func_probs.append(f_probs.cpu().numpy())
        all_token_probs.append(t_probs.cpu().numpy())

    return np.concatenate(all_func_probs, axis=0), np.concatenate(all_token_probs, axis=0)


def train_model(
    mode: str = "full",
    backbone_key: str = "unixcoder",
    epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.TRAIN_BATCH_SIZE,
    grad_accum_steps: int = config.GRAD_ACCUM_STEPS,
    lr: float = config.DEFAULT_LR,
    head_lr: float = config.HEAD_LR,
) -> dict:
    """
    Trains either:
      - mode == 'full': All 5 architectural upgrades (Hierarchical Multi-Chunk +
        Siamese Anonymization Invariance + AST/Def-Use Edge Bias + Layer-Weighted
        Multi-Head Attention Pooling + Joint Line-Level Multi-Task Head).
      - mode == 'baseline': Single-window Head+Tail UniXcoder reference model for ablation.
    """
    config.set_seed()
    backbone_name = config.MODEL_REGISTRY[backbone_key]
    is_full = mode == "full"
    run_name = f"{backbone_key}_5upgrade_full" if is_full else f"{backbone_key}_baseline_single"
    save_dir = config.CHECKPOINT_DIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(f" TRAINING RUN: {run_name} | Backbone: {backbone_name}")
    print(f" Device: {config.DEVICE} (GPUs: {config.NUM_GPUS}) | BF16: {config.USE_BF16}")
    print(f" Batch Size: {batch_size} x {grad_accum_steps} accum = {batch_size * grad_accum_steps} effective")
    print("=" * 78)

    train_df = pd.read_parquet(config.PROCESSED_DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(config.PROCESSED_DATA_DIR / "validation.parquet")
    test_df = pd.read_parquet(config.PROCESSED_DATA_DIR / "test.parquet")

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    train_dataset = PrimeVulGraphSiameseDataset(
        codes=train_df["func_clean"].tolist(),
        targets=train_df["target"].tolist(),
        line_masks_json=train_df["line_vuln_mask"].tolist(),
        tokenizer=tokenizer,
        hierarchical=is_full,
        include_siamese=is_full,
    )
    val_dataset = PrimeVulGraphSiameseDataset(
        codes=val_df["func_clean"].tolist(),
        targets=val_df["target"].tolist(),
        line_masks_json=val_df["line_vuln_mask"].tolist(),
        tokenizer=tokenizer,
        hierarchical=is_full,
        include_siamese=False,
    )
    test_dataset = PrimeVulGraphSiameseDataset(
        codes=test_df["func_clean"].tolist(),
        targets=test_df["target"].tolist(),
        line_masks_json=test_df["line_vuln_mask"].tolist(),
        tokenizer=tokenizer,
        hierarchical=is_full,
        include_siamese=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    model = HierarchicalGraphSiameseVulnDetector(
        backbone_name=backbone_name,
        max_chunks=config.MAX_CHUNKS if is_full else 1,
        use_graph_bias=is_full,
        use_layer_pooling=is_full,
        use_hierarchical=is_full,
        use_line_head=is_full,
    ).to(config.DEVICE)

    # Multi-GPU support for Kaggle 2x T4 setups
    if config.NUM_GPUS > 1:
        print(f"Wrapping model in nn.DataParallel across {config.NUM_GPUS} GPUs...")
        model = nn.DataParallel(model)

    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    # Differential learning rate: lower LR for pretrained encoder, higher LR for new heads
    no_decay = ["bias", "LayerNorm.weight"]
    encoder_params_decay, encoder_params_no_decay = [], []
    head_params_decay, head_params_no_decay = [], []

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = any(nd in name for nd in no_decay)
        if name.startswith("encoder."):
            (encoder_params_no_decay if is_no_decay else encoder_params_decay).append(param)
        else:
            (head_params_no_decay if is_no_decay else head_params_decay).append(param)

    optimizer = torch.optim.AdamW([
        {"params": encoder_params_decay, "lr": lr, "weight_decay": config.WEIGHT_DECAY},
        {"params": encoder_params_no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": head_params_decay, "lr": head_lr, "weight_decay": config.WEIGHT_DECAY},
        {"params": head_params_no_decay, "lr": head_lr, "weight_decay": 0.0},
    ])

    total_optim_steps = max(1, (len(train_loader) // max(1, grad_accum_steps)) * epochs)
    warmup_steps = int(total_optim_steps * config.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_FP16)
    amp_dtype = torch.bfloat16 if config.USE_BF16 else torch.float16

    # Compute dynamic positive-class weight from training split (capped for stability)
    pos_rate = float(train_df["target"].mean())
    dynamic_pos_weight = min(config.POS_CLASS_WEIGHT, max(1.0, (1.0 - pos_rate) / max(pos_rate, 1e-4)))

    y_val = val_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()
    best_val_score = -1.0
    best_thresh = 0.50
    best_epoch = 1

    # Save initial weights in case validation score ties at 0 on tiny smoke-test splits
    torch.save(raw_model.state_dict(), save_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(save_dir)
    with open(save_dir / "arch_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "backbone_name": backbone_name,
                "max_chunks": config.MAX_CHUNKS if is_full else 1,
                "use_graph_bias": is_full,
                "use_layer_pooling": is_full,
                "use_hierarchical": is_full,
                "use_line_head": is_full,
            },
            f,
            indent=2,
        )

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Linearly warm up Siamese consistency weight so early epochs stabilize on imbalanced labels
        warmup_factor = min(1.0, epoch / max(1.0, epochs * 0.5))
        curr_siamese_w = (config.LAMBDA_SIAMESE_JSD * warmup_factor) if is_full else 0.0
        curr_cosine_w = (config.LAMBDA_SIAMESE_COS * warmup_factor) if is_full else 0.0
        curr_line_w = config.BETA_LINE_MULTITASK if is_full else 0.0

        running = {"loss_total": 0.0, "loss_func": 0.0, "loss_line": 0.0, "loss_jsd": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)

        for step, batch in enumerate(pbar, start=1):
            kwargs = {
                "input_ids": batch["input_ids"].to(config.DEVICE, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(config.DEVICE, non_blocking=True),
                "chunk_mask": batch["chunk_mask"].to(config.DEVICE, non_blocking=True),
                "edge_types": batch["edge_types"].to(config.DEVICE, non_blocking=True),
            }
            if is_full and "anon_input_ids" in batch:
                kwargs["anon_input_ids"] = batch["anon_input_ids"].to(config.DEVICE, non_blocking=True)
                kwargs["anon_attention_mask"] = batch["anon_attention_mask"].to(config.DEVICE, non_blocking=True)
                kwargs["anon_chunk_mask"] = batch["anon_chunk_mask"].to(config.DEVICE, non_blocking=True)
                kwargs["anon_edge_types"] = batch["anon_edge_types"].to(config.DEVICE, non_blocking=True)

            labels = batch["labels"].to(config.DEVICE, non_blocking=True)
            line_labels = batch["line_labels"].to(config.DEVICE, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=(config.DEVICE == "cuda")
            ):
                outputs = model(**kwargs)
                loss, step_stats = compute_multitask_siamese_loss(
                    outputs=outputs,
                    labels=labels,
                    line_labels=line_labels if is_full else None,
                    siamese_weight=curr_siamese_w,
                    cosine_weight=curr_cosine_w,
                    line_weight=curr_line_w,
                    pos_weight=dynamic_pos_weight,
                )
                scaled_loss = loss / grad_accum_steps

            if config.USE_FP16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            for k in running:
                running[k] += step_stats[k]

            if step % grad_accum_steps == 0 or step == len(train_loader):
                if config.USE_FP16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({
                "tot": f"{running['loss_total'] / step:.3f}",
                "fn": f"{running['loss_func'] / step:.3f}",
                "ln": f"{running['loss_line'] / step:.3f}",
                "jsd": f"{running['loss_jsd'] / step:.3f}",
            })

        val_probs, _ = predict_dataset(model, val_loader)
        curr_thresh = find_best_threshold(y_val, val_probs, optimize_for="vuln_f1")
        val_m = evaluate_predictions(y_val, val_probs, threshold=curr_thresh)
        # Composite selection metric: PR-AUC + Vuln F1 (ideal for imbalanced PrimeVul)
        composite_val = 0.5 * val_m["PR_AUC (%)"] + 0.5 * val_m["Vuln_F1 (%)"]

        print(
            f"  --> [Epoch {epoch} ({time.time() - t0:.1f}s)] "
            f"Val PR-AUC: {val_m['PR_AUC (%)']:.2f}% | Val Vuln F1: {val_m['Vuln_F1 (%)']:.2f}% | "
            f"Val Macro F1: {val_m['Macro_F1 (%)']:.2f}% | FPR@R80: {val_m['FPR@R80 (%)']:.2f}% (tau={curr_thresh:.3f})"
        )

        if composite_val > best_val_score:
            best_val_score = composite_val
            best_thresh = curr_thresh
            best_epoch = epoch
            torch.save(raw_model.state_dict(), save_dir / "pytorch_model.bin")
            print(f"      *** Saved best checkpoint (Epoch {epoch}, Score={composite_val:.2f}) ***")

    # Reload best checkpoint and evaluate on locked Test split
    raw_model.load_state_dict(torch.load(save_dir / "pytorch_model.bin", map_location=config.DEVICE))
    val_probs, _ = predict_dataset(model, val_loader)
    test_probs, test_token_probs = predict_dataset(model, test_loader)

    pretty_name = (
        f"{backbone_key.upper()} + 5-Upgrades (Hier+Graph+Siamese+Line)"
        if is_full
        else f"{backbone_key.upper()} (Single-Window Baseline)"
    )
    val_res = evaluate_predictions(y_val, val_probs, best_thresh, pretty_name, "Validation")
    test_res = evaluate_predictions(y_test, test_probs, best_thresh, pretty_name, "Test")

    np.savez(
        config.PREDICTION_DIR / f"{run_name}_preds.npz",
        y_val=y_val,
        val_probs=val_probs,
        y_test=y_test,
        test_probs=test_probs,
        test_token_probs=test_token_probs,
        threshold=best_thresh,
        exceeds_512=test_df["exceeds_512"].to_numpy(),
    )

    metrics_csv = config.TABLE_DIR / "transformer_metrics.csv"
    new_df = pd.DataFrame([val_res, test_res])
    if metrics_csv.exists():
        old_df = pd.read_csv(metrics_csv)
        old_df = old_df[old_df["Model"] != pretty_name]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(metrics_csv, index=False)

    print("\n================ FINAL MODEL EVALUATION ================")
    print(new_df.to_string(index=False))
    return {"val": val_res, "test": test_res}
