"""
Comprehensive Evaluation, PrimeVul Paired Benchmark, Counterfactual AST
Shortcut Audit, and Length Stratification Pipeline.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src import config
from src.ast_graph import generate_anonymized_split
from src.dataset import PrimeVulGraphSiameseDataset
from src.metrics import (
    compute_flip_rate,
    evaluate_paired_predictions,
    evaluate_predictions,
)
from src.model import HierarchicalGraphSiameseVulnDetector
from src.train import predict_dataset


def load_trained_detector(run_name: str) -> tuple[HierarchicalGraphSiameseVulnDetector, AutoTokenizer, float]:
    """Loads a saved HierarchicalGraphSiameseVulnDetector checkpoint and threshold."""
    ckpt_dir = config.CHECKPOINT_DIR / run_name
    pred_npz = config.PREDICTION_DIR / f"{run_name}_preds.npz"
    with open(ckpt_dir / "arch_config.json", "r", encoding="utf-8") as f:
        arch_cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = HierarchicalGraphSiameseVulnDetector(**arch_cfg).to(config.DEVICE)
    state_dict = torch.load(ckpt_dir / "pytorch_model.bin", map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    thresh = float(np.load(pred_npz)["threshold"]) if pred_npz.exists() else 0.50
    return model, tokenizer, thresh


def run_counterfactual_shortcut_audit(anon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Audits all trained models on Original vs. AST-Anonymized (var_1, func_1) code
    to measure shortcut invariance (Delta PR-AUC, Delta Vuln F1, and Flip Rate).
    """
    print("\n================ 1. COUNTERFACTUAL AST SHORTCUT AUDIT ================")
    y_test = anon_df["target"].to_numpy()
    rows = []

    # 1. Baseline 2: TF-IDF
    b2_path = config.CHECKPOINT_DIR / "baseline2_tfidf.joblib"
    if b2_path.exists():
        b2 = joblib.load(b2_path)
        pipe, thresh = b2["pipeline"], float(b2["threshold"])
        orig_probs = pipe.predict_proba(anon_df["func_clean"])[:, 1]
        anon_probs = pipe.predict_proba(anon_df["func_anon"])[:, 1]

        m_orig = evaluate_predictions(y_test, orig_probs, thresh)
        m_anon = evaluate_predictions(y_test, anon_probs, thresh)
        flip = compute_flip_rate((orig_probs >= thresh).astype(int), (anon_probs >= thresh).astype(int))
        rows.append({
            "Model": "Baseline 2: TF-IDF + LogReg",
            "Orig_PR_AUC (%)": m_orig["PR_AUC (%)"],
            "Anon_PR_AUC (%)": m_anon["PR_AUC (%)"],
            "Delta_PR_AUC (%)": round(m_anon["PR_AUC (%)"] - m_orig["PR_AUC (%)"], 2),
            "Orig_Vuln_F1 (%)": m_orig["Vuln_F1 (%)"],
            "Anon_Vuln_F1 (%)": m_anon["Vuln_F1 (%)"],
            "Flip_Rate (%)": round(flip, 2),
            "Mean_Prob_Shift (%)": round(float(np.mean(np.abs(orig_probs - anon_probs)) * 100.0), 2),
        })

    # 2. Neural Models
    model_runs = [
        ("unixcoder_baseline_single", "UniXcoder (Single-Window Baseline)"),
        ("unixcoder_5upgrade_full", "UniXcoder + 5-Upgrades (Full Model)"),
        ("codebert_5upgrade_full", "CodeBERT + 5-Upgrades (Full Model)"),
    ]

    for run_name, label in model_runs:
        ckpt_dir = config.CHECKPOINT_DIR / run_name
        pred_npz = config.PREDICTION_DIR / f"{run_name}_preds.npz"
        if not ckpt_dir.exists() or not pred_npz.exists():
            continue

        model, tokenizer, thresh = load_trained_detector(run_name)
        orig_probs = np.load(pred_npz)["test_probs"]
        is_hier = "5upgrade" in run_name

        anon_ds = PrimeVulGraphSiameseDataset(
            codes=anon_df["func_anon"].tolist(),
            targets=anon_df["target"].tolist(),
            line_masks_json=anon_df["line_vuln_mask"].tolist(),
            tokenizer=tokenizer,
            hierarchical=is_hier,
            include_siamese=False,
        )
        anon_loader = DataLoader(anon_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)
        anon_probs, _ = predict_dataset(model, anon_loader)

        m_orig = evaluate_predictions(y_test, orig_probs, thresh)
        m_anon = evaluate_predictions(y_test, anon_probs, thresh)
        flip = compute_flip_rate((orig_probs >= thresh).astype(int), (anon_probs >= thresh).astype(int))

        rows.append({
            "Model": label,
            "Orig_PR_AUC (%)": m_orig["PR_AUC (%)"],
            "Anon_PR_AUC (%)": m_anon["PR_AUC (%)"],
            "Delta_PR_AUC (%)": round(m_anon["PR_AUC (%)"] - m_orig["PR_AUC (%)"], 2),
            "Orig_Vuln_F1 (%)": m_orig["Vuln_F1 (%)"],
            "Anon_Vuln_F1 (%)": m_anon["Vuln_F1 (%)"],
            "Flip_Rate (%)": round(flip, 2),
            "Mean_Prob_Shift (%)": round(float(np.mean(np.abs(orig_probs - anon_probs)) * 100.0), 2),
        })

    audit_df = pd.DataFrame(rows)
    out_csv = config.TABLE_DIR / "shortcut_audit_results.csv"
    audit_df.to_csv(out_csv, index=False)
    print(audit_df.to_string(index=False))
    return audit_df


def run_paired_primevul_evaluation() -> pd.DataFrame:
    """Evaluates models on matched (pre-patch vulnerable, post-patch fixed) PrimeVul pairs."""
    paired_path = config.PROCESSED_DATA_DIR / "test_paired.parquet"
    if not paired_path.exists():
        return pd.DataFrame()

    paired_df = pd.read_parquet(paired_path)
    if paired_df.empty:
        return pd.DataFrame()

    print("\n================ 2. PRIMEVUL PAIRED BUG-VS-FIX EVALUATION ================")
    rows = []

    b2_path = config.CHECKPOINT_DIR / "baseline2_tfidf.joblib"
    if b2_path.exists():
        b2 = joblib.load(b2_path)
        pipe, thresh = b2["pipeline"], float(b2["threshold"])
        v_probs = pipe.predict_proba(paired_df["vuln_func"])[:, 1]
        s_probs = pipe.predict_proba(paired_df["safe_func"])[:, 1]
        rows.append(evaluate_paired_predictions(v_probs, s_probs, thresh, "Baseline 2: TF-IDF + LogReg"))

    for run_name, label in [
        ("unixcoder_baseline_single", "UniXcoder (Single-Window Baseline)"),
        ("unixcoder_5upgrade_full", "UniXcoder + 5-Upgrades (Full Model)"),
        ("codebert_5upgrade_full", "CodeBERT + 5-Upgrades (Full Model)"),
    ]:
        ckpt_dir = config.CHECKPOINT_DIR / run_name
        if not ckpt_dir.exists():
            continue

        model, tokenizer, thresh = load_trained_detector(run_name)
        is_hier = "5upgrade" in run_name
        dummy_masks = ["[-100]"] * len(paired_df)

        v_ds = PrimeVulGraphSiameseDataset(
            paired_df["vuln_func"].tolist(), [1] * len(paired_df), dummy_masks,
            tokenizer=tokenizer, hierarchical=is_hier, include_siamese=False
        )
        s_ds = PrimeVulGraphSiameseDataset(
            paired_df["safe_func"].tolist(), [0] * len(paired_df), dummy_masks,
            tokenizer=tokenizer, hierarchical=is_hier, include_siamese=False
        )
        v_probs, _ = predict_dataset(model, DataLoader(v_ds, batch_size=config.EVAL_BATCH_SIZE))
        s_probs, _ = predict_dataset(model, DataLoader(s_ds, batch_size=config.EVAL_BATCH_SIZE))
        rows.append(evaluate_paired_predictions(v_probs, s_probs, thresh, label))

    paired_res_df = pd.DataFrame(rows)
    paired_res_df.to_csv(config.TABLE_DIR / "paired_evaluation_results.csv", index=False)
    print(paired_res_df.to_string(index=False))
    return paired_res_df


def run_length_stratification(test_df: pd.DataFrame) -> pd.DataFrame:
    """Compares Short (<=512 tokens) vs. Long (>512 tokens) performance across models."""
    print("\n================ 3. CONTEXT LENGTH STRATIFICATION (<=512 vs >512) ================")
    y_test = test_df["target"].to_numpy()
    is_long = test_df["exceeds_512"].to_numpy() == 1
    is_short = ~is_long

    entries = []
    base_npz = config.PREDICTION_DIR / "baselines_test_preds.npz"
    if base_npz.exists():
        b = np.load(base_npz)
        entries.append(("Baseline 1: Complexity", b["b1_probs"], float(b["b1_thresh"])))
        entries.append(("Baseline 2: TF-IDF", b["b2_probs"], float(b["b2_thresh"])))

    for run_name, label in [
        ("unixcoder_baseline_single", "UniXcoder (Single-Window)"),
        ("unixcoder_5upgrade_full", "UniXcoder + 5-Upgrades"),
        ("codebert_5upgrade_full", "CodeBERT + 5-Upgrades"),
    ]:
        p_path = config.PREDICTION_DIR / f"{run_name}_preds.npz"
        if p_path.exists():
            d = np.load(p_path)
            entries.append((label, d["test_probs"], float(d["threshold"])))

    rows = []
    for label, probs, thresh in entries:
        m_all = evaluate_predictions(y_test, probs, thresh)
        m_short = evaluate_predictions(y_test[is_short], probs[is_short], thresh) if is_short.any() else m_all
        m_long = evaluate_predictions(y_test[is_long], probs[is_long], thresh) if is_long.any() else m_all
        rows.append({
            "Model": label,
            "All_PR_AUC (%)": m_all["PR_AUC (%)"],
            "Short_<=512_PR_AUC (%)": m_short["PR_AUC (%)"],
            "Long_>512_PR_AUC (%)": m_long["PR_AUC (%)"],
            "All_Vuln_F1 (%)": m_all["Vuln_F1 (%)"],
            "Short_<=512_Vuln_F1 (%)": m_short["Vuln_F1 (%)"],
            "Long_>512_Vuln_F1 (%)": m_long["Vuln_F1 (%)"],
        })

    strat_df = pd.DataFrame(rows)
    strat_df.to_csv(config.TABLE_DIR / "length_stratification_ablation.csv", index=False)
    print(strat_df.to_string(index=False))
    return strat_df


def inspect_learned_layer_weights() -> None:
    """Logs the learned softmax layer weights alpha_1..alpha_12 from Upgrade #4."""
    for run_name in ["unixcoder_5upgrade_full", "codebert_5upgrade_full"]:
        ckpt_dir = config.CHECKPOINT_DIR / run_name
        if not ckpt_dir.exists():
            continue
        model, _, _ = load_trained_detector(run_name)
        if getattr(model, "use_layer_pooling", False):
            alphas = torch.softmax(model.layer_weights.detach().cpu(), dim=0).numpy()
            print(f"\n[Upgrade #4 Learned Layer Weights for {run_name}]:")
            print("  " + " | ".join([f"L{i+1}:{w:.3f}" for i, w in enumerate(alphas)]))


def generate_audit_figures(audit_df: pd.DataFrame, strat_df: pd.DataFrame) -> None:
    """Saves publication-quality PDF/PNG figures for the shortcut audit and length ablation."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    if not audit_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        melted = audit_df.melt(
            id_vars="Model",
            value_vars=["Orig_PR_AUC (%)", "Anon_PR_AUC (%)"],
            var_name="Condition",
            value_name="PR-AUC (%)",
        )
        melted["Condition"] = melted["Condition"].map({
            "Orig_PR_AUC (%)": "Original Code",
            "Anon_PR_AUC (%)": "AST-Anonymized (var_1, func_1)",
        })
        sns.barplot(data=melted, x="Model", y="PR-AUC (%)", hue="Condition", palette=["#2b5c8f", "#e67e22"], ax=axes[0])
        axes[0].set_title("PR-AUC Before vs. After AST Variable Anonymization")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=15)

        sns.barplot(data=audit_df, x="Model", y="Flip_Rate (%)", color="#c0392b", ax=axes[1])
        axes[1].set_title("Counterfactual Prediction Flip Rate (%)")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=15)

        plt.tight_layout()
        fig.savefig(config.FIGURE_DIR / "shortcut_audit_comparison.pdf", dpi=300)
        fig.savefig(config.FIGURE_DIR / "shortcut_audit_comparison.png", dpi=300)
        plt.close(fig)

    if not strat_df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        melted_len = strat_df.melt(
            id_vars="Model",
            value_vars=["Short_<=512_PR_AUC (%)", "Long_>512_PR_AUC (%)"],
            var_name="Regime",
            value_name="PR-AUC (%)",
        )
        melted_len["Regime"] = melted_len["Regime"].map({
            "Short_<=512_PR_AUC (%)": "Short Functions (<=512 Tokens)",
            "Long_>512_PR_AUC (%)": "Long Functions (>512 Tokens)",
        })
        sns.barplot(data=melted_len, x="Model", y="PR-AUC (%)", hue="Regime", palette=["#27ae60", "#8e44ad"], ax=ax)
        ax.set_title("PR-AUC Across Short vs. Long Function Regimes")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        fig.savefig(config.FIGURE_DIR / "length_stratification_f1.pdf", dpi=300)
        fig.savefig(config.FIGURE_DIR / "length_stratification_f1.png", dpi=300)
        plt.close(fig)


def run_full_audit() -> None:
    """Runs all evaluation, paired benchmark, shortcut audit, and plotting routines."""
    config.set_seed()
    anon_path = config.ANONYMIZED_DATA_DIR / "test_anonymized.parquet"
    anon_df = pd.read_parquet(anon_path) if anon_path.exists() else generate_anonymized_split("test")

    audit_df = run_counterfactual_shortcut_audit(anon_df)
    run_paired_primevul_evaluation()
    strat_df = run_length_stratification(anon_df)
    inspect_learned_layer_weights()
    generate_audit_figures(audit_df, strat_df)
