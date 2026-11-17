"""
Live Interactive Inference & Line-Level Bug Localization Demo.
Given any C/C++ function (from a file, test-set index, or built-in snippet):
  1. Strips C/C++ comments while preserving exact line numbers
  2. Generates the tree-sitter-cpp AST-anonymized twin (var_1, func_1)
  3. Returns predicted vulnerability probabilities P(y=1 | x) & P(y=1 | x_anon)
     across all trained models (Baselines, Single-Window Baseline, and 5-Upgrade Model)
  4. Ranks the top suspicious lines using Upgrade #5's Line-Level Localization Head
"""

import argparse
import bisect
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src import config
from src.ast_graph import anonymize_code_function
from src.dataset import _compute_line_char_starts, encode_hierarchical_function
from src.model import HierarchicalGraphSiameseVulnDetector
from src.preprocessing import extract_structural_features, strip_comments_preserve_lines

DEFAULT_SAMPLE_CODE = """
static int qemu_net_receive_packet(QEMUClientState *s, const uint8_t *pkt_buf, int pkt_size) {
    /* FIXME: unchecked packet length copy into fixed DMA buffer */
    char *dma_rx_buffer = (char *)malloc(256);
    if (!dma_rx_buffer) {
        return -ENOMEM;
    }
    memcpy(dma_rx_buffer, pkt_buf, pkt_size);
    if (s->irq_enabled) {
        qemu_set_irq(s->irq_line, 1);
    }
    free(dma_rx_buffer);
    return pkt_size;
}
""".strip()


def _map_token_probs_to_lines(
    clean_code: str,
    tokenizer,
    token_probs: np.ndarray,
    max_chunks: int,
) -> list[tuple[int, float, str]]:
    """Aggregates per-chunk subword probabilities back to 1-indexed source lines."""
    lines = clean_code.splitlines()
    n_lines = max(1, len(lines))
    line_starts = _compute_line_char_starts(clean_code)
    line_scores = np.zeros(n_lines, dtype=np.float64)

    enc = tokenizer(clean_code, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    full_offsets = enc["offset_mapping"]
    total_tokens = len(full_offsets)
    max_content = config.CHUNK_SIZE - 2

    # Reconstruct chunk slices matching encode_hierarchical_function
    chunk_slices = []
    if max_chunks == 1:
        if total_tokens <= max_content:
            chunk_slices.append((0, total_tokens))
        else:
            full_offsets = full_offsets[: config.HEAD_TOKENS] + full_offsets[-config.TAIL_TOKENS :]
            chunk_slices.append((0, len(full_offsets)))
    else:
        start = 0
        while start < total_tokens and len(chunk_slices) < max_chunks:
            end = min(start + max_content, total_tokens)
            if len(chunk_slices) == max_chunks - 1 and end < total_tokens:
                start = max(0, total_tokens - max_content)
                end = total_tokens
            chunk_slices.append((start, end))
            if end >= total_tokens:
                break
            start += config.CHUNK_STRIDE

    for k, (tok_start, tok_end) in enumerate(chunk_slices):
        if k >= token_probs.shape[0]:
            break
        sub_offs = full_offsets[tok_start:tok_end]
        for local_idx, (c_start, c_end) in enumerate(sub_offs):
            seq_pos = local_idx + 1  # Offset by [CLS]
            if c_end > c_start and seq_pos < token_probs.shape[1]:
                ln_idx = bisect.bisect_right(line_starts, c_start) - 1
                ln_idx = max(0, min(ln_idx, n_lines - 1))
                line_scores[ln_idx] = max(line_scores[ln_idx], float(token_probs[k, seq_pos]))

    ranked = [(i + 1, float(line_scores[i]), lines[i]) for i in range(n_lines) if lines[i].strip()]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


@torch.no_grad()
def predict_on_code_snippet(raw_code: str, true_label: int | None = None) -> pd.DataFrame:
    """Runs full multi-model vulnerability prediction, shortcut audit, and line localization."""
    clean_code = strip_comments_preserve_lines(raw_code)
    anon_code, n_vars, n_funcs = anonymize_code_function(clean_code)
    struct = extract_structural_features(clean_code)

    ref_tok = AutoTokenizer.from_pretrained(config.MODEL_REGISTRY["unixcoder"])
    bpe_len = len(ref_tok.encode(clean_code, truncation=False, add_special_tokens=True))

    print("\n" + "=" * 90)
    print(" LIVE C/C++ VULNERABILITY PREDICTION, SHORTCUT AUDIT & LINE LOCALIZATION")
    print("=" * 90)
    if true_label is not None:
        print(f" Ground-Truth Label : {'1 (VULNERABLE)' if true_label == 1 else '0 (SAFE)'}")
    print(
        f" Code Profile       : LOC={struct['loc']} | BPE Tokens={bpe_len} | "
        f"Branches={struct['branch_count']} | Pointers={struct['pointer_count']}"
    )
    print(f" AST Anonymization  : Renamed {n_vars} unique variables & {n_funcs} functions -> (var_1, func_1)")
    print("-" * 90)
    for idx, line in enumerate(clean_code.splitlines()[:25], start=1):
        print(f" {idx:3d} | {line}")
    if len(clean_code.splitlines()) > 25:
        print(" ... [remaining lines omitted for display] ...")
    print("=" * 90)

    rows = []

    # 1. Baseline 1: Complexity
    b1_path = config.CHECKPOINT_DIR / "baseline1_complexity.joblib"
    if b1_path.exists():
        b1 = joblib.load(b1_path)
        raw_v = np.array([[struct["loc"], bpe_len, struct["branch_count"], struct["pointer_count"]]], dtype=float)
        x_vec = np.hstack([raw_v, np.log1p(np.maximum(raw_v, 0.0))])
        p_o = float(b1["pipeline"].predict_proba(x_vec)[0, 1])
        th = float(b1["threshold"])
        rows.append({
            "Model": "Baseline 1: Complexity",
            "Threshold": round(th, 3),
            "Orig_Prob (%)": round(p_o * 100, 2),
            "Orig_Verdict": "VULN (1)" if p_o >= th else "SAFE (0)",
            "Anon_Prob (%)": round(p_o * 100, 2),
            "Delta_Prob (%)": 0.0,
            "Flipped?": "No",
        })

    # 2. Baseline 2: TF-IDF
    b2_path = config.CHECKPOINT_DIR / "baseline2_tfidf.joblib"
    if b2_path.exists():
        b2 = joblib.load(b2_path)
        p_o = float(b2["pipeline"].predict_proba([clean_code])[0, 1])
        p_a = float(b2["pipeline"].predict_proba([anon_code])[0, 1])
        th = float(b2["threshold"])
        v_o = "VULN (1)" if p_o >= th else "SAFE (0)"
        v_a = "VULN (1)" if p_a >= th else "SAFE (0)"
        rows.append({
            "Model": "Baseline 2: TF-IDF + LogReg",
            "Threshold": round(th, 3),
            "Orig_Prob (%)": round(p_o * 100, 2),
            "Orig_Verdict": v_o,
            "Anon_Prob (%)": round(p_a * 100, 2),
            "Delta_Prob (%)": round((p_a - p_o) * 100, 2),
            "Flipped?": "YES" if v_o != v_a else "No",
        })

    # 3. Neural Models
    top_suspicious_lines = []
    for run_name, label in [
        ("unixcoder_baseline_single", "UniXcoder (Single-Window)"),
        ("unixcoder_5upgrade_full", "UniXcoder + 5-Upgrades (Full)"),
        ("codebert_5upgrade_full", "CodeBERT + 5-Upgrades (Full)"),
    ]:
        ckpt_dir = config.CHECKPOINT_DIR / run_name
        pred_npz = config.PREDICTION_DIR / f"{run_name}_preds.npz"
        if not ckpt_dir.exists():
            continue

        with open(ckpt_dir / "arch_config.json", "r", encoding="utf-8") as f:
            arch_cfg = json.load(f)

        tok = AutoTokenizer.from_pretrained(ckpt_dir)
        model = HierarchicalGraphSiameseVulnDetector(**arch_cfg).to(config.DEVICE)
        model.load_state_dict(torch.load(ckpt_dir / "pytorch_model.bin", map_location=config.DEVICE))
        model.eval()

        th = float(np.load(pred_npz)["threshold"]) if pred_npz.exists() else 0.50
        is_hier = arch_cfg.get("use_hierarchical", False)
        max_k = arch_cfg.get("max_chunks", 1)
        dummy_lbls = [-100] * max(1, len(clean_code.splitlines()))

        enc_o = encode_hierarchical_function(clean_code, dummy_lbls, tok, max_chunks=max_k, hierarchical=is_hier)
        enc_a = encode_hierarchical_function(anon_code, dummy_lbls, tok, max_chunks=max_k, hierarchical=is_hier)

        out_o = model(
            input_ids=torch.tensor(enc_o["input_ids"], device=config.DEVICE).unsqueeze(0),
            attention_mask=torch.tensor(enc_o["attention_mask"], device=config.DEVICE).unsqueeze(0),
            chunk_mask=torch.tensor(enc_o["chunk_mask"], device=config.DEVICE).unsqueeze(0),
            edge_types=torch.tensor(enc_o["edge_types"], device=config.DEVICE).unsqueeze(0),
        )
        out_a = model(
            input_ids=torch.tensor(enc_a["input_ids"], device=config.DEVICE).unsqueeze(0),
            attention_mask=torch.tensor(enc_a["attention_mask"], device=config.DEVICE).unsqueeze(0),
            chunk_mask=torch.tensor(enc_a["chunk_mask"], device=config.DEVICE).unsqueeze(0),
            edge_types=torch.tensor(enc_a["edge_types"], device=config.DEVICE).unsqueeze(0),
        )

        p_o = float(torch.softmax(out_o["func_logits"].float(), dim=-1)[0, 1].item())
        p_a = float(torch.softmax(out_a["func_logits"].float(), dim=-1)[0, 1].item())
        v_o = "VULN (1)" if p_o >= th else "SAFE (0)"
        v_a = "VULN (1)" if p_a >= th else "SAFE (0)"

        rows.append({
            "Model": label,
            "Threshold": round(th, 3),
            "Orig_Prob (%)": round(p_o * 100, 2),
            "Orig_Verdict": v_o,
            "Anon_Prob (%)": round(p_a * 100, 2),
            "Delta_Prob (%)": round((p_a - p_o) * 100, 2),
            "Flipped?": "YES" if v_o != v_a else "No",
        })

        if arch_cfg.get("use_line_head", False):
            tok_probs = torch.sigmoid(out_o["token_logits"].float())[0].cpu().numpy()
            top_suspicious_lines = _map_token_probs_to_lines(clean_code, tok, tok_probs, max_k)

    res_df = pd.DataFrame(rows)
    print("\n--- PREDICTED VULNERABILITY PROBABILITIES P(y=1 | x) ---")
    print(res_df.to_string(index=False))

    if top_suspicious_lines:
        print("\n--- TOP SUSPICIOUS LINES (Upgrade #5 Line-Level Localization Head) ---")
        for ln_no, score, code_ln in top_suspicious_lines[:5]:
            print(f"  Line {ln_no:3d} [Risk: {score:.3f}] : {code_ln.strip()}")
    print("=" * 90 + "\n")
    return res_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Live C/C++ Vulnerability Prediction Demo")
    parser.add_argument("--file", type=str, default=None, help="Path to a .c/.cpp file")
    parser.add_argument("--test_idx", type=int, default=None, help="Row index in processed test split")
    parser.add_argument("--code", type=str, default=None, help="Inline C/C++ code string")
    args = parser.parse_args()

    if args.file is not None:
        predict_on_code_snippet(Path(args.file).read_text(encoding="utf-8"))
    elif args.test_idx is not None:
        test_df = pd.read_parquet(config.PROCESSED_DATA_DIR / "test.parquet")
        row = test_df.iloc[args.test_idx]
        print(f"Loaded Test Row #{args.test_idx} | ID={row['id']} | Project={row['project']}")
        predict_on_code_snippet(row["func_clean"], true_label=int(row["target"]))
    elif args.code is not None:
        predict_on_code_snippet(args.code)
    else:
        predict_on_code_snippet(DEFAULT_SAMPLE_CODE, true_label=1)


if __name__ == "__main__":
    main()
