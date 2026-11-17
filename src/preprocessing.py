"""
Data ingestion, line-preserving C/C++ comment stripping, paired diff line-mask
generation, structural feature extraction, and token-length EDA for PrimeVul.
"""

import difflib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src import config

COMMENT_REGEX = re.compile(
    r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\r\n]*|/\*[\s\S]*?\*/)',
    re.MULTILINE,
)

BRANCH_REGEX = re.compile(r"\b(if|else|for|while|switch|case|goto|catch|throw)\b")


def strip_comments_preserve_lines(code: str) -> str:
    """
    Removes C/C++ block (/* ... */) and inline (// ...) comments while preserving
    string literals AND exact line numbers (newlines inside block comments are kept)
    so line-level diff masks stay 1-to-1 aligned with source line indices.
    """
    def _replacer(match: re.Match) -> str:
        token = match.group(0)
        if token.startswith("/"):
            newline_count = token.count("\n")
            return "\n" * newline_count if newline_count > 0 else " "
        return token

    cleaned = COMMENT_REGEX.sub(_replacer, code)
    # Strip trailing whitespace per line without removing lines
    lines = [ln.rstrip() for ln in cleaned.splitlines()]
    return "\n".join(lines)


def extract_structural_features(code: str) -> dict:
    """Computes static complexity heuristics on comment-stripped C/C++ code."""
    non_empty_lines = [ln for ln in code.splitlines() if ln.strip()]
    return {
        "loc": len(non_empty_lines),
        "total_lines": max(1, len(code.splitlines())),
        "branch_count": len(BRANCH_REGEX.findall(code)),
        "pointer_count": code.count("*"),
    }


def compute_paired_line_masks(vuln_code: str, fixed_code: str) -> list[int]:
    """
    Computes a binary vulnerability mask per line of `vuln_code` by diffing
    against its paired post-patch `fixed_code` via SequenceMatcher.
    Lines deleted or replaced in the security patch receive label 1; unchanged lines receive 0.
    If the patch purely inserted a bounds check (only 'insert' opcodes), the lines
    immediately adjacent to the insertion point are marked 1 as the vulnerable site.
    """
    vuln_lines = vuln_code.splitlines()
    fixed_lines = fixed_code.splitlines()
    n_lines = len(vuln_lines)
    if n_lines == 0:
        return []

    mask = [0] * n_lines
    matcher = difflib.SequenceMatcher(None, vuln_lines, fixed_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            for idx in range(i1, i2):
                if 0 <= idx < n_lines:
                    mask[idx] = 1
        elif tag == "insert":
            # Pure insertion patch (e.g. adding missing if-bounds check before sink)
            if 0 <= i1 < n_lines:
                mask[i1] = 1
            if i1 - 1 >= 0:
                mask[i1 - 1] = 1

    # Ensure at least one line is marked if code differs
    if sum(mask) == 0 and vuln_lines != fixed_lines:
        mask[min(n_lines - 1, n_lines // 2)] = 1
    return mask


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_split_dataframe(
    unpaired_records: list[dict],
    paired_records: list[dict],
    tokenizer,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merges PrimeVul unpaired and paired records, generates line-level vulnerability
    masks for paired samples, and computes BPE token statistics.
    Returns (full_split_df, paired_only_df).
    """
    # Pair consecutive (vuln, benign) entries sharing commit_id in paired_records
    paired_line_lookup: dict[tuple[str, str], list[int]] = {}
    paired_eval_rows = []

    i = 0
    while i + 1 < len(paired_records):
        r1 = paired_records[i]
        r2 = paired_records[i + 1]
        if r1.get("commit_id") == r2.get("commit_id") and int(r1["target"]) != int(r2["target"]):
            vuln_rec = r1 if int(r1["target"]) == 1 else r2
            safe_rec = r2 if int(r1["target"]) == 1 else r1

            v_clean = strip_comments_preserve_lines(vuln_rec["func"])
            s_clean = strip_comments_preserve_lines(safe_rec["func"])
            v_mask = compute_paired_line_masks(v_clean, s_clean)
            s_mask = [0] * max(1, len(s_clean.splitlines()))

            paired_line_lookup[(str(vuln_rec.get("idx", i)), v_clean)] = v_mask
            paired_line_lookup[(str(safe_rec.get("idx", i + 1)), s_clean)] = s_mask

            paired_eval_rows.append({
                "pair_id": len(paired_eval_rows),
                "commit_id": str(vuln_rec.get("commit_id", "")),
                "project": str(vuln_rec.get("project", "unknown")),
                "vuln_func": v_clean,
                "safe_func": s_clean,
                "vuln_line_mask": json.dumps(v_mask),
            })
            i += 2
        else:
            i += 1

    # Combine unpaired + paired records (deduplicating by cleaned code + target)
    combined_raw = list(unpaired_records) + list(paired_records)
    seen_keys = set()
    processed_rows = []

    for row_idx, rec in enumerate(tqdm(combined_raw, desc=f"Processing {split_name}")):
        raw_code = rec["func"]
        target = int(rec["target"])
        clean_code = strip_comments_preserve_lines(raw_code)
        dedup_key = (clean_code, target)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        rec_id = str(rec.get("idx", f"{split_name}_{row_idx}"))
        lookup_key = (rec_id, clean_code)

        if lookup_key in paired_line_lookup:
            line_mask = paired_line_lookup[lookup_key]
            has_line_mask = 1
        elif target == 0:
            # Safe functions have 0 vulnerable lines everywhere
            line_mask = [0] * max(1, len(clean_code.splitlines()))
            has_line_mask = 1
        else:
            # Unpaired vulnerable function without post-patch diff -> ignore line loss (-100)
            line_mask = [-100] * max(1, len(clean_code.splitlines()))
            has_line_mask = 0

        struct = extract_structural_features(clean_code)
        bpe_len = len(tokenizer.encode(clean_code, truncation=False, add_special_tokens=True))

        processed_rows.append({
            "id": rec_id,
            "project": str(rec.get("project", "unknown")),
            "commit_id": str(rec.get("commit_id", "")),
            "target": target,
            "func_clean": clean_code,
            "line_vuln_mask": json.dumps(line_mask),
            "has_line_mask": has_line_mask,
            "loc": struct["loc"],
            "total_lines": struct["total_lines"],
            "branch_count": struct["branch_count"],
            "pointer_count": struct["pointer_count"],
            "bpe_length": bpe_len,
            "exceeds_512": int(bpe_len > config.CHUNK_SIZE),
        })

    return pd.DataFrame(processed_rows), pd.DataFrame(paired_eval_rows)


def prepare_primevul_data(use_sample: bool = False) -> dict[str, pd.DataFrame]:
    """
    Loads PrimeVul JSONL splits (or permitted sample dataset if `use_sample=True`),
    strips comments, builds diff-derived line masks, saves Parquet splits, and
    generates dataset statistics and token-length distribution figures.
    """
    from transformers import AutoTokenizer
    config.set_seed()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_REGISTRY["unixcoder"])

    if use_sample:
        sample_path = config.SAMPLE_DATA_DIR / "primevul_sample.jsonl"
        print(f"[Sample Mode] Loading permitted C/C++ sample dataset from: {sample_path}")
        all_sample = _load_jsonl(sample_path)
        # Partition 6 pairs (12 functions) into 4 train pairs, 1 val pair, 1 test pair
        raw_splits = {
            "train": ([], all_sample[:8]),
            "validation": ([], all_sample[8:10]),
            "test": ([], all_sample[10:12]),
        }
    else:
        raw_splits = {}
        for split_name in ["train", "validation", "test"]:
            unpaired_path = config.PRIMEVUL_RAW_DIR / config.PRIMEVUL_FILES[split_name]
            paired_path = config.PRIMEVUL_RAW_DIR / config.PRIMEVUL_FILES[f"{split_name}_paired"]
            if not unpaired_path.exists() and not paired_path.exists():
                raise FileNotFoundError(
                    f"PrimeVul files not found in {config.PRIMEVUL_RAW_DIR}.\n"
                    f"Please download PrimeVul JSONL files as described in README.md, "
                    f"set PRIMEVUL_RAW_DIR, or pass --use_sample for immediate testing."
                )
            unpaired_recs = _load_jsonl(unpaired_path) if unpaired_path.exists() else []
            paired_recs = _load_jsonl(paired_path) if paired_path.exists() else []
            raw_splits[split_name] = (unpaired_recs, paired_recs)

    splits_data = {}
    summary_rows = []

    for split_name in ["train", "validation", "test"]:
        unpaired_recs, paired_recs = raw_splits[split_name]
        df, paired_df = _build_split_dataframe(unpaired_recs, paired_recs, tokenizer, split_name)

        out_path = config.PROCESSED_DATA_DIR / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        splits_data[split_name] = df

        if not paired_df.empty:
            paired_out = config.PROCESSED_DATA_DIR / f"{split_name}_paired.parquet"
            paired_df.to_parquet(paired_out, index=False)
            splits_data[f"{split_name}_paired"] = paired_df

        summary_rows.append({
            "Split": split_name.capitalize(),
            "Total_Functions": len(df),
            "Vuln_Count (y=1)": int(df["target"].sum()),
            "Vuln_Ratio (%)": round(df["target"].mean() * 100, 2),
            "Paired_Functions": len(paired_df) * 2,
            "Unique_Projects": int(df["project"].nunique()),
            "Median_Tokens": int(df["bpe_length"].median()),
            "Mean_Tokens": round(df["bpe_length"].mean(), 1),
            "Exceeds_512 (%)": round(df["exceeds_512"].mean() * 100, 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = config.TABLE_DIR / "data_summary_stats.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\n=== PrimeVul Dataset Quality & Split Summary ===")
    print(summary_df.to_string(index=False))
    print(f"Saved summary statistics to: {summary_csv}")

    if len(splits_data["train"]) >= 10:
        plot_token_length_distribution(splits_data["train"])

    return splits_data


def plot_token_length_distribution(train_df: pd.DataFrame) -> None:
    """Generates token-length distribution plot across vulnerability classes."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    plot_df = train_df.copy()
    plot_df["bpe_clipped"] = np.clip(plot_df["bpe_length"], 0, 2500)
    plot_df["Class"] = plot_df["target"].map({0: "Safe (y=0)", 1: "Vulnerable (y=1)"})

    sns.histplot(
        data=plot_df,
        x="bpe_clipped",
        hue="Class",
        bins=min(50, max(10, len(plot_df) // 2)),
        kde=(len(plot_df) >= 20),
        stat="density",
        common_norm=False,
        palette={"Safe (y=0)": "#2b5c8f", "Vulnerable (y=1)": "#d95f02"},
        alpha=0.35,
        ax=ax,
    )
    pct_exceed = train_df["exceeds_512"].mean() * 100
    ax.axvline(
        config.CHUNK_SIZE,
        color="#c0392b",
        linestyle="--",
        linewidth=2,
        label=f"Single-Window Limit ($L=512$, {pct_exceed:.1f}% exceed)",
    )
    ax.set_title("Subword Token Length Distribution by Class (PrimeVul Train Split)")
    ax.set_xlabel("Subword Token Length (Clipped at 2,500)")
    ax.set_ylabel("Density")
    ax.legend(frameon=True)
    plt.tight_layout()

    fig.savefig(config.FIGURE_DIR / "token_length_distribution.pdf", dpi=300)
    fig.savefig(config.FIGURE_DIR / "token_length_distribution.png", dpi=300)
    plt.close(fig)
