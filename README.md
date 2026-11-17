# Hierarchical Graph-Biased Siamese Multi-Task Vulnerability Detection for C/C++

A program-analysis pipeline for **function-level and line-level C/C++ vulnerability detection** on the **PrimeVul** benchmark (`DLVulDet/PrimeVul`, arXiv:2403.18624).

This repository extends pretrained code encoders (`UniXcoder` / `CodeBERT`) designed to overcome context-window truncation, variable-name shortcut reliance, and coarse function-only supervision:

1. **Hierarchical Chunk-Attention Encoder (`src/model.py`, `src/dataset.py`):** Splits long C/C++ functions into up to $K=4$ overlapping 512-token windows (`stride = 256`, covering **1,280+ tokens** plus function epilogue preservation) and aggregates chunk embeddings using a **2-layer Transformer Encoder + Gated Attention Pooling**.
2. **Siamese Counterfactual Invariance Head (`src/model.py`, `src/ast_graph.py`):** Uses `tree-sitter-cpp` to rewrite user-defined variables and functions into canonical placeholders (`var_1`, `func_1`) while preserving C/C++ keywords and security APIs (`memcpy`, `malloc`, `free`). Trains a dual-branch shared encoder with symmetric **Jensen-Shannon Divergence (JSD) + Cosine Alignment** to eliminate lexical shortcut reliance.
3. **AST & Data-Flow Edge-Biased Self-Attention (`src/ast_graph.py`, `src/model.py`):** Extracts **AST Parent-Child syntactic edges**, **Variable Definition-Use (Def-Use) data-flow edges**, and **Control-Flow guard edges** via `tree-sitter-cpp`, injecting a learned per-head relational bias matrix $B_{\text{AST/DFG}} \in \mathbb{R}^{H \times L \times L}$ directly into every Transformer self-attention layer.
4. **Layer-Weighted Multi-Head Attention Pooling (`src/model.py`):** Replaces single-layer `[CLS]` pooling with learnable softmax weights across all 12 Transformer layers ($H = \sum_{l=1}^{12} \alpha_l H^{(l)}$) followed by **8-head token attention pooling**.
5. **Joint Function + Line-Level Multi-Task Head (`src/preprocessing.py`, `src/model.py`):** Diffs matched pre-patch (vulnerable) and post-patch (fixed) commits from `PrimeVul_paired` to generate exact line-level vulnerability masks, training a token/line localization head jointly with class-weighted Focal Cross-Entropy.

---

## Repository Structure

```text
primevul-vuln-transformers/
├── main.py                              # Master CLI entry point for all pipeline stages
├── requirements.txt                     # Pinned Python dependencies
├── .gitignore                           # Excludes large raw/processed data & weights; keeps sample data
├── data/
│   └── sample/
│       └── primevul_sample.jsonl        # Permitted 12-function C/C++ sample (6 vuln + 6 paired fixes)
└── src/
    ├── __init__.py
    ├── config.py                        # Centralized hyperparameters, Kaggle/local paths & deterministic seed
    ├── preprocessing.py                 # Line-preserving comment stripping, paired diff masks & EDA
    ├── ast_graph.py                     # tree-sitter-cpp AST anonymizer & Def-Use/AST graph builder
    ├── dataset.py                       # Multi-chunk + graph edge + line-mask + Siamese PyTorch Dataset
    ├── model.py                         # Unified 5-upgrade architecture & multi-task focal/JSD loss
    ├── baselines.py                     # Tier 1 (Complexity) & Tier 2 (TF-IDF + L2 LogReg) baselines
    ├── train.py                         # Multi-GPU DataParallel / BF16 / FP16 training & validation loop
    ├── metrics.py                       # PR-AUC, Vuln F1, FPR@R80, Pairwise P-C/P-R & Flip Rate metrics
    ├── evaluate_audit.py                # Test evaluation, PrimeVul Paired audit, shortcut audit & plots
    └── demo.py                          # Live single-snippet inference, flip audit & line localization
```

---

## Environment Setup

### Local Workstation (Conda / Python 3.10+)
```bash
conda create -n vuln-detect python=3.10 -y
conda activate vuln-detect
pip install -r requirements.txt
```

## Dataset Access & Reproducibility Notes

### Option A: Immediate Zero-Download Verification
To allow immediate verification without downloading much data, this repository includes a permitted, self-contained sample dataset at `data/sample/primevul_sample.jsonl` containing **12 real-world-style C/C++ functions** (6 vulnerable functions paired with 6 post-patch security fixes across buffer overflows, use-after-free, integer overflow, out-of-bounds read, and null pointer dereference).

Run the complete end-to-end smoke test (preprocessing $\rightarrow$ classical baselines $\rightarrow$ single-window baseline $\rightarrow$ 5-upgrade model $\rightarrow$ paired & shortcut audits $\rightarrow$ live demo) in ~60 seconds:
```bash
python main.py --stage smoke_test
```

### Option B: Full PrimeVul Benchmark (`DLVulDet/PrimeVul`)
1. Obtain the official PrimeVul dataset releases from the PrimeVul repository ([https://github.com/DLVulDet/PrimeVul](https://github.com/DLVulDet/PrimeVul), Ding et al., arXiv:2403.18624):
   - Unpaired splits: `primevul_train.jsonl`, `primevul_valid.jsonl`, `primevul_test.jsonl`
   - Paired commit splits: `primevul_train_paired.jsonl`, `primevul_valid_paired.jsonl`, `primevul_test_paired.jsonl`
2. Place the `.jsonl` files inside `data/raw/` (or attach them as a Kaggle Dataset at `/kaggle/input/primevul`, or set `export PRIMEVUL_RAW_DIR=/path/to/primevul`).
3. All random seeds, splits, validation thresholds, and `tree-sitter` traversal orders are locked deterministically in `src/config.py`.

---

## Full Training, Evaluation & Testing Pipeline

### 1. Data Preprocessing, Line-Diff Mask Generation & EDA
Strips C/C++ comments while preserving exact line numbers, computes line-level vulnerability masks from `PrimeVul_paired` commit diffs, and saves Parquet splits:
```bash
python main.py --stage preprocess
```
*(Add `--use_sample` to run on `data/sample/primevul_sample.jsonl`.)*

### 2. Classical ML Baselines (Tier 1 Complexity & Tier 2 TF-IDF)
Trains class-balanced L2 Logistic Regression on static complexity heuristics and code-aware 1–3 token n-grams:
```bash
python main.py --stage baselines
```

### 3. Single-Window Reference Model (Ablation Baseline)
Trains standard single-window (`Head+Tail`, 512 tokens) `UniXcoder` on PrimeVul for direct ablation comparison:
```bash
python main.py --stage train_baseline --backbone unixcoder --epochs 4
```

### 4. Train Full 5-Upgrade Architecture
Trains `UniXcoder` with Hierarchical Multi-Chunk Aggregation, AST/Def-Use Edge-Biased Attention, Layer-Weighted Multi-Head Pooling, Siamese Anonymization Invariance, and Joint Line-Level Multi-Task Supervision:
```bash
python main.py --stage train --backbone unixcoder --epochs 4 --batch_size 8 --grad_accum 4
```

### 5. Full Evaluation, Paired Benchmark & Counterfactual Shortcut Audit
Evaluates all trained models on:
- Locked chronological Test split (`PR-AUC`, `Vuln F1`, `Macro F1`, `ROC-AUC`, `FPR@R80`)
- `PrimeVul_paired` bug-vs-fix pairs (`P-C` Both Correct, `P-R` Reversed, `Pairwise_Rank_Acc`)
- Counterfactual AST identifier anonymization (`var_1`, `func_1`) shortcut audit (`Flip_Rate (%)`, `Delta_PR_AUC`)
- Short ($\le 512$ tokens) vs. Long ($> 512$ tokens) length stratification
```bash
python main.py --stage audit
```
Generated CSV tables are saved to `outputs/tables/` and plots are saved to `outputs/figures/`.

### 6. Live Interactive Inference & Line Localization Demo
Run vulnerability prediction, variable-renaming flip testing, and top-line bug localization on any C/C++ snippet:
```bash
# Built-in vulnerable C snippet
python main.py --stage demo

# Evaluate any custom .c or .cpp file
python main.py --stage demo --file path/to/snippet.cpp

# Evaluate a specific function from the processed test split
python main.py --stage demo --test_idx 0
```
