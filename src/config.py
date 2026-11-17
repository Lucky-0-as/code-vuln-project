"""
Centralized configuration for PrimeVul C/C++ Vulnerability Detection.
Automatically adapts paths and hardware settings for both Kaggle GPU environments
(/kaggle/working) and local workstations.
"""

import os
import random
from pathlib import Path
import numpy as np
import torch

# Detect Kaggle environment automatically
IS_KAGGLE = Path("/kaggle/working").exists()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORK_ROOT = Path("/kaggle/working/vuln-detect") if IS_KAGGLE else PROJECT_ROOT

DATA_DIR = WORK_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANONYMIZED_DATA_DIR = DATA_DIR / "anonymized"
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"

OUTPUT_DIR = WORK_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ANONYMIZED_DATA_DIR,
    CHECKPOINT_DIR,
    PREDICTION_DIR,
    FIGURE_DIR,
    TABLE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Reproducibility Seed
SEED = 26210029
TARGET_RECALL = 0.80

# PrimeVul Data Access Configuration
# Users can point PRIMEVUL_RAW_DIR to a Kaggle dataset input folder (e.g. /kaggle/input/primevul)
# or download via the PrimeVul Google Drive / HuggingFace links documented in README.md.
PRIMEVUL_RAW_DIR = Path(
    os.environ.get(
        "PRIMEVUL_RAW_DIR",
        "/kaggle/input/primevul" if Path("/kaggle/input/primevul").exists() else str(RAW_DATA_DIR),
    )
)

PRIMEVUL_FILES = {
    "train": "primevul_train.jsonl",
    "validation": "primevul_valid.jsonl",
    "test": "primevul_test.jsonl",
    "train_paired": "primevul_train_paired.jsonl",
    "validation_paired": "primevul_valid_paired.jsonl",
    "test_paired": "primevul_test_paired.jsonl",
}

# Upgrade #1: Hierarchical Multi-Chunk Windowing (covers up to 1,280+ tokens)
CHUNK_SIZE = 512
CHUNK_STRIDE = 256
MAX_CHUNKS = 4  # 4 overlapping chunks of 512 tokens with stride 256 -> 1,280 tokens
HEAD_TOKENS = 254
TAIL_TOKENS = 256

# Upgrade #3: AST + Data-Flow Relational Edge Types
# 0 = No edge, 1 = AST Parent-Child, 2 = Variable Def-Use Data-Flow, 3 = Control-Flow Block
NUM_EDGE_TYPES = 4

# Upgrade #4: Layer-Weighted Multi-Head Attention Pooling
POOLING_NUM_HEADS = 8

# Upgrade #2 & #5: Multi-Task & Siamese Invariance Loss Hyperparameters
LAMBDA_SIAMESE_JSD = 0.30   # Weight for symmetric JSD invariance under variable renaming
LAMBDA_SIAMESE_COS = 0.10   # Weight for cosine embedding alignment
BETA_LINE_MULTITASK = 0.50  # Weight for line/token-level bug localization loss (PrimeVul_paired)
FOCAL_GAMMA = 2.0           # Focal modulation factor for 97:3 class imbalance
POS_CLASS_WEIGHT = 6.0      # Positive-class upweighting for imbalanced PrimeVul splits

# Classical Baseline Hyperparameters
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MAX_FEATURES = 50000
TFIDF_MIN_DF = 3
LOGREG_C_GRID = [0.01, 0.1, 1.0, 10.0]
LOGREG_MAX_ITER = 1000

# Backbone Registry
MODEL_REGISTRY = {
    "unixcoder": "microsoft/unixcoder-base",
    "codebert": "microsoft/codebert-base",
}

# Training Hyperparameters (tuned for Kaggle T4/P100 or RTX GPUs)
TRAIN_BATCH_SIZE = 8 if torch.cuda.is_available() else 2
EVAL_BATCH_SIZE = 16 if torch.cuda.is_available() else 4
GRAD_ACCUM_STEPS = 4
DEFAULT_LR = 2e-5
HEAD_LR = 1e-4              # Slightly higher LR for newly initialized pooling/aggregator heads
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 4
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0

# Hardware Acceleration Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16
NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 2)


def set_seed(seed: int = SEED) -> None:
    """Locks random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
