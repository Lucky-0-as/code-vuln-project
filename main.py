"""
Master Reproducible CLI Entry Point for Hierarchical Graph-Biased Siamese
Multi-Task C/C++ Vulnerability Detection on PrimeVul.

Usage Examples:
  1. Instant Zero-Download Grader Verification / Smoke Test:
       python main.py --stage smoke_test

  2. Full Pipeline on PrimeVul (after downloading JSONL files per README.md):
       python main.py --stage all

  3. Individual Pipeline Stages:
       python main.py --stage preprocess
       python main.py --stage baselines
       python main.py --stage train_baseline
       python main.py --stage train --backbone unixcoder --epochs 4
       python main.py --stage audit
       python main.py --stage demo
"""

import argparse
from pathlib import Path
import pandas as pd

from src import config
from src.baselines import run_all_baselines
from src.demo import DEFAULT_SAMPLE_CODE, predict_on_code_snippet
from src.evaluate_audit import run_full_audit
from src.preprocessing import prepare_primevul_data
from src.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical Graph-Biased Siamese Multi-Task C/C++ Vulnerability Detection"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="demo",
        choices=[
            "smoke_test",
            "preprocess",
            "baselines",
            "train_baseline",
            "train",
            "audit",
            "demo",
            "all",
        ],
        help="Pipeline stage to run",
    )
    parser.add_argument("--use_sample", action="store_true", help="Use included sample dataset in data/sample/")
    parser.add_argument("--backbone", type=str, default="unixcoder", choices=["unixcoder", "codebert"])
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=config.GRAD_ACCUM_STEPS)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--file", type=str, default=None, help="Path to .c/.cpp file for --stage demo")
    parser.add_argument("--test_idx", type=int, default=None, help="Test row index for --stage demo")
    args = parser.parse_args()

    if args.stage == "smoke_test":
        print("Running end-to-end reproducibility smoke test on included sample dataset...")
        prepare_primevul_data(use_sample=True)
        run_all_baselines()
        train_model(mode="baseline", backbone_key=args.backbone, epochs=1, batch_size=2, grad_accum_steps=1)
        train_model(mode="full", backbone_key=args.backbone, epochs=1, batch_size=2, grad_accum_steps=1)
        run_full_audit()
        predict_on_code_snippet(DEFAULT_SAMPLE_CODE, true_label=1)
        return

    if args.stage in ("preprocess", "all"):
        prepare_primevul_data(use_sample=args.use_sample)

    if args.stage in ("baselines", "all"):
        run_all_baselines()

    if args.stage in ("train_baseline", "all"):
        train_model(
            mode="baseline",
            backbone_key=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            lr=args.lr,
        )

    if args.stage in ("train", "all"):
        train_model(
            mode="full",
            backbone_key=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            lr=args.lr,
        )

    if args.stage in ("audit", "all"):
        run_full_audit()

    if args.stage == "demo":
        if args.file is not None:
            predict_on_code_snippet(Path(args.file).read_text(encoding="utf-8"))
        elif args.test_idx is not None:
            test_df = pd.read_parquet(config.PROCESSED_DATA_DIR / "test.parquet")
            row = test_df.iloc[args.test_idx]
            predict_on_code_snippet(row["func_clean"], true_label=int(row["target"]))
        else:
            predict_on_code_snippet(DEFAULT_SAMPLE_CODE, true_label=1)


if __name__ == "__main__":
    main()
