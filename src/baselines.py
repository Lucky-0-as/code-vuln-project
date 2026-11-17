"""
Classical ML Baselines adapted for PrimeVul (C/C++):
  - Tier 1: Structural Complexity Heuristics + Class-Balanced L2 Logistic Regression
  - Tier 2: Code-Aware TF-IDF (1-3 N-grams) + Class-Balanced L2 Logistic Regression
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.metrics import evaluate_paired_predictions, evaluate_predictions, find_best_threshold

CODE_TOKEN_PATTERN = (
    r"[A-Za-z_][A-Za-z0-9_]*|::|->|<<|>>|<=|>=|==|!=|&&|\|\||"
    r"[\+\-\*/%&|\^~<>!=]=?|[\*\[\]\(\)\{\};,\.]"
)
STRUCTURAL_COLS = ["loc", "bpe_length", "branch_count", "pointer_count"]


def build_structural_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extracts 4 complexity metrics + log1p transforms to handle right-skewness."""
    raw = df[STRUCTURAL_COLS].to_numpy(dtype=np.float64)
    log_raw = np.log1p(np.maximum(raw, 0.0))
    return np.hstack([raw, log_raw])


def run_all_baselines() -> pd.DataFrame:
    """Trains and evaluates Tier 1 and Tier 2 baselines on processed PrimeVul splits."""
    config.set_seed()
    train_df = pd.read_parquet(config.PROCESSED_DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(config.PROCESSED_DATA_DIR / "validation.parquet")
    test_df = pd.read_parquet(config.PROCESSED_DATA_DIR / "test.parquet")

    y_train = train_df["target"].to_numpy()
    y_val = val_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    results = []
    paired_results = []
    paired_test_path = config.PROCESSED_DATA_DIR / "test_paired.parquet"
    paired_df = pd.read_parquet(paired_test_path) if paired_test_path.exists() else None

    # --- Tier 1: Complexity Heuristics ---
    print("\n--- Running Baseline 1: Structural Complexity Heuristics ---")
    X_tr_s = build_structural_matrix(train_df)
    X_va_s = build_structural_matrix(val_df)
    X_te_s = build_structural_matrix(test_df)

    best_b1_pipe, best_b1_c, best_b1_thresh, best_b1_score = None, 1.0, 0.5, -1.0
    for c_val in config.LOGREG_C_GRID:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=c_val,
                class_weight="balanced",
                max_iter=config.LOGREG_MAX_ITER,
                random_state=config.SEED,
            )),
        ])
        pipe.fit(X_tr_s, y_train)
        val_probs = pipe.predict_proba(X_va_s)[:, 1]
        thresh = find_best_threshold(y_val, val_probs, optimize_for="vuln_f1")
        m = evaluate_predictions(y_val, val_probs, threshold=thresh)
        if m["Vuln_F1 (%)"] > best_b1_score:
            best_b1_score = m["Vuln_F1 (%)"]
            best_b1_pipe = pipe
            best_b1_c = c_val
            best_b1_thresh = thresh

    b1_val_probs = best_b1_pipe.predict_proba(X_va_s)[:, 1]
    b1_test_probs = best_b1_pipe.predict_proba(X_te_s)[:, 1]
    results.append(evaluate_predictions(
        y_val, b1_val_probs, best_b1_thresh, f"Baseline 1: Complexity (C={best_b1_c})", "Validation"
    ))
    results.append(evaluate_predictions(
        y_test, b1_test_probs, best_b1_thresh, f"Baseline 1: Complexity (C={best_b1_c})", "Test"
    ))
    joblib.dump(
        {"pipeline": best_b1_pipe, "threshold": best_b1_thresh},
        config.CHECKPOINT_DIR / "baseline1_complexity.joblib",
    )

    # --- Tier 2: TF-IDF (1-3 N-grams) + Class-Balanced L2 LogReg ---
    print("\n--- Running Baseline 2: TF-IDF (1-3 N-grams) + L2 LogReg ---")
    min_df = 1 if len(train_df) < 100 else config.TFIDF_MIN_DF
    vec = TfidfVectorizer(
        ngram_range=config.TFIDF_NGRAM_RANGE,
        max_features=config.TFIDF_MAX_FEATURES,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=CODE_TOKEN_PATTERN,
    )
    X_tr_t = vec.fit_transform(train_df["func_clean"])
    X_va_t = vec.transform(val_df["func_clean"])
    X_te_t = vec.transform(test_df["func_clean"])

    best_b2_clf, best_b2_c, best_b2_thresh, best_b2_score = None, 1.0, 0.5, -1.0
    for c_val in config.LOGREG_C_GRID:
        clf = LogisticRegression(
            C=c_val,
            class_weight="balanced",
            max_iter=config.LOGREG_MAX_ITER,
            random_state=config.SEED,
        )
        clf.fit(X_tr_t, y_train)
        val_probs = clf.predict_proba(X_va_t)[:, 1]
        thresh = find_best_threshold(y_val, val_probs, optimize_for="vuln_f1")
        m = evaluate_predictions(y_val, val_probs, threshold=thresh)
        if m["Vuln_F1 (%)"] > best_b2_score:
            best_b2_score = m["Vuln_F1 (%)"]
            best_b2_clf = clf
            best_b2_c = c_val
            best_b2_thresh = thresh

    b2_val_probs = best_b2_clf.predict_proba(X_va_t)[:, 1]
    b2_test_probs = best_b2_clf.predict_proba(X_te_t)[:, 1]
    results.append(evaluate_predictions(
        y_val, b2_val_probs, best_b2_thresh, f"Baseline 2: TF-IDF + LogReg (C={best_b2_c})", "Validation"
    ))
    results.append(evaluate_predictions(
        y_test, b2_test_probs, best_b2_thresh, f"Baseline 2: TF-IDF + LogReg (C={best_b2_c})", "Test"
    ))

    b2_pipe = Pipeline([("tfidf", vec), ("clf", best_b2_clf)])
    joblib.dump(
        {"pipeline": b2_pipe, "threshold": best_b2_thresh},
        config.CHECKPOINT_DIR / "baseline2_tfidf.joblib",
    )

    if paired_df is not None and not paired_df.empty:
        v_probs = b2_pipe.predict_proba(paired_df["vuln_func"])[:, 1]
        s_probs = b2_pipe.predict_proba(paired_df["safe_func"])[:, 1]
        paired_results.append(
            evaluate_paired_predictions(v_probs, s_probs, best_b2_thresh, "Baseline 2: TF-IDF + LogReg")
        )
        pd.DataFrame(paired_results).to_csv(config.TABLE_DIR / "paired_baseline_metrics.csv", index=False)

    np.savez(
        config.PREDICTION_DIR / "baselines_test_preds.npz",
        y_test=y_test,
        exceeds_512=test_df["exceeds_512"].to_numpy(),
        b1_probs=b1_test_probs,
        b1_thresh=best_b1_thresh,
        b2_probs=b2_test_probs,
        b2_thresh=best_b2_thresh,
    )

    res_df = pd.DataFrame(results)
    res_df.to_csv(config.TABLE_DIR / "baseline_metrics.csv", index=False)
    print("\n================ CLASSICAL BASELINES SUMMARY ================")
    print(res_df.to_string(index=False))
    return res_df
