"""
Comprehensive Evaluation Metrics for Imbalanced & Paired Vulnerability Detection:
- Standard & Imbalance-Aware Classification: Accuracy, Macro F1, Vuln F1, Precision,
  Recall, PR-AUC, ROC-AUC, and False Positive Rate at 80% Recall (FPR@R80)
- PrimeVul Paired Benchmark Metrics: P-C (Both Correct), P-V (Both Vulnerable),
  P-B (Both Benign), P-R (Reversed), and Pairwise Ranking Accuracy
- Counterfactual Shortcut Audit: Prediction Flip Rate & Mean Probability Shift
- Line-Level Localization: Token/Line F1 & IoU against Patch Diff Masks
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config


def compute_fpr_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = config.TARGET_RECALL,
) -> tuple[float, float]:
    """
    Computes the minimum False Positive Rate (FPR) at a threshold achieving
    at least `target_recall` (default TPR >= 0.80).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.50

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_indices = np.where(tpr >= target_recall)[0]
    if len(valid_indices) == 0:
        return 1.0, 0.0
    idx = valid_indices[0]
    return float(fpr[idx]), float(thresholds[idx])


def find_best_threshold(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    optimize_for: str = "vuln_f1",
) -> float:
    """
    Tunes the binary decision threshold tau in [0.10, 0.90] strictly on the
    Validation split to maximize Vuln F1 (or Macro F1).
    """
    y_val = np.asarray(y_val, dtype=int)
    val_probs = np.asarray(val_probs, dtype=float)
    best_thresh = 0.50
    best_score = -1.0

    avg_mode = "binary" if optimize_for == "vuln_f1" else "macro"
    for thresh in np.linspace(0.10, 0.90, 161):
        preds = (val_probs >= thresh).astype(int)
        score = f1_score(y_val, preds, average=avg_mode, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
    return best_thresh


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "Model",
    split_name: str = "Test",
) -> dict:
    """Computes full classification & ranking metrics for binary vulnerability detection."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    binary_f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    has_both_classes = len(np.unique(y_true)) >= 2
    pr_auc = average_precision_score(y_true, y_prob) if has_both_classes else 0.0
    roc_auc = roc_auc_score(y_true, y_prob) if has_both_classes else 0.5
    fpr_r80, _ = compute_fpr_at_recall(y_true, y_prob, config.TARGET_RECALL)

    return {
        "Model": model_name,
        "Split": split_name,
        "Threshold": round(threshold, 4),
        "Accuracy (%)": round(acc * 100, 2),
        "Macro_F1 (%)": round(macro_f1 * 100, 2),
        "Vuln_F1 (%)": round(binary_f1 * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "PR_AUC (%)": round(pr_auc * 100, 2),
        "ROC_AUC (%)": round(roc_auc * 100, 2),
        "FPR@R80 (%)": round(fpr_r80 * 100, 2),
    }


def evaluate_paired_predictions(
    vuln_probs: np.ndarray,
    safe_probs: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "Model",
) -> dict:
    """
    Computes PrimeVul Pairwise Benchmark Metrics (Ding et al., 2024) on matched
    (pre-patch vulnerable, post-patch fixed) function pairs:
      - P-C (%): Both Correct (vuln_pred == 1 AND safe_pred == 0)
      - P-V (%): Both Predicted Vulnerable (vuln_pred == 1 AND safe_pred == 1)
      - P-B (%): Both Predicted Benign (vuln_pred == 0 AND safe_pred == 0)
      - P-R (%): Reversed / Inverted (vuln_pred == 0 AND safe_pred == 1)
      - Pairwise_Rank_Acc (%): Fraction of pairs where P(vuln) > P(safe)
    """
    vuln_probs = np.asarray(vuln_probs, dtype=float)
    safe_probs = np.asarray(safe_probs, dtype=float)
    n_pairs = len(vuln_probs)
    if n_pairs == 0:
        return {}

    v_pred = (vuln_probs >= threshold).astype(int)
    s_pred = (safe_probs >= threshold).astype(int)

    p_c = np.mean((v_pred == 1) & (s_pred == 0)) * 100.0
    p_v = np.mean((v_pred == 1) & (s_pred == 1)) * 100.0
    p_b = np.mean((v_pred == 0) & (s_pred == 0)) * 100.0
    p_r = np.mean((v_pred == 0) & (s_pred == 1)) * 100.0
    rank_acc = np.mean(vuln_probs > safe_probs) * 100.0
    mean_margin = np.mean(vuln_probs - safe_probs) * 100.0

    return {
        "Model": model_name,
        "Num_Pairs": int(n_pairs),
        "P-C_BothCorrect (%)": round(float(p_c), 2),
        "P-V_BothVuln (%)": round(float(p_v), 2),
        "P-B_BothBenign (%)": round(float(p_b), 2),
        "P-R_Reversed (%)": round(float(p_r), 2),
        "Pairwise_Rank_Acc (%)": round(float(rank_acc), 2),
        "Mean_Prob_Margin (%)": round(float(mean_margin), 2),
    }


def compute_flip_rate(orig_preds: np.ndarray, anon_preds: np.ndarray) -> float:
    """Computes % of functions whose binary verdict flips after AST variable renaming."""
    orig_preds = np.asarray(orig_preds, dtype=int)
    anon_preds = np.asarray(anon_preds, dtype=int)
    return float(np.mean(orig_preds != anon_preds) * 100.0)
