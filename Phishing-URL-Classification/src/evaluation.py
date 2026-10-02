"""
evaluation.py
-------------
Reusable evaluation utilities:
  - train_and_evaluate()  : train a model, time it, collect all metrics
  - latency_test()        : measure per-sample inference latency
  - support_vector_analysis() : extract and explain SV statistics
  - build_results_df()    : aggregate results into a comparison DataFrame
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
LATENCY_N_SAMPLES = 1000
LATENCY_REPEATS = 5   # repeat to reduce timing noise, report minimum


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def train_and_evaluate(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Fit model on X_train, evaluate on X_test.
    Returns a dict of all metrics.

    Parameters
    ----------
    name   : human-readable model name
    model  : unfitted scikit-learn estimator
    X_train, y_train : training data
    X_test, y_test   : held-out test data

    Returns
    -------
    dict with keys: name, accuracy, precision, recall, f1,
                    train_time, pred_time, latency_ms,
                    confusion_mat, classification_report,
                    y_pred, model (fitted)
    """
    # --- Train ---
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # --- Predict ---
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - t0

    latency_ms = (pred_time / len(X_test)) * 1000.0

    # --- Metrics ---
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    cr   = classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Phishing/Malicious"],
        zero_division=0,
    )

    result = {
        "name":         name,
        "accuracy":     acc,
        "precision":    prec,
        "recall":       rec,
        "f1":           f1,
        "train_time":   train_time,
        "pred_time":    pred_time,
        "latency_ms":   latency_ms,
        "confusion_mat": cm,
        "class_report": cr,
        "y_pred":       y_pred,
        "model":        model,
    }

    logger.info(
        f"{name:<25} | Acc={acc:.4f} | P={prec:.4f} | R={rec:.4f} "
        f"| F1={f1:.4f} | TrainT={train_time:.3f}s | PredT={pred_time:.4f}s"
    )
    return result


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def latency_test(
    models_dict: dict,
    X_test: np.ndarray,
    n_samples: int = LATENCY_N_SAMPLES,
    repeats: int = LATENCY_REPEATS,
) -> pd.DataFrame:
    """
    Measure inference latency for `n_samples` from X_test.

    Parameters
    ----------
    models_dict : {name: fitted_model}
    X_test      : scaled test array
    n_samples   : number of samples to time (default 1000)
    repeats     : number of repetitions; minimum time is reported

    Returns
    -------
    DataFrame with columns:
        Model | N_Samples | Total_Time_s | Avg_Time_ms_per_URL
    """
    available = len(X_test)
    if available >= n_samples:
        X_lat = X_test[:n_samples]
        actual_n = n_samples
    else:
        logger.warning(
            f"Only {available} test samples available -- "
            f"using all for latency test (requested {n_samples})."
        )
        X_lat = X_test
        actual_n = available

    rows = []
    for name, model in models_dict.items():
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model.predict(X_lat)
            times.append(time.perf_counter() - t0)
        best_time = min(times)
        avg_ms = (best_time / actual_n) * 1000.0
        rows.append({
            "Model":                name,
            "N_Samples":            actual_n,
            "Total_Time_s":         round(best_time, 6),
            "Avg_Time_ms_per_URL":  round(avg_ms, 6),
        })
        logger.info(
            f"Latency [{name}]: {best_time:.6f}s total | "
            f"{avg_ms:.6f} ms/URL ({actual_n} samples)"
        )

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Support-vector analysis
# ---------------------------------------------------------------------------

def support_vector_analysis(model, model_name: str, n_train: int) -> dict:
    """
    Extract support vector statistics from a fitted SVC.

    Parameters
    ----------
    model      : fitted SVC (or Pipeline whose last step is SVC)
    model_name : human-readable name for logging
    n_train    : number of training samples

    Returns
    -------
    dict with sv statistics and explanation text
    """
    # Handle Pipeline wrapper
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        svc = model.named_steps.get("svc", None)
        if svc is None:
            raise ValueError("Pipeline does not have a step named 'svc'.")
    else:
        svc = model

    support_indices  = svc.support_
    support_vectors  = svc.support_vectors_
    n_support        = svc.n_support_

    total_sv    = int(np.sum(n_support))
    sv_fraction = total_sv / n_train

    # Per-class breakdown
    sv_per_class = {
        "Legitimate (0)":        int(n_support[0]) if len(n_support) > 0 else None,
        "Malicious/Phishing (1)": int(n_support[1]) if len(n_support) > 1 else None,
    }

    explanation = (
        f"\nSUPPORT VECTOR ANALYSIS -- {model_name}\n"
        "=" * 60 + "\n"
        f"  Training samples          : {n_train}\n"
        f"  Total support vectors     : {total_sv}\n"
        f"  Support vector fraction   : {sv_fraction:.4f} ({sv_fraction*100:.2f}%)\n"
        f"  Per-class SV counts       : {sv_per_class}\n"
        "\n"
        "INTERPRETATION\n"
        "--------------\n"
        f"  A support-vector fraction of {sv_fraction*100:.1f}% means that "
        f"{sv_fraction*100:.1f}% of training\n"
        "  samples lie on or within the margin boundary.\n"
        "\n"
        "  Large SV fraction implications:\n"
        "    - Higher model complexity (more memory, slower prediction)\n"
        "    - Decision boundary depends on many training points\n"
        "    - Prediction cost: O(n_sv * n_features) per sample\n"
        "    - May indicate overlap between classes or soft-margin tolerance\n"
        "\n"
        "  Small SV fraction implications:\n"
        "    - Clean, wide margin between classes\n"
        "    - Better generalisation (fewer training points define boundary)\n"
        "    - Faster prediction\n"
        "\n"
        "  NOTE: Exact identification of margin violators vs. margin-support\n"
        "  points requires inspecting decision_function() values:\n"
        "    0 < |dec_f| < 1  -> inside / violating margin (soft-margin)\n"
        "    |dec_f| ~= 1     -> exactly on the margin\n"
        "  This analysis is possible only when probability=False (default).\n"
    )

    stats = {
        "model_name":     model_name,
        "n_train":        n_train,
        "total_sv":       total_sv,
        "sv_fraction":    sv_fraction,
        "sv_per_class":   sv_per_class,
        "n_support":      n_support.tolist(),
        "support_indices_sample": support_indices[:10].tolist(),
        "explanation":    explanation,
    }

    logger.info(
        f"SV Analysis [{model_name}]: {total_sv} SVs / {n_train} train "
        f"= {sv_fraction:.4f}"
    )
    print(explanation)
    return stats


# ---------------------------------------------------------------------------
# False-positive / False-negative analysis
# ---------------------------------------------------------------------------

def fp_fn_analysis(cm: np.ndarray) -> dict:
    """
    Extract FP/FN counts and rates from a 2x2 confusion matrix.

    Matrix layout (scikit-learn default):
        [[TN  FP]
         [FN  TP]]

    Where:
        Negative class = Legitimate (0)
        Positive class = Malicious / Phishing (1)
    """
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp),
        "FPR": fpr, "FNR": fnr,
    }


# ---------------------------------------------------------------------------
# Build summary DataFrames
# ---------------------------------------------------------------------------

def build_results_df(results_list: list[dict]) -> pd.DataFrame:
    """
    Convert a list of result dicts (from train_and_evaluate) to a
    clean comparison DataFrame.
    """
    rows = []
    for r in results_list:
        rows.append({
            "Model":            r["name"],
            "Accuracy":         round(r["accuracy"],  4),
            "Precision":        round(r["precision"], 4),
            "Recall":           round(r["recall"],    4),
            "F1":               round(r["f1"],        4),
            "Train_Time_s":     round(r["train_time"], 4),
            "Pred_Time_s":      round(r["pred_time"],  6),
            "Latency_ms/URL":   round(r["latency_ms"], 6),
        })
    return pd.DataFrame(rows)


def print_results_table(df: pd.DataFrame, title: str = "Model Comparison"):
    sep = "=" * 110
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(df.to_string(index=False))
    print(sep)
