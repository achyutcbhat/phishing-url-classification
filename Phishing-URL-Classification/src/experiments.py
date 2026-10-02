"""
experiments.py
--------------
High-level experiment orchestration:
  1. run_baseline_models()          - Train/evaluate all baseline models
  2. run_gridsearch()               - SVM GridSearchCV
  3. run_poly_degree_sensitivity()  - Polynomial degree 2/3/4 comparison
  4. run_support_vector_analysis()  - SV count/fraction for best SVM
  5. run_latency_test()             - Inference latency benchmark
  6. run_feature_importance()       - Linear SVM feature importance
  7. select_final_model()           - Choose best model with justification

Each function returns its results and saves CSVs to results/tables/.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluation import (
    build_results_df,
    fp_fn_analysis,
    latency_test,
    print_results_table,
    support_vector_analysis,
    train_and_evaluate,
)
from .models import (
    build_pipeline,
    get_baseline_models,
    get_knn,
    get_linear_svm,
    get_logistic_regression,
    get_poly_svm,
    get_rbf_svm,
    get_svm_param_grid,
)

logger = logging.getLogger(__name__)
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Baseline models
# ---------------------------------------------------------------------------

def run_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tables_dir: Path,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Train and evaluate all baseline models on pre-scaled data.
    Saves CSV to tables_dir/baseline_comparison.csv.

    Returns (results_list, results_df)
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print("  BASELINE MODEL TRAINING & EVALUATION")
    print("=" * 70)

    models = get_baseline_models()
    results = []
    for name, model in models:
        r = train_and_evaluate(name, model, X_train, y_train, X_test, y_test)
        results.append(r)

    df = build_results_df(results)
    out = tables_dir / "baseline_comparison.csv"
    df.to_csv(out, index=False)
    logger.info(f"Baseline results saved: {out}")
    print_results_table(df, "BASELINE MODEL COMPARISON")

    # Print full classification reports
    print("\n--- Detailed Classification Reports ---")
    for r in results:
        print(f"\n{r['name']}")
        print(r["class_report"])

    return results, df


# ---------------------------------------------------------------------------
# 2. GridSearchCV
# ---------------------------------------------------------------------------

def run_gridsearch(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    tables_dir: Path,
    experiments_dir: Path,
    cv: int = 5,
    scoring: str = "accuracy",
) -> dict:
    """
    Run GridSearchCV over SVM kernels (linear, poly, rbf).

    Uses raw (unscaled) X_train so the Pipeline handles scaling internally
    per CV fold -- no data leakage.

    Saves:
      - experiments/gridsearch_cv_results.csv   (full CV results)
      - tables/gridsearch_best_summary.csv       (top 20 configs)
      - models/best_svm_pipeline.joblib          (best estimator)

    Scoring metric: accuracy
    Rationale: The classes are approximately balanced (6157 vs 4898 in
    this dataset).  Accuracy is interpretable and commonly reported.
    We additionally report precision, recall, F1 on the test set.

    Returns dict with best params, scores, confusion matrix, etc.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  GRIDSEARCHCV -- SVM HYPERPARAMETER TUNING")
    print(f"  cv={cv}  scoring='{scoring}'")
    print("=" * 70)
    print("  (This may take several minutes...)")

    # Pipeline: raw -> StandardScaler -> SVC
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(random_state=RANDOM_STATE)),
    ])

    param_grid = get_svm_param_grid()

    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=False,
    )

    t0 = time.perf_counter()
    gs.fit(X_train_raw, y_train)
    gs_time = time.perf_counter() - t0

    print(f"\n  GridSearchCV completed in {gs_time:.1f}s")

    # Best params / score
    best_params  = gs.best_params_
    best_cv_acc  = gs.best_score_
    best_est     = gs.best_estimator_

    print(f"\n  Best Parameters  : {best_params}")
    print(f"  Best CV Accuracy : {best_cv_acc:.6f}")

    # Evaluate best on test set
    t0 = time.perf_counter()
    y_pred = best_est.predict(X_test_raw)
    pred_time = time.perf_counter() - t0

    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_score, recall_score, classification_report,
    )
    test_acc  = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec  = recall_score(y_test, y_pred, zero_division=0)
    test_f1   = f1_score(y_test, y_pred, zero_division=0)
    test_cm   = confusion_matrix(y_test, y_pred)
    test_cr   = classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Phishing/Malicious"],
        zero_division=0,
    )
    fp_fn = fp_fn_analysis(test_cm)

    print(f"\n  TEST SET RESULTS (Best Estimator):")
    print(f"    Accuracy  : {test_acc:.6f}")
    print(f"    Precision : {test_prec:.6f}")
    print(f"    Recall    : {test_rec:.6f}")
    print(f"    F1-score  : {test_f1:.6f}")
    print(f"    Train time (GS total): {gs_time:.2f}s")
    print(f"    Pred time : {pred_time:.4f}s")
    print(f"\n  Confusion Matrix:\n{test_cm}")
    print(f"\n  FPR={fp_fn['FPR']:.4f}  FNR={fp_fn['FNR']:.4f}")
    print(f"\n  Classification Report:\n{test_cr}")

    # CV results DataFrame
    cv_df = pd.DataFrame(gs.cv_results_)
    cv_df_out = experiments_dir / "gridsearch_cv_results.csv"
    cv_df.to_csv(cv_df_out, index=False)
    logger.info(f"Full CV results saved: {cv_df_out}")

    # Top 20 summary
    relevant_cols = [c for c in cv_df.columns if c in [
        "param_svc__kernel", "param_svc__C", "param_svc__gamma",
        "param_svc__degree", "mean_test_score", "std_test_score",
        "rank_test_score", "mean_fit_time",
    ]]
    top20 = cv_df[relevant_cols].sort_values("rank_test_score").head(20)
    top20_out = tables_dir / "gridsearch_best_summary.csv"
    top20.to_csv(top20_out, index=False)
    print(f"\n  Top 20 GridSearchCV Configurations:")
    print(top20.to_string(index=False))

    # Save best model
    models_dir = tables_dir.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_out = models_dir / "best_svm_pipeline.joblib"
    joblib.dump(best_est, model_out)
    logger.info(f"Best model saved: {model_out}")

    return {
        "best_params":   best_params,
        "best_cv_acc":   best_cv_acc,
        "best_estimator": best_est,
        "gs_time":       gs_time,
        "test_acc":      test_acc,
        "test_precision": test_prec,
        "test_recall":   test_rec,
        "test_f1":       test_f1,
        "test_cm":       test_cm,
        "fp_fn":         fp_fn,
        "pred_time":     pred_time,
        "y_pred":        y_pred,
        "cv_results_df": cv_df,
        "top20_df":      top20,
        "class_report":  test_cr,
    }


# ---------------------------------------------------------------------------
# 3. Polynomial degree sensitivity
# ---------------------------------------------------------------------------

def run_poly_degree_sensitivity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tables_dir: Path,
) -> pd.DataFrame:
    """
    Compare SVM Polynomial kernel for degree = 2, 3, 4.
    Uses pre-scaled data.

    Explanation of degree effects:
    - degree=2 : Quadratic boundary. Low complexity, fast training,
                 risk of underfitting on complex class boundaries.
    - degree=3 : Cubic boundary. Good balance. Standard default.
    - degree=4 : Quartic boundary. Higher complexity, slower training,
                 risk of overfitting on small/noisy datasets.
    Increasing degree does NOT automatically improve performance --
    conclusions must be based on actual measured results.

    Returns DataFrame with per-degree metrics.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print("  POLYNOMIAL DEGREE SENSITIVITY ANALYSIS")
    print("=" * 70)

    degrees = [2, 3, 4]
    rows = []

    for deg in degrees:
        model = get_poly_svm(degree=deg)
        r = train_and_evaluate(
            f"Poly SVM deg={deg}", model,
            X_train, y_train, X_test, y_test,
        )
        # Support vectors
        n_sv = int(np.sum(r["model"].n_support_))
        rows.append({
            "Degree":             deg,
            "Accuracy":           round(r["accuracy"],  4),
            "Precision":          round(r["precision"], 4),
            "Recall":             round(r["recall"],    4),
            "F1":                 round(r["f1"],        4),
            "Train_Time_s":       round(r["train_time"], 4),
            "Pred_Time_s":        round(r["pred_time"],  6),
            "Latency_ms/URL":     round(r["latency_ms"], 6),
            "N_Support_Vectors":  n_sv,
        })
        print(f"  deg={deg} | Acc={r['accuracy']:.4f} | F1={r['f1']:.4f} "
              f"| TrainT={r['train_time']:.3f}s | SV={n_sv}")

    df = pd.DataFrame(rows)
    out = tables_dir / "poly_degree_sensitivity.csv"
    df.to_csv(out, index=False)
    print_results_table(df, "POLYNOMIAL DEGREE SENSITIVITY")
    logger.info(f"Poly-degree results saved: {out}")

    # Narrative
    print("\n  INTERPRETATION:")
    print("  - Degree 2 (quadratic): lowest complexity, fastest training.")
    print("  - Degree 3 (cubic)    : standard choice, moderate complexity.")
    print("  - Degree 4 (quartic)  : highest complexity, may overfit or")
    print("    require more training time. Performance depends on data geometry.")
    print("  Conclusions are drawn from the measured results above.")

    return df


# ---------------------------------------------------------------------------
# 4. Support-vector analysis
# ---------------------------------------------------------------------------

def run_support_vector_analysis(
    model,
    model_name: str,
    n_train: int,
    tables_dir: Path,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
) -> dict:
    """
    Full support-vector analysis including optional margin classification.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)

    stats = support_vector_analysis(model, model_name, n_train)

    # Optional: margin analysis via decision_function
    if X_train is not None and y_train is not None:
        try:
            from sklearn.pipeline import Pipeline
            svc = model.named_steps["svc"] if isinstance(model, Pipeline) else model
            # For Pipeline, we need to transform X first
            if isinstance(model, Pipeline):
                X_tr = model.named_steps["scaler"].transform(X_train)
            else:
                X_tr = X_train

            dec = svc.decision_function(X_tr)
            # Support vector indices (in training set)
            sv_idx = svc.support_
            sv_dec = dec[sv_idx]

            # Soft-margin: |decision_function| < 1 -> inside margin
            inside_margin = np.sum(np.abs(sv_dec) < 1.0)
            on_margin     = np.sum(np.abs(np.abs(sv_dec) - 1.0) < 0.01)

            stats["inside_margin_sv"] = int(inside_margin)
            stats["on_margin_sv"]     = int(on_margin)
            print(
                f"  Margin analysis:\n"
                f"    SVs with |dec_f| < 1 (inside/violating margin): {inside_margin}\n"
                f"    SVs with |dec_f| ~= 1 (exactly on margin):       {on_margin}\n"
                f"  (NOTE: For SVC with kernel, decision_function values\n"
                f"   in kernelised space; exact geometric interpretation\n"
                f"   requires care -- treat these as approximations.)"
            )
        except Exception as exc:
            logger.warning(f"Margin analysis failed: {exc}")
            stats["margin_note"] = f"Margin analysis not available: {exc}"

    # Save
    sv_summary = {k: v for k, v in stats.items()
                  if k not in ("support_indices_sample", "explanation")}
    sv_df = pd.DataFrame([sv_summary])
    out = tables_dir / "support_vector_analysis.csv"
    sv_df.to_csv(out, index=False)
    logger.info(f"SV analysis saved: {out}")

    return stats


# ---------------------------------------------------------------------------
# 5. Latency benchmark
# ---------------------------------------------------------------------------

def run_latency_test(
    fitted_models: dict,
    X_test: np.ndarray,
    tables_dir: Path,
    n_samples: int = 1000,
) -> pd.DataFrame:
    """
    Benchmark inference latency for n_samples URLs.
    fitted_models: {name: fitted_estimator}
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print(f"  INFERENCE LATENCY TEST ({n_samples} samples)")
    print("=" * 70)

    df = latency_test(fitted_models, X_test, n_samples=n_samples)
    out = tables_dir / "inference_latency.csv"
    df.to_csv(out, index=False)
    print_results_table(df, f"INFERENCE LATENCY ({n_samples} URLs)")
    logger.info(f"Latency results saved: {out}")

    print("\n  TRADE-OFF DISCUSSION:")
    print("  - Linear SVM: prediction cost = O(n_features) -- fastest.")
    print("  - RBF/Poly SVM: cost = O(n_sv x n_features) -- depends on SV count.")
    print("  - Logistic Regression: O(n_features) -- comparable to linear SVM.")
    print("  - KNN: O(n_train x n_features) -- slowest at inference (brute).")
    print("  For real-time URL screening, linear models are preferred.")

    return df


# ---------------------------------------------------------------------------
# 6. Feature importance (Linear SVM)
# ---------------------------------------------------------------------------

def run_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    tables_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Train a dedicated Linear SVM and extract coef_ for feature importance.
    Returns (coef_array, importance_df).
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print("  LINEAR SVM FEATURE IMPORTANCE")
    print("=" * 70)

    model = get_linear_svm()
    model.fit(X_train, y_train)

    coef = model.coef_[0]
    abs_coef = np.abs(coef)
    indices = np.argsort(abs_coef)[::-1]

    fi_df = pd.DataFrame({
        "Feature":    [feature_names[i] for i in indices],
        "Coefficient": [coef[i] for i in indices],
        "Abs_Coef":   [abs_coef[i] for i in indices],
        "Direction":  ["-> Phishing" if coef[i] > 0 else "-> Legitimate"
                       for i in indices],
    })

    out = tables_dir / "feature_importance.csv"
    fi_df.to_csv(out, index=False)
    print(fi_df.head(15).to_string(index=False))
    logger.info(f"Feature importance saved: {out}")

    return coef, fi_df


# ---------------------------------------------------------------------------
# 7. Final model selection
# ---------------------------------------------------------------------------

def select_final_model(
    baseline_df: pd.DataFrame,
    gs_result: dict,
    tables_dir: Path,
) -> str:
    """
    Select and justify the final model based on measured metrics.

    Criteria (in priority order for cybersecurity applications):
      1. Recall (minimise false negatives -- missed phishing sites)
      2. F1-score (balance precision/recall)
      3. Accuracy
      4. Training time
      5. Inference latency

    Returns the name of the selected model.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Add GridSearch best as a row
    gs_row = pd.DataFrame([{
        "Model":          "Best SVM (GridSearch)",
        "Accuracy":       round(gs_result["test_acc"],       4),
        "Precision":      round(gs_result["test_precision"], 4),
        "Recall":         round(gs_result["test_recall"],    4),
        "F1":             round(gs_result["test_f1"],        4),
        "Train_Time_s":   round(gs_result["gs_time"],        4),
        "Pred_Time_s":    round(gs_result["pred_time"],      6),
        "Latency_ms/URL": round(
            gs_result["pred_time"] / max(len(gs_result["y_pred"]), 1) * 1000, 6
        ),
    }])
    combined = pd.concat([baseline_df, gs_row], ignore_index=True)

    # Rank by Recall then F1
    combined["rank_score"] = combined["Recall"] * 0.5 + combined["F1"] * 0.3 + combined["Accuracy"] * 0.2
    combined_sorted = combined.sort_values("rank_score", ascending=False).reset_index(drop=True)

    final_name = combined_sorted.iloc[0]["Model"]

    print("\n" + "=" * 70)
    print("  FINAL MODEL SELECTION")
    print("=" * 70)
    print(combined_sorted[[
        "Model", "Accuracy", "Precision", "Recall", "F1",
        "Train_Time_s", "Latency_ms/URL"
    ]].to_string(index=False))
    print(f"\n  Selected model: {final_name}")
    print("\n  JUSTIFICATION:")
    print("  For phishing URL classification, a missed phishing site (false negative)")
    print("  poses greater risk than a false positive (legitimate site blocked).")
    print("  Therefore we prioritise Recall, then F1, then Accuracy.")
    print("  The selected model provides the best balance across these metrics")
    print("  based on actual measured experimental results.")

    out = tables_dir / "final_model_selection.csv"
    combined_sorted.to_csv(out, index=False)
    logger.info(f"Final selection table saved: {out}")

    return final_name
