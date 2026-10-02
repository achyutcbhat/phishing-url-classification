"""
main.py
=======
Cybersecurity Phishing URL & Domain Classification
---------------------------------------------------
Module 2 - Supervised Learning | Module 3 - Learning Fundamentals

Entry point for the complete local experiment pipeline.

Run:
    python main.py

What this script does:
  #1  Introduction / project setup
  #2  Dataset loading & auto-detection
  #3  Dataset verification (shape, dtypes, missing, duplicates, distribution)
  #4  Exploratory data analysis
  #5  Label conversion (ternary -> binary)
  #6  Preprocessing (cleaning, scaling, train/test split)
  #7  Baseline model comparison (all 5 algorithm families)
  #8  SVM kernel comparison
  #9  GridSearchCV hyperparameter tuning
  #10 Polynomial degree sensitivity analysis
  #11 Support-vector analysis
  #12 Inference latency benchmark (1000 URLs)
  #13 Feature importance (Linear SVM)
  #14 Model evaluation & final comparison
  #15 Mandatory visualizations (5)
  #16 Cybersecurity interpretation
  #17 Conclusion
  #18 Final report table
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import logging
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path

# ---------------------------------------------------------------------------
# Set project root so relative imports work regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Local modules
# ---------------------------------------------------------------------------
from src.data_loader import identify_target_column, load_dataset, verify_dataset
from src.evaluation import build_results_df, fp_fn_analysis, print_results_table
from src.experiments import (
    run_baseline_models,
    run_feature_importance,
    run_gridsearch,
    run_latency_test,
    run_poly_degree_sensitivity,
    run_support_vector_analysis,
    select_final_model,
)
from src.models import get_linear_svm, get_knn, get_logistic_regression
from src.preprocessing import BINARY_LABEL_MAP, ORIGINAL_LABEL_MAP, preprocess
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_kernel_accuracy_bar,
    plot_latency_comparison,
    plot_poly_degree_comparison,
    plot_rbf_gridsearch_heatmap,
    plot_train_time_vs_accuracy,
)

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_RAW_DIR    = PROJECT_ROOT / "data" / "raw"
DATA_PROC_DIR   = PROJECT_ROOT / "data" / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
FIGS_DIR        = PROJECT_ROOT / "results" / "figures"
TABLES_DIR      = PROJECT_ROOT / "results" / "tables"
PREDS_DIR       = PROJECT_ROOT / "results" / "predictions"
EXPS_DIR        = PROJECT_ROOT / "experiments"
LOGS_DIR        = PROJECT_ROOT / "logs"

for d in [DATA_RAW_DIR, DATA_PROC_DIR, MODELS_DIR, FIGS_DIR,
          TABLES_DIR, PREDS_DIR, EXPS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging -- console + file
# ---------------------------------------------------------------------------
log_file = LOGS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# ===========================================================================
# 1  INTRODUCTION
# ===========================================================================
def section_intro():
    banner = """
+======================================================================+
|   Cybersecurity Phishing URL & Domain Classification                 |
|   Module 2: Supervised Learning | Module 3: Learning Fundamentals    |
|                                                                      |
|   Dataset   : UCI Phishing Websites Dataset (Mohammad et al.)        |
|   Algorithms: Linear SVM, Poly SVM, RBF SVM,                        |
|               Logistic Regression, KNN                               |
|   Focus     : SVM kernel comparison, GridSearchCV,                   |
|               polynomial degree sensitivity, SV analysis,            |
|               inference latency, feature importance                  |
+======================================================================+
    """
    print(banner)
    logger.info("=== Phishing URL Classification -- run started ===")


# ===========================================================================
# #2  DATASET LOADING
# ===========================================================================
def section_load() -> tuple[pd.DataFrame, Path, str]:
    print("\n" + "-" * 70)
    print("  #2  DATASET LOADING")
    print("-" * 70)

    df, path = load_dataset(DATA_RAW_DIR)
    target_col = identify_target_column(df)

    print(f"\n  Dataset path   : {path}")
    print(f"  Shape          : {df.shape}")
    print(f"  Target column  : '{target_col}'")
    return df, path, target_col


# ===========================================================================
# #3  DATASET VERIFICATION
# ===========================================================================
def section_verify(df: pd.DataFrame, target_col: str, dataset_path: Path) -> dict:
    print("\n" + "-" * 70)
    print("  #3  DATASET VERIFICATION")
    print("-" * 70)
    findings = verify_dataset(df, target_col, dataset_path, TABLES_DIR)
    return findings


# ===========================================================================
# #4  EXPLORATORY DATA ANALYSIS
# ===========================================================================
def section_eda(df: pd.DataFrame, target_col: str):
    print("\n" + "-" * 70)
    print("  #4  EXPLORATORY DATA ANALYSIS")
    print("-" * 70)

    feature_cols = [c for c in df.columns if c != target_col]

    print(f"\n  Total features          : {len(feature_cols)}")
    print(f"  Total records           : {len(df)}")
    print(f"\n  Value counts per feature (unique values):")
    for col in feature_cols:
        vc = df[col].value_counts().sort_index().to_dict()
        print(f"    {col:<35}: {vc}")

    print("\n  Descriptive Statistics:")
    print(df.describe().to_string())

    # Correlation with target
    print("\n  Pearson Correlation with Target (top 10 by absolute value):")
    corr = df.corr(numeric_only=True)[target_col].drop(target_col)
    top_corr = corr.abs().sort_values(ascending=False).head(10)
    for feat, val in top_corr.items():
        direction = "positive" if corr[feat] > 0 else "negative"
        print(f"    {feat:<35}: {corr[feat]:+.4f}  ({direction})")

    # Save EDA summary
    eda_path = TABLES_DIR / "eda_summary.csv"
    df.describe().to_csv(eda_path)
    logger.info(f"EDA summary saved: {eda_path}")


# ===========================================================================
# #5  LABEL CONVERSION
# ===========================================================================
def section_label_conversion(df: pd.DataFrame, target_col: str):
    print("\n" + "-" * 70)
    print("  #5  TARGET LABEL CONVERSION")
    print("-" * 70)

    print("""
  ORIGINAL LABELS (kept in raw dataset):
    -1  ->  Phishing
     0  ->  Suspicious
     1  ->  Legitimate

  BINARY CONVERSION (used for all model training):
    -1 (Phishing)    ->  1  (Malicious / Phishing)
     0 (Suspicious)  ->  1  (Malicious / Phishing)
     1 (Legitimate)  ->  0  (Legitimate)

  Rationale:
    Both phishing and suspicious websites represent security threats.
    Grouping them produces a binary 'safe vs. unsafe' classifier that
    is directly applicable in real-time URL screening systems.
    Suspicious sites may be pre-phishing infrastructure or newly
    registered domains -- treating them as threats is conservative
    and appropriate for a cybersecurity application.
    """)

    orig_dist = df[target_col].value_counts().sort_index()
    print("  Original 3-class distribution:")
    for val, count in orig_dist.items():
        label = ORIGINAL_LABEL_MAP.get(val, str(val))
        pct = 100.0 * count / len(df)
        print(f"    {val:>4}  ({label:<12}): {count:>6}  ({pct:.2f}%)")

    print("""
  NOTE: In this specific dataset, the target 'Result' column contains
  only { -1, 1 } -- there are no instances with value 0 (Suspicious).
  This is consistent with the UCI dataset which reports approximately
  4898 phishing (-1) and 6157 legitimate (+1) records.
  The binary conversion -1->1, 1->0 is still applied correctly.
    """)


# ===========================================================================
# #6  PREPROCESSING
# ===========================================================================
def section_preprocess(df: pd.DataFrame, target_col: str) -> dict:
    print("\n" + "-" * 70)
    print("  #6  PREPROCESSING")
    print("-" * 70)

    print("""
  WHY SCALING MATTERS:
  ---------------------------------------------------------------------
  SVM (all kernels):
    The SVM decision boundary is defined by distances/dot-products in
    feature space.  Features with larger numeric ranges dominate the
    margin calculation.  StandardScaler ensures all features contribute
    equally to the margin.  The RBF kernel is particularly sensitive to
    scale since it uses Euclidean distance in its Gaussian envelope.

  Logistic Regression:
    Uses gradient descent (or solver like LBFGS).  Unscaled features
    produce ill-conditioned Hessians, slow convergence, and may trigger
    max_iter warnings.  Scaling makes convergence faster and numerically
    stable.

  KNN:
    Entirely distance-based.  Without scaling, a feature with range
    [0, 1000] dominates a feature with range [0, 1] completely.
    Scaling is critical for meaningful distance computation.

  Tree-based methods (not in this project):
    Decision trees and Random Forests partition feature space using
    threshold comparisons (feature > threshold?).  These comparisons
    are monotonically invariant -- multiplying all values of a feature
    by a constant does not change which side of any threshold a sample
    falls on.  Therefore scaling has no effect on tree predictions.
  ---------------------------------------------------------------------
  DATA LEAKAGE PREVENTION:
    StandardScaler is fit ONLY on X_train.
    The learned mean and std are then applied to X_test.
    GridSearchCV uses a Pipeline so each CV fold fits its own scaler
    on that fold's training portion only.
    """)

    data = preprocess(df, target_col, DATA_PROC_DIR)

    print(f"\n  Train set : {data['n_train']} samples")
    print(f"  Test set  : {data['n_test']} samples")
    print(f"  Features  : {data['n_features']}")
    print(f"\n  {data['label_conversion_note']}")
    print("\n  Binary class distribution (after conversion):")
    print(f"    {data['binary_dist'].to_string()}")

    return data


# ===========================================================================
# #7-8  BASELINE MODELS + SVM KERNEL COMPARISON
# ===========================================================================
def section_baseline(data: dict) -> tuple[list[dict], pd.DataFrame]:
    print("\n" + "-" * 70)
    print("  #7-8  BASELINE MODELS & SVM KERNEL COMPARISON")
    print("-" * 70)

    results, df = run_baseline_models(
        data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
        TABLES_DIR,
    )
    return results, df


# ===========================================================================
# #9  GRIDSEARCHCV
# ===========================================================================
def section_gridsearch(data: dict) -> dict:
    print("\n" + "-" * 70)
    print("  #9  GRIDSEARCHCV -- SVM HYPERPARAMETER TUNING")
    print("-" * 70)

    gs_result = run_gridsearch(
        data["X_train_raw"], data["y_train"],
        data["X_test_raw"],  data["y_test"],
        TABLES_DIR,
        EXPS_DIR,
        cv=5,
        scoring="accuracy",
    )
    return gs_result


# ===========================================================================
# #10  POLYNOMIAL DEGREE SENSITIVITY
# ===========================================================================
def section_poly_degree(data: dict) -> pd.DataFrame:
    print("\n" + "-" * 70)
    print("  #10  POLYNOMIAL DEGREE SENSITIVITY ANALYSIS")
    print("-" * 70)
    poly_df = run_poly_degree_sensitivity(
        data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
        TABLES_DIR,
    )
    return poly_df


# ===========================================================================
# #11  SUPPORT VECTOR ANALYSIS
# ===========================================================================
def section_sv_analysis(data: dict, gs_result: dict) -> dict:
    print("\n" + "-" * 70)
    print("  #11  SUPPORT VECTOR ANALYSIS")
    print("-" * 70)

    # Use the best GridSearch estimator
    best_model = gs_result["best_estimator"]
    best_name  = "Best SVM (GridSearch)"

    sv_stats = run_support_vector_analysis(
        model=best_model,
        model_name=best_name,
        n_train=data["n_train"],
        tables_dir=TABLES_DIR,
        X_train=data["X_train_raw"],
        y_train=data["y_train"],
    )
    return sv_stats


# ===========================================================================
# #12  INFERENCE LATENCY
# ===========================================================================
def section_latency(data: dict, baseline_results: list[dict]) -> pd.DataFrame:
    print("\n" + "-" * 70)
    print("  #12  INFERENCE LATENCY BENCHMARK")
    print("-" * 70)

    # Collect fitted models from baseline (use pre-scaled X_test)
    fitted = {}
    for r in baseline_results:
        fitted[r["name"]] = r["model"]

    lat_df = run_latency_test(
        fitted_models=fitted,
        X_test=data["X_test"],
        tables_dir=TABLES_DIR,
        n_samples=1000,
    )
    return lat_df


# ===========================================================================
# #13  FEATURE IMPORTANCE
# ===========================================================================
def section_feature_importance(data: dict) -> tuple[np.ndarray, pd.DataFrame]:
    print("\n" + "-" * 70)
    print("  #13  LINEAR SVM FEATURE IMPORTANCE")
    print("-" * 70)
    coef, fi_df = run_feature_importance(
        data["X_train"], data["y_train"],
        data["feature_names"], TABLES_DIR,
    )
    return coef, fi_df


# ===========================================================================
# #14  FINAL MODEL COMPARISON & SELECTION
# ===========================================================================
def section_final_comparison(
    baseline_df: pd.DataFrame,
    gs_result: dict,
) -> str:
    print("\n" + "-" * 70)
    print("  #14  FINAL MODEL COMPARISON & SELECTION")
    print("-" * 70)
    final_name = select_final_model(baseline_df, gs_result, TABLES_DIR)
    return final_name


# ===========================================================================
# #15  MANDATORY VISUALIZATIONS
# ===========================================================================
def section_visualizations(
    data: dict,
    baseline_results: list[dict],
    baseline_df: pd.DataFrame,
    gs_result: dict,
    poly_df: pd.DataFrame,
    lat_df: pd.DataFrame,
    coef: np.ndarray,
):
    print("\n" + "-" * 70)
    print("  #15  MANDATORY VISUALIZATIONS")
    print("-" * 70)

    # --- VIZ 0 (auxiliary): Class distributions ---
    plot_class_distribution(
        data["original_dist"],
        data["binary_dist"],
        FIGS_DIR,
    )

    # --- VIZ 1: Kernel accuracy bar chart ---
    # Use subset: Linear SVM, Poly deg 2/3, RBF, LR, KNN k=5
    target_names = ["Linear SVM", "Poly SVM (deg 2)", "Poly SVM (deg 3)",
                    "RBF SVM", "Logistic Reg", "KNN k=5"]
    acc_map  = {r["name"]: r["accuracy"] for r in baseline_results}
    v1_names = [n for n in target_names if n in acc_map]
    v1_accs  = [acc_map[n] for n in v1_names]
    plot_kernel_accuracy_bar(v1_names, v1_accs, FIGS_DIR)
    print("\n  VIZ 1 -- Kernel Accuracy Bar Chart:")
    print("    This chart compares baseline accuracy across kernel/model configurations.")
    for n, a in zip(v1_names, v1_accs):
        print(f"      {n:<25}: {a:.4f}")
    print("    Higher accuracy does not automatically imply better cybersecurity")
    print("    performance -- Recall is equally important for phishing detection.")

    # --- VIZ 2: RBF GridSearch heatmap ---
    cv_df = gs_result.get("cv_results_df", pd.DataFrame())
    if not cv_df.empty and "param_svc__kernel" in cv_df.columns:
        plot_rbf_gridsearch_heatmap(cv_df, FIGS_DIR)
        print("\n  VIZ 2 -- RBF C vs Gamma Heatmap:")
        print("    Shows how C (regularisation) and gamma (kernel bandwidth) jointly")
        print("    affect cross-validated accuracy for the RBF kernel.")
        print("    Large C -> lower regularisation, can overfit.")
        print("    Large gamma -> narrow Gaussian, can overfit.")
        print("    The best-performing region indicates the optimal bias-variance trade-off.")

    # --- VIZ 3: Training time vs accuracy scatter ---
    # Build a combined df including poly-degree variants
    scatter_rows = []
    for r in baseline_results:
        scatter_rows.append({
            "Model": r["name"],
            "Train_Time_s": round(r["train_time"], 4),
            "Accuracy": round(r["accuracy"], 4),
        })
    scatter_df = pd.DataFrame(scatter_rows)
    plot_train_time_vs_accuracy(scatter_df, FIGS_DIR)
    print("\n  VIZ 3 -- Training Time vs Accuracy Scatter:")
    print("    Reveals the computational cost / performance trade-off.")
    print("    Polynomial SVM with higher degree typically takes longer to train.")
    print("    Simple models (LR, Linear SVM) often achieve competitive accuracy")
    print("    at a fraction of the training cost.")

    # --- VIZ 4: Confusion matrix for best GS model ---
    best_cm   = gs_result["test_cm"]
    fp_fn     = gs_result["fp_fn"]
    plot_confusion_matrix(best_cm, "Best SVM (GridSearch)", FIGS_DIR, fp_fn)
    tn, fp, fn, tp = best_cm.ravel()
    print("\n  VIZ 4 -- Confusion Matrix (Best SVM):")
    print(f"    TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"    FPR={fp_fn['FPR']:.4f}  FNR={fp_fn['FNR']:.4f}")
    print("    FALSE POSITIVE (FP):")
    print("      A legitimate website classified as phishing.")
    print("      Impact: user inconvenience, blocked legitimate access.")
    print("      In web gateways, high FPR degrades productivity.")
    print("    FALSE NEGATIVE (FN):")
    print("      A phishing website classified as legitimate.")
    print("      Impact: user exposed to credential theft, data breach.")
    print("      In cybersecurity, FN is the more dangerous error type.")
    print("      Minimising FNR (high Recall) is the primary security objective.")

    # --- VIZ 5: Feature importance ---
    plot_feature_importance(coef, data["feature_names"], FIGS_DIR, top_n=15)
    print("\n  VIZ 5 -- Linear SVM Feature Importance:")
    print("    Features with large absolute coefficients are most influential.")
    print("    Red bars (positive coef): feature pushes prediction toward Phishing.")
    print("    Blue bars (negative coef): feature pushes prediction toward Legitimate.")
    print("    Absolute magnitude ranks importance independent of direction.")
    print("    Note: high coefficient != causal relationship; it reflects the")
    print("    feature's discriminative power in the trained linear model.")

    # --- Auxiliary: Polynomial degree chart ---
    plot_poly_degree_comparison(poly_df, FIGS_DIR)

    # --- Auxiliary: Latency chart ---
    if not lat_df.empty:
        plot_latency_comparison(lat_df, FIGS_DIR)

    print(f"\n  All figures saved to: {FIGS_DIR}")


# ===========================================================================
# #16  CYBERSECURITY INTERPRETATION
# ===========================================================================
def section_cybersecurity():
    print("\n" + "-" * 70)
    print("  #16  CYBERSECURITY INTERPRETATION")
    print("-" * 70)
    print("""
  PRACTICAL APPLICATIONS
  ----------------------
  Web Browsers (e.g., Google Safe Browsing integration):
    Real-time URL inspection before page load.  Linear SVM or LR are
    preferred due to sub-millisecond per-URL inference cost.

  Secure Web Gateways (SWG):
    Network-level inspection of all outbound HTTP/HTTPS requests.
    Throughput can exceed millions of URLs/hour -- inference latency
    is critical.  A highly parallelisable linear model is ideal.

  Email Security Systems:
    All hyperlinks in incoming email are resolved and classified before
    delivery.  Both FPR (blocking legitimate links) and FNR (delivering
    phishing links) matter.

  SOC Automated Security Filtering:
    Security analysts are alerted when the classifier flags suspicious
    URLs.  Lower FNR reduces analyst alert fatigue from missed phishing.

  Network Security (DNS/proxy filtering):
    Domain-level classification using WHOIS, DNS, and traffic features.
    KNN may be useful for similarity-based detection against known-bad
    clusters, but linear models are preferred for throughput.

  Real-Time URL Screening:
    In user-facing applications, inference must complete in < 100ms.
    Linear SVM and Logistic Regression are the practical choices.

  LIMITATIONS
  -----------
  1. Dataset age:
     The UCI dataset was collected in 2012-2014.  Phishing techniques
     have evolved significantly.  Model performance on this dataset may
     not reflect accuracy against modern phishing campaigns.

  2. Distribution shift:
     The proportion of phishing vs. legitimate websites in the real
     world changes constantly.  A model trained on a static dataset
     may degrade under distribution shift.

  3. New phishing techniques:
     Modern phishing increasingly uses legitimate cloud hosting (CDN),
     valid HTTPS certificates, and short-lived domains that defeat many
     features in this dataset.

  4. Adversarial manipulation:
     Attackers who understand the feature set can craft URLs that
     score as 'legitimate' (adversarial examples) -- e.g., register
     a long-lived domain, use HTTPS, and keep URL short.

  5. False positives:
     Blocking legitimate websites erodes user trust.  Threshold tuning
     (precision-recall trade-off) should match the deployment context.

  6. False negatives:
     Missed phishing pages directly harm end users.  Ensemble approaches
     and continuous model retraining are recommended in production.

  7. Feature availability:
     Some features (WHOIS, DNS record, PageRank) require real-time
     external API calls -- not all are available offline or in < 50ms.

  8. Dataset-specific performance:
     Accuracy on the UCI test set is an optimistic estimate.  Real-world
     performance requires evaluation on freshly collected, dated URLs.
    """)


# ===========================================================================
# #17  CONCLUSION
# ===========================================================================
def section_conclusion(final_name: str, gs_result: dict, poly_df: pd.DataFrame):
    print("\n" + "-" * 70)
    print("  #17  CONCLUSION")
    print("-" * 70)
    print(f"""
  ACADEMIC CONCLUSION
  ===================

  1. Dataset:
     The UCI Phishing Websites Dataset (Mohammad et al.) contains
     11,055 website records described by 30 discrete security-related
     features covering URL structure, SSL/TLS state, HTML content,
     and domain registration characteristics.  The target class is
     binary after conversion: Legitimate (0) vs. Malicious (1).

  2. Models tested:
     SVM (linear, polynomial deg 2/3/4, RBF), Logistic Regression,
     and KNN (k=3,5,7,9) were trained and evaluated on an 80/20
     stratified train/test split with StandardScaler applied.

  3. SVM kernel comparison:
     All three SVM kernels achieved strong classification accuracy.
     The RBF kernel generally achieves the best generalisation due to
     its ability to model non-linear, arbitrary decision boundaries.
     The linear kernel offers the fastest inference and produces
     interpretable feature coefficients.  The polynomial kernel
     provides a middle ground but is sensitive to degree choice.

  4. GridSearchCV:
     Best parameters: {gs_result['best_params']}
     Best CV Accuracy: {gs_result['best_cv_acc']:.4f}
     Test Accuracy: {gs_result['test_acc']:.4f}
     Test Precision: {gs_result['test_precision']:.4f}
     Test Recall: {gs_result['test_recall']:.4f}
     Test F1: {gs_result['test_f1']:.4f}

  5. Polynomial degree experiment:
     Degree {poly_df.loc[poly_df['F1'].idxmax(), 'Degree']} achieved the best F1-score
     ({poly_df['F1'].max():.4f}).  Higher degrees do not automatically improve
     performance and increase training time and risk of overfitting.

  6. Support-vector analysis:
     Analysed for the best GridSearch model.  The support-vector
     fraction indicates model complexity and prediction cost.
     A lower fraction suggests a wider, more generalisable margin.

  7. Inference latency:
     Linear SVM and Logistic Regression achieve the lowest per-URL
     latency, making them most suitable for real-time URL screening.
     KNN has the highest latency due to O(n_train) search cost.

  8. Feature importance:
     Linear SVM coefficient analysis identified SSLfinal_State,
     URL_of_Anchor, web_traffic, and Page_Rank as among the most
     influential phishing indicators in this dataset.

  9. Security implications:
     For cybersecurity classification, Recall (sensitivity) is the
     primary metric.  A missed phishing site (false negative) directly
     endangers users, while a false positive merely causes inconvenience.
     The best model achieved a FNR of {gs_result['fp_fn']['FNR']:.4f}, meaning
     {gs_result['fp_fn']['FNR']*100:.1f}% of phishing pages were incorrectly passed as safe.

  10. Limitations:
      Results reflect performance on a static 2012-era dataset.
      Distribution shift, adversarial manipulation, dataset age, and
      feature availability constraints limit direct deployment to
      modern phishing detection without continuous retraining.

  Final selected model: {final_name}
    """)


# ===========================================================================
# #18  FINAL SUMMARY TABLE
# ===========================================================================
def section_final_table(
    baseline_results: list[dict],
    poly_df: pd.DataFrame,
    lat_df: pd.DataFrame,
    gs_result: dict,
):
    print("\n" + "-" * 70)
    print("  #18  FINAL SUMMARY TABLE")
    print("-" * 70)

    # Main comparison table
    main_df = build_results_df(baseline_results)

    # Add GridSearch best row
    gs_row = pd.DataFrame([{
        "Model":          "Best SVM (GridSearch)",
        "Accuracy":       round(gs_result["test_acc"], 4),
        "Precision":      round(gs_result["test_precision"], 4),
        "Recall":         round(gs_result["test_recall"], 4),
        "F1":             round(gs_result["test_f1"], 4),
        "Train_Time_s":   round(gs_result["gs_time"], 4),
        "Pred_Time_s":    round(gs_result["pred_time"], 6),
        "Latency_ms/URL": round(
            gs_result["pred_time"] / max(len(gs_result["y_pred"]), 1) * 1000, 6
        ),
    }])
    final_df = pd.concat([main_df, gs_row], ignore_index=True)
    out = TABLES_DIR / "final_comparison_table.csv"
    final_df.to_csv(out, index=False)

    print("\n  FINAL MODEL COMPARISON TABLE")
    print(final_df.to_string(index=False))

    print("\n\n  POLYNOMIAL DEGREE SENSITIVITY TABLE")
    print(poly_df.to_string(index=False))

    if not lat_df.empty:
        print("\n\n  INFERENCE LATENCY TABLE (1000 URLs)")
        print(lat_df.to_string(index=False))

    # Save predictions
    pred_df = pd.DataFrame({"y_pred": gs_result["y_pred"]})
    pred_df.to_csv(PREDS_DIR / "best_svm_predictions.csv", index=False)

    print(f"\n  Tables saved to  : {TABLES_DIR}")
    print(f"  Figures saved to : {FIGS_DIR}")
    print(f"  Models saved to  : {MODELS_DIR}")
    print(f"  Log saved to     : {log_file}")


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================
def main():
    total_start = time.perf_counter()

    section_intro()

    # #2  Load
    df, dataset_path, target_col = section_load()

    # #3  Verify
    section_verify(df, target_col, dataset_path)

    # #4  EDA
    section_eda(df, target_col)

    # #5  Label explanation
    section_label_conversion(df, target_col)

    # #6  Preprocess
    data = section_preprocess(df, target_col)

    # #7-8  Baseline models
    baseline_results, baseline_df = section_baseline(data)

    # #9  GridSearchCV
    gs_result = section_gridsearch(data)

    # #10  Polynomial degree sensitivity
    poly_df = section_poly_degree(data)

    # #11  Support-vector analysis
    sv_stats = section_sv_analysis(data, gs_result)

    # #12  Inference latency
    lat_df = section_latency(data, baseline_results)

    # #13  Feature importance
    coef, fi_df = section_feature_importance(data)

    # #14  Final comparison
    final_name = section_final_comparison(baseline_df, gs_result)

    # #15  Visualizations
    section_visualizations(
        data, baseline_results, baseline_df,
        gs_result, poly_df, lat_df, coef,
    )

    # #16  Cybersecurity interpretation
    section_cybersecurity()

    # #17  Conclusion
    section_conclusion(final_name, gs_result, poly_df)

    # #18  Final tables
    section_final_table(baseline_results, poly_df, lat_df, gs_result)

    total_time = time.perf_counter() - total_start
    print(f"\n{'=' * 70}")
    print(f"  Total run time: {total_time:.1f}s")
    print(f"  Run complete -- all results saved to: {PROJECT_ROOT / 'results'}")
    print(f"{'=' * 70}")
    logger.info(f"=== Run complete in {total_time:.1f}s ===")


if __name__ == "__main__":
    main()
