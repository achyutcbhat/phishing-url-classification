"""
preprocessing.py
----------------
Data preprocessing pipeline for the Phishing Websites Dataset.

Steps:
  1. Drop duplicates
  2. Drop rows with missing values (if any)
  3. Binary label conversion:
       Original -1 (Phishing)   -> 1 (Malicious)
       Original  0 (Suspicious) -> 1 (Malicious)
       Original  1 (Legitimate) -> 0 (Legitimate)
  4. Feature / target separation
  5. Stratified train/test split (80/20, random_state=42)
  6. StandardScaler fitted ONLY on training data (no leakage)
  7. Save processed splits to data/processed/

WHY SCALING MATTERS
-------------------
- SVM: distance-based decision boundary -> features on different scales
  cause the hyperplane to be dominated by high-magnitude dimensions.
- Logistic Regression: gradient-based optimisation converges much faster
  and is numerically more stable when inputs are standardised.
- KNN: purely distance-based; large-magnitude features completely overshadow
  small-magnitude ones without scaling.
- Tree-based methods (e.g. Decision Tree, Random Forest): use threshold-based
  splits that are invariant to monotonic feature transformations -- scaling
  would make no difference to their predictions.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
RANDOM_STATE = 42

ORIGINAL_LABEL_MAP = {-1: "Phishing", 0: "Suspicious", 1: "Legitimate"}
BINARY_LABEL_MAP = {0: "Legitimate", 1: "Malicious/Phishing"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_labels(series: pd.Series) -> pd.Series:
    """
    Convert original ternary labels to binary security labels.

    Mapping:
        -1 (Phishing)   -> 1  (Malicious)
        0  (Suspicious) -> 1  (Malicious)
        1  (Legitimate) -> 0  (Legitimate)

    Rationale: For a cybersecurity classifier, both phishing and suspicious
    sites represent potential threats.  Grouping them together produces a
    binary 'safe vs. unsafe' decision that is operationally useful for
    real-time URL screening.
    """
    def _map(val):
        if val == 1:
            return 0   # Legitimate -> safe
        else:
            return 1   # -1 or 0   -> malicious / phishing

    return series.map(_map).astype(int)


def preprocess(
    df: pd.DataFrame,
    target_col: str,
    processed_dir: Path,
) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df            : raw DataFrame
    target_col    : name of the target column
    processed_dir : directory to persist processed splits

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test  (numpy arrays)
        X_train_raw, X_test_raw           (DataFrames before scaling)
        feature_names                     (list of str)
        scaler                            (fitted StandardScaler)
        label_conversion_note             (str, human-readable explanation)
        binary_dist_train / binary_dist_test
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Drop duplicates
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        logger.info(f"Dropped {n_before - n_after} duplicate rows.")
    else:
        logger.info("No duplicate rows found.")

    # ------------------------------------------------------------------
    # Step 2: Drop rows with NaN
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) != n_before:
        logger.info(f"Dropped {n_before - len(df)} rows with missing values.")

    # ------------------------------------------------------------------
    # Step 3: Original class distribution (before conversion)
    # ------------------------------------------------------------------
    original_dist = df[target_col].value_counts().sort_index()
    logger.info("Original class distribution:\n" + original_dist.to_string())

    # ------------------------------------------------------------------
    # Step 4: Binary label conversion
    # ------------------------------------------------------------------
    y_binary = convert_labels(df[target_col])
    binary_dist = y_binary.value_counts().sort_index()
    logger.info("Binary class distribution:\n" + binary_dist.to_string())

    label_note = (
        "LABEL CONVERSION\n"
        "  Original -1 (Phishing)    ->  Binary 1 (Malicious/Phishing)\n"
        "  Original  0 (Suspicious)  ->  Binary 1 (Malicious/Phishing)\n"
        "  Original  1 (Legitimate)  ->  Binary 0 (Legitimate)\n"
        "\n"
        "Rationale: Both phishing and suspicious sites pose security risks.\n"
        "For binary classification: 1 = threat, 0 = safe.\n"
    )

    # ------------------------------------------------------------------
    # Step 5: Feature / target separation
    # ------------------------------------------------------------------
    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values.astype(float)
    y = y_binary.values

    # ------------------------------------------------------------------
    # Step 6: Stratified train/test split
    # ------------------------------------------------------------------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        f"Train size: {X_train_raw.shape[0]} | Test size: {X_test_raw.shape[0]}"
    )

    # ------------------------------------------------------------------
    # Step 7: StandardScaler -- fit ONLY on training data
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)   # learn ? and ? from train
    X_test = scaler.transform(X_test_raw)         # apply same transform to test
    # NO information from the test set is used to compute ? or ?.

    # ------------------------------------------------------------------
    # Step 8: Save processed splits
    # ------------------------------------------------------------------
    _save_split(X_train, y_train, feature_names, processed_dir / "train.csv")
    _save_split(X_test, y_test, feature_names, processed_dir / "test.csv")

    # Distribution summary
    binary_dist_train = pd.Series(y_train).value_counts().sort_index()
    binary_dist_test = pd.Series(y_test).value_counts().sort_index()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "feature_names": feature_names,
        "scaler": scaler,
        "original_dist": original_dist,
        "binary_dist": binary_dist,
        "binary_dist_train": binary_dist_train,
        "binary_dist_test": binary_dist_test,
        "label_conversion_note": label_note,
        "n_train": X_train_raw.shape[0],
        "n_test": X_test_raw.shape[0],
        "n_features": len(feature_names),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_split(X: np.ndarray, y: np.ndarray, cols: list, path: Path):
    df_out = pd.DataFrame(X, columns=cols)
    df_out["label"] = y
    df_out.to_csv(path, index=False)
    logger.info(f"Saved: {path}")
