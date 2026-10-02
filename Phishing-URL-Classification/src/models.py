"""
models.py
---------
Defines and trains all required classifiers:
  - SVM Linear
  - SVM Polynomial (degree 2, 3, 4)
  - SVM RBF
  - Logistic Regression
  - KNN (k = 3, 5, 7, 9)

Uses scikit-learn Pipelines to bundle StandardScaler + classifier so that
GridSearchCV never leaks test-set statistics into the scaler.

NOTE: The scaler is included in each Pipeline.
      When the pipeline is used with pre-scaled data (X_train / X_test),
      we set scaler__copy=True and pass scaled inputs directly, OR we
      configure the pipeline to accept raw inputs and scale internally.

      In this project we provide BOTH options:
        - build_pipeline()  : receives RAW features, scales internally
        - build_model()     : receives pre-scaled features, no internal scaling
      main.py uses build_pipeline() for GridSearchCV and build_model()
      for quick baseline runs where we already have X_train_scaled.
"""

import logging
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Model factories (bare estimators -- work with pre-scaled data)
# ---------------------------------------------------------------------------

def get_linear_svm(C: float = 1.0) -> SVC:
    return SVC(kernel="linear", C=C, random_state=RANDOM_STATE, probability=False)


def get_poly_svm(degree: int = 3, C: float = 1.0, gamma: str = "scale") -> SVC:
    return SVC(
        kernel="poly",
        degree=degree,
        C=C,
        gamma=gamma,
        random_state=RANDOM_STATE,
        probability=False,
    )


def get_rbf_svm(C: float = 1.0, gamma: str = "scale") -> SVC:
    return SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        random_state=RANDOM_STATE,
        probability=False,
    )


def get_logistic_regression(max_iter: int = 2000) -> LogisticRegression:
    return LogisticRegression(
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )


def get_knn(n_neighbors: int = 5) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=n_neighbors)


# ---------------------------------------------------------------------------
# Pipeline factory (raw -> scale -> classify; used for GridSearchCV)
# ---------------------------------------------------------------------------

def build_pipeline(estimator) -> Pipeline:
    """
    Wrap estimator in a Pipeline with StandardScaler.
    Use this with RAW (unscaled) X_train / X_test.
    GridSearchCV cross-validation folds will each fit the scaler
    independently on the fold's training portion -- no data leakage.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", estimator),
    ])


# ---------------------------------------------------------------------------
# Baseline model catalogue
# ---------------------------------------------------------------------------

def get_baseline_models() -> list[tuple[str, Any]]:
    """
    Returns a list of (name, estimator) tuples for baseline comparison.
    All estimators expect PRE-SCALED input (use with X_train / X_test).

    KNN effect of K:
      - Small K (e.g. 3): low bias, high variance -- sensitive to noise.
      - Large K (e.g. 9): higher bias, lower variance -- smoother boundary.
      - Very large K: may under-fit, and inference time grows as O(n*d).
      - Inference time increases linearly with K for brute-force search.
    """
    models = [
        ("Linear SVM",       get_linear_svm()),
        ("Poly SVM (deg 2)", get_poly_svm(degree=2)),
        ("Poly SVM (deg 3)", get_poly_svm(degree=3)),
        ("Poly SVM (deg 4)", get_poly_svm(degree=4)),
        ("RBF SVM",          get_rbf_svm()),
        ("Logistic Reg",     get_logistic_regression()),
        ("KNN k=3",          get_knn(3)),
        ("KNN k=5",          get_knn(5)),
        ("KNN k=7",          get_knn(7)),
        ("KNN k=9",          get_knn(9)),
    ]
    return models


# ---------------------------------------------------------------------------
# GridSearchCV parameter grid (kernel-specific, no invalid combinations)
# ---------------------------------------------------------------------------

def get_svm_param_grid() -> list[dict]:
    """
    Valid SVM parameter grid for GridSearchCV.

    Uses Pipeline prefix 'svc__' for all SVC parameters.

    Gamma is only meaningful for 'rbf' and 'poly' kernels.
    Degree is only meaningful for the 'poly' kernel.
    Using a list of dicts prevents invalid combinations.
    """
    return [
        # --- Linear ---
        {
            "svc__kernel": ["linear"],
            "svc__C":      [0.1, 10, 100],
        },
        # --- Polynomial ---
        {
            "svc__kernel": ["poly"],
            "svc__C":      [0.1, 10, 100],
            "svc__gamma":  ["scale", "auto", 0.01, 0.1],
            "svc__degree": [2, 3, 4],
        },
        # --- RBF ---
        {
            "svc__kernel": ["rbf"],
            "svc__C":      [0.1, 10, 100],
            "svc__gamma":  ["scale", "auto", 0.01, 0.1],
        },
    ]
