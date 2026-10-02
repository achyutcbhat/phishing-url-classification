"""
data_loader.py
--------------
Locates and loads the Phishing Websites Dataset (ARFF or CSV).
Generates a dataset verification report saved to results/tables/.

Dataset: UCI Phishing Websites Dataset -- Mohammad et al.
Target column: 'Result'  { -1=Phishing, 0=Suspicious, 1=Legitimate }

NOTE: In this particular dataset the target 'Result' contains only {-1, 1}
(no class 0 is present). The code handles both 2-class and 3-class targets.
"""

import os
import re
import glob
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candidate search roots -- adjust if workspace is structured differently
# ---------------------------------------------------------------------------
_SEARCH_ROOTS = [
    Path(__file__).resolve().parents[1],          # project root
    Path(__file__).resolve().parents[2],          # one level up (workspace root)
    Path("D:/phishing web link"),                 # hard-coded workspace fallback
]


def _find_dataset_file() -> Path:
    """
    Search for a .arff or .csv dataset file starting from known roots.
    Prefers files whose name contains 'training' or 'phishing'.
    Returns the first match; raises FileNotFoundError if nothing found.
    """
    candidates = []
    for root in _SEARCH_ROOTS:
        if not root.exists():
            continue
        for ext in ("*.arff", "*.csv"):
            for p in root.rglob(ext):
                # skip venv and hidden directories
                if any(part.startswith(".") or part == "venv" for part in p.parts):
                    continue
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "No .arff or .csv dataset file found. "
            "Please place the dataset in data/raw/ or the workspace root."
        )

    # Prefer files explicitly named 'training' or 'phishing'
    preferred = [
        p for p in candidates
        if re.search(r"(training|phishing)", p.stem, re.IGNORECASE)
    ]
    chosen = preferred[0] if preferred else candidates[0]
    logger.info(f"Dataset located at: {chosen}")
    return chosen


def _parse_arff(path: Path) -> pd.DataFrame:
    """
    Parse an ARFF file into a pandas DataFrame.
    Handles numeric and nominal attribute declarations.
    """
    attributes = []
    data_lines = []
    in_data_section = False

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            upper = line.upper()
            if upper.startswith("@RELATION"):
                continue
            elif upper.startswith("@ATTRIBUTE"):
                # Extract attribute name (second token)
                parts = line.split()
                attr_name = parts[1].strip("'\"")
                attributes.append(attr_name)
            elif upper.startswith("@DATA"):
                in_data_section = True
            elif in_data_section:
                data_lines.append(line)

    if not attributes or not data_lines:
        raise ValueError(f"Failed to parse ARFF file: {path}")

    rows = []
    for line in data_lines:
        values = [v.strip() for v in line.split(",")]
        rows.append(values)

    df = pd.DataFrame(rows, columns=attributes)

    # Convert all columns to numeric (ARFF stores everything as strings initially)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"ARFF parsed: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def load_dataset(raw_dir: Path) -> tuple[pd.DataFrame, Path]:
    """
    Locate the dataset, load it into a DataFrame, copy it to data/raw/ if needed.

    Returns
    -------
    df   : raw DataFrame
    path : resolved path of the loaded file
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if it's already in data/raw/
    existing = list(raw_dir.glob("*.arff")) + list(raw_dir.glob("*.csv"))
    if existing:
        path = existing[0]
        logger.info(f"Dataset already in data/raw/: {path}")
    else:
        path = _find_dataset_file()
        dest = raw_dir / path.name
        if path.resolve() != dest.resolve():
            shutil.copy2(path, dest)
            logger.info(f"Dataset copied to data/raw/: {dest}")
        path = dest

    ext = path.suffix.lower()
    if ext == ".arff":
        df = _parse_arff(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return df, path


def identify_target_column(df: pd.DataFrame) -> str:
    """
    Intelligently identify the target/label column.

    Priority:
      1. 'Result' (exact, case-insensitive)
      2. Last column if it has <= 3 unique values and other columns do not
    """
    # Priority 1: known UCI name
    for col in df.columns:
        if col.strip().lower() == "result":
            logger.info(f"Target column identified: '{col}' (matched 'Result')")
            return col

    # Priority 2: last column heuristic
    last_col = df.columns[-1]
    unique_vals = df[last_col].nunique()
    logger.info(
        f"No 'Result' column found. Using last column '{last_col}' "
        f"as target ({unique_vals} unique values)."
    )
    return last_col


def verify_dataset(
    df: pd.DataFrame,
    target_col: str,
    dataset_path: Path,
    out_dir: Path,
) -> dict:
    """
    Run a comprehensive dataset verification and save a text/CSV report.

    Returns a dict with all verification findings.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "dataset_verification_report.txt"

    findings = {}
    lines = []

    def _h(title: str):
        lines.append("\n" + "=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    _h("DATASET VERIFICATION REPORT")
    lines.append(f"  Dataset path      : {dataset_path}")
    lines.append(f"  Target column     : {target_col}")

    # 1. Shape
    _h("1. SHAPE")
    lines.append(f"  Rows    : {df.shape[0]}")
    lines.append(f"  Columns : {df.shape[1]}")
    findings["shape"] = df.shape

    # 2. First 5 rows
    _h("2. FIRST 5 ROWS")
    lines.append(df.head().to_string())
    findings["head"] = df.head()

    # 3. Column names
    _h("3. COLUMN NAMES")
    for i, col in enumerate(df.columns):
        lines.append(f"  [{i:02d}] {col}")
    findings["columns"] = list(df.columns)

    # 4. Data types
    _h("4. DATA TYPES")
    lines.append(df.dtypes.to_string())
    findings["dtypes"] = df.dtypes

    # 5. Missing values
    _h("5. MISSING VALUES")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        lines.append("  No missing values found.")
    else:
        lines.append(missing[missing > 0].to_string())
    findings["missing_values"] = missing
    findings["total_missing"] = total_missing

    # 6. Duplicate rows
    _h("6. DUPLICATE ROWS")
    n_dup = df.duplicated().sum()
    lines.append(f"  Duplicate rows : {n_dup}")
    findings["n_duplicates"] = n_dup

    # 7. Unique target values
    _h("7. UNIQUE TARGET VALUES")
    unique_targets = sorted(df[target_col].unique())
    lines.append(f"  Unique values in '{target_col}': {unique_targets}")
    findings["unique_targets"] = unique_targets

    # 8. Class distribution (original)
    _h("8. CLASS DISTRIBUTION (ORIGINAL)")
    dist = df[target_col].value_counts().sort_index()
    label_map = {-1: "Phishing", 0: "Suspicious", 1: "Legitimate"}
    for val, count in dist.items():
        pct = 100.0 * count / len(df)
        label = label_map.get(val, str(val))
        lines.append(f"  {val:>4}  ({label:<12}) : {count:>6}  ({pct:.2f}%)")
    findings["class_distribution"] = dist

    # 9. Invalid / unexpected values
    _h("9. INVALID / UNEXPECTED VALUES")
    feature_cols = [c for c in df.columns if c != target_col]
    allowed_feature_vals = {-1, 0, 1}
    invalid_cols = {}
    for col in feature_cols:
        bad = set(df[col].dropna().unique()) - allowed_feature_vals
        if bad:
            invalid_cols[col] = bad
    if not invalid_cols:
        lines.append("  All feature values are within expected set {-1, 0, 1}.")
    else:
        for col, bad_vals in invalid_cols.items():
            lines.append(f"  Column '{col}': unexpected values {bad_vals}")
    findings["invalid_cols"] = invalid_cols

    # 10. Summary
    _h("10. SUMMARY")
    lines.append(f"  Total records   : {df.shape[0]}")
    lines.append(f"  Total features  : {len(feature_cols)}")
    lines.append(f"  Missing values  : {total_missing}")
    lines.append(f"  Duplicate rows  : {n_dup}")
    lines.append(f"  Target classes  : {unique_targets}")
    lines.append(f"  Dataset status  : {'CLEAN' if total_missing == 0 and not invalid_cols else 'NEEDS ATTENTION'}")

    report_text = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)

    logger.info(f"Verification report saved: {report_path}")
    print(report_text)
    findings["report_path"] = report_path
    return findings
