# Cybersecurity Phishing URL & Domain Classification

**Module 2 – Supervised Learning | Module 3 – Learning Fundamentals**

A complete local machine-learning project for classifying phishing URLs and domains using the UCI Phishing Websites Dataset (Mohammad et al.).

---

## Project Objective

Classify websites as **Phishing / Malicious** or **Legitimate** using five supervised learning algorithms:

1. SVM – Linear Kernel
2. SVM – Polynomial Kernel (degree 2, 3, 4)
3. SVM – RBF Kernel
4. Logistic Regression
5. K-Nearest Neighbors (KNN)

Key focus areas:
- SVM kernel comparison
- GridSearchCV hyperparameter tuning
- Polynomial degree sensitivity analysis
- Support-vector analysis
- Inference latency benchmarking
- Feature importance (Linear SVM)

---

## Dataset

**Phishing Websites Dataset — UCI / Mohammad et al.**
- ~11,055 website records
- 30 security-related discrete features (URL structure, HTML, domain-based)
- Target (`Result`): `-1` = Phishing, `0` = Suspicious, `1` = Legitimate

**Binary label conversion used for experiments:**
```
Phishing (-1) + Suspicious (0)  →  Malicious (1)
Legitimate (1)                  →  Legitimate (0)
```

---

## Project Structure

```
Phishing-URL-Classification/
├── data/
│   ├── raw/          ← original dataset placed here automatically
│   └── processed/    ← scaled/split data saved here
├── models/           ← saved model files (.joblib)
├── results/
│   ├── figures/      ← all PNG visualizations
│   ├── tables/       ← CSV result tables
│   └── predictions/  ← test set predictions
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── experiments.py
│   └── visualization.py
├── experiments/      ← GridSearch CV results
├── logs/             ← run logs
├── tests/
├── requirements.txt
├── README.md
└── main.py
```

---

## Quick Start (Windows PowerShell)

### 1. Create and activate virtual environment

```powershell
cd "Phishing-URL-Classification"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the full project

```powershell
python main.py
```

The script will:
1. Locate the dataset automatically (searches workspace and `data/raw/`)
2. Run dataset verification and save a report
3. Perform preprocessing with no data leakage
4. Train and evaluate all baseline models
5. Run GridSearchCV over SVM kernels
6. Run polynomial degree sensitivity analysis
7. Perform support-vector analysis
8. Benchmark inference latency on 1,000 samples
9. Generate all 5 mandatory visualizations
10. Save a final comparison table

---

## Output Files

| Location | Contents |
|---|---|
| `results/tables/` | CSV tables: baseline comparison, GridSearch results, poly-degree analysis, final summary |
| `results/figures/` | PNG plots: kernel accuracy bar, RBF heatmap, train-time scatter, confusion matrix, feature importance |
| `results/predictions/` | Test set predictions CSV |
| `models/` | Saved best model (joblib) |
| `logs/` | Full run log |
| `data/processed/` | Processed train/test splits |
| `experiments/` | Full GridSearchCV CV results |

---

## Reproducibility

All experiments use `random_state=42`.
Preprocessing (scaling) is fitted on training data only — no data leakage.
GridSearchCV operates exclusively on the training set.
The test set is held out until final evaluation.

---

## References

Mohammad, R.M., Thabtah, F., & McCluskey, L. (2014).
*Predicting Phishing Websites based on Self-Structuring Neural Network.*
Neural Computing and Applications.
UCI Machine Learning Repository: Phishing Websites Dataset.
