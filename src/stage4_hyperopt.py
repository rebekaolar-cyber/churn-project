"""
Stage 4 — Hyperparameter Optimisation
======================================
Best model   : RandomForestClassifier
Dataset      : Telco Customer Churn
Preprocessing: Already done (OHE via pd.get_dummies in preprocess.py).
               The processed CSVs are fully numeric, so the sklearn Pipeline
               here only needs to StandardScale the four continuous columns
               and pass the 26 binary dummy columns through unchanged.

Outputs (all written to models/)
  churn_model.pkl              — full fitted sklearn Pipeline
  scaler.pkl                   — fitted StandardScaler (extracted from pipeline)
  roc_pr_curves.png            — ROC + Precision-Recall curves side by side
  final_model_performance.json — best_params, CV & test metrics, confusion matrix
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
DATA_DIR     = "data/processed"
MODELS_DIR   = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Continuous columns that benefit from scaling.
# All other columns in the processed CSVs are binary dummies (0/1) and are
# passed through the ColumnTransformer unchanged.
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


# ── 1. Load processed data ─────────────────────────────────────────────────────
print("=" * 60)
print("Stage 4 — Hyperparameter Optimisation")
print("=" * 60)
print("\n[1/6]  Loading processed data …")

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

# Derive binary-dummy column list dynamically so the script is robust to any
# future changes in preprocessing.
BINARY_FEATURES = [c for c in X_train.columns if c not in NUMERIC_FEATURES]

print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"  Numeric features  ({len(NUMERIC_FEATURES)}) : {NUMERIC_FEATURES}")
print(f"  Binary (passthru) ({len(BINARY_FEATURES)}) : {BINARY_FEATURES}")


# ── 2. Build sklearn Pipeline ──────────────────────────────────────────────────
print("\n[2/6]  Building sklearn Pipeline …")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("bin", "passthrough",    BINARY_FEATURES),
    ],
    remainder="drop",
    verbose_feature_names_out=False,   # keeps original column names in output
)

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model",      RandomForestClassifier(random_state=RANDOM_STATE)),
])

print("  Pipeline steps:", [name for name, _ in pipe.steps])


# ── 3. Hyperparameter search space ────────────────────────────────────────────
print("\n[3/6]  Defining hyperparameter search space …")

# Parameters are referenced as  stepname__param  (e.g. "model__n_estimators").
param_distributions = {
    "model__n_estimators":      [100, 200, 300, 500],
    "model__max_depth":         [None, 5, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf":  [1, 2, 4],
    "model__max_features":      ["sqrt", "log2", 0.5],
    "model__class_weight":      ["balanced", "balanced_subsample", None],
    "model__bootstrap":         [True, False],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    estimator          = pipe,
    param_distributions= param_distributions,
    n_iter             = 30,
    scoring            = "roc_auc",
    cv                 = cv,
    n_jobs             = -1,
    refit              = True,
    random_state       = RANDOM_STATE,
    verbose            = 1,
)


# ── 4. Fit ─────────────────────────────────────────────────────────────────────
print("\n[4/6]  Running RandomizedSearchCV  (n_iter=30, cv=5, scoring=roc_auc) …")
search.fit(X_train, y_train)
best_model = search.best_estimator_

print("\n── Best Parameters ──────────────────────────────────────")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\n  Best CV ROC-AUC : {search.best_score_:.4f}")


# ── 5. Test-set evaluation ─────────────────────────────────────────────────────
print("\n[5/6]  Evaluating best_estimator_ on held-out test set …")

y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n── Classification Report ────────────────────────────────")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("── Confusion Matrix ─────────────────────────────────────")
print(cm)

test_roc_auc = roc_auc_score(y_test, y_proba)
test_pr_auc  = average_precision_score(y_test, y_proba)

print(f"\n  Test ROC-AUC             : {test_roc_auc:.4f}")
print(f"  Test PR-AUC (Avg Prec.) : {test_pr_auc:.4f}")


# ── 6. ROC + Precision-Recall curves ──────────────────────────────────────────
print("\n[6/6]  Plotting and saving curves …")

fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))

# — ROC curve —
RocCurveDisplay.from_predictions(
    y_test, y_proba,
    ax   = ax_roc,
    name = "RandomForest (tuned)",
)
ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray",
            label="Random baseline", alpha=0.7)
ax_roc.set_title(f"ROC Curve  (AUC = {test_roc_auc:.4f})", fontsize=13)
ax_roc.legend(loc="lower right")
ax_roc.grid(alpha=0.3)

# — Precision-Recall curve —
PrecisionRecallDisplay.from_predictions(
    y_test, y_proba,
    ax   = ax_pr,
    name = "RandomForest (tuned)",
)
baseline_precision = y_test.mean()
ax_pr.axhline(baseline_precision, linestyle="--", color="gray",
              label=f"No-skill baseline ({baseline_precision:.2f})", alpha=0.7)
ax_pr.set_title(f"Precision-Recall Curve  (AP = {test_pr_auc:.4f})", fontsize=13)
ax_pr.legend(loc="upper right")
ax_pr.grid(alpha=0.3)

plt.suptitle("RandomForestClassifier — Tuned Model", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()

curves_path = os.path.join(MODELS_DIR, "roc_pr_curves.png")
plt.savefig(curves_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Curves saved → {curves_path}")


# ── 7. Save model artefacts ────────────────────────────────────────────────────
model_path  = os.path.join(MODELS_DIR, "churn_model.pkl")
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

# Full pipeline (preprocessor + classifier) — use this for inference
joblib.dump(best_model, model_path)
print(f"  ✓ Full pipeline saved  → {model_path}")

# Extract the fitted StandardScaler from inside the ColumnTransformer
# Access path: pipeline → preprocess (ColumnTransformer) → "num" transformer (StandardScaler)
fitted_scaler = (
    best_model
    .named_steps["preprocess"]
    .named_transformers_["num"]
)
joblib.dump(fitted_scaler, scaler_path)
print(f"  ✓ Fitted scaler saved  → {scaler_path}")


# ── 8. Performance JSON ────────────────────────────────────────────────────────
def _serialisable(v):
    """Convert numpy scalars/types to plain Python so json.dump doesn't choke."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


performance = {
    "best_params":    {k: _serialisable(v) for k, v in search.best_params_.items()},
    "cv_roc_auc":     round(float(search.best_score_), 6),
    "test_roc_auc":   round(float(test_roc_auc), 6),
    "test_pr_auc":    round(float(test_pr_auc), 6),
    "confusion_matrix": cm.tolist(),   # [[TN, FP], [FN, TP]]
}

json_path = os.path.join(MODELS_DIR, "final_model_performance.json")
with open(json_path, "w") as f:
    json.dump(performance, f, indent=2)
print(f"  ✓ Performance JSON saved → {json_path}")


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 4 complete.  Artefacts written to models/")
print("=" * 60)
print(f"  churn_model.pkl              — full tuned pipeline")
print(f"  scaler.pkl                   — fitted StandardScaler")
print(f"  roc_pr_curves.png            — ROC & PR curves")
print(f"  final_model_performance.json — metrics + best params")
print(f"\n  CV ROC-AUC  : {search.best_score_:.4f}")
print(f"  Test ROC-AUC: {test_roc_auc:.4f}")
print(f"  Test PR-AUC : {test_pr_auc:.4f}")
