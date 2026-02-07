# --- 0) Assumptions from your previous step ---
# You already have:
# X_train, X_test, y_train, y_test  (split done BEFORE any CV)
# numeric_features, categorical_features (lists of column names)
# and your best base model chosen previously, e.g.:
# best_base_model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV  # or GridSearchCV

# Example best model placeholder (replace with your actual best model from previous step)
from sklearn.linear_model import LogisticRegression
best_base_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)

# --- 1) Preprocess + model pipeline (recommended) ---
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", best_base_model),
])

# --- 2) Hyperparameter search space (edit to match your chosen model) ---
# Parameters are referenced as: stepname__param (e.g., "model__C")
param_distributions = {
    "model__C": np.logspace(-3, 2, 30),
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs", "saga"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    random_state=42,
    verbose=1,
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Best params:", search.best_params_)
print("Best CV ROC-AUC:", search.best_score_)
