import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Constants
RANDOM_STATE = 42
MODELS_DIR = "models"
DATA_DIR = "data/processed"


def load_data():
    """Load processed data from CSVs."""
    print("Loading data...")
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def get_hyperparameter_search_space():
    """Define models and their hyperparameter grids."""
    # We will tune RandomForest as it's usually robust for this type of data
    rf_params = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': ['balanced', 'balanced_subsample', None]
    }

    return rf_params


def train_and_tune():
    # 1. Setup
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()

    # 2. Define Pipeline (SMOTE + RandomForest)
    # Note: We use imblearn Pipeline to ensure SMOTE only happens on training folds during CV
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    # 3. Hyperparameter Tuning (Step 3.2 logic integrated here)
    print("\nStarting Hyperparameter Tuning (RandomizedSearchCV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_hyperparameter_search_space(),
        n_iter=10,  # Keep it fast for now (increase to 20-50 for better results)
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE
    )

    search.fit(X_train, y_train)

    print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best Parameters: {search.best_params_}")

    best_model = search.best_estimator_

    # 4. Final Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("FINAL MODEL REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred))
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # 5. Save Artifacts (CRITICAL STEP)
    print("\nSaving artifacts...")

    # Save the pipeline (includes the model)
    # We strip the SMOTE step for the saved model because we don't need to oversample during inference
    final_classifier = best_model.named_steps['classifier']

    # Save model
    joblib.dump(final_classifier, f"{MODELS_DIR}/rf_model.joblib")
    print(f"✓ Model saved to {MODELS_DIR}/rf_model.joblib")

    # Save feature names (THIS WAS MISSING/UNCLEAR IN PREVIOUS STEPS)
    joblib.dump(feature_names, f"{MODELS_DIR}/features.joblib")
    print(f"✓ Feature names saved to {MODELS_DIR}/features.joblib")

    # 6. Generate & Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(f"{MODELS_DIR}/confusion_matrix.png")
    print(f"✓ Confusion matrix saved to {MODELS_DIR}/confusion_matrix.png")


if __name__ == "__main__":
    train_and_tune()
