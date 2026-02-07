import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os


def train_model():
    """
    Train a Random Forest Classifier on the processed churn data.
    
    Steps:
    1. Load processed data from data/processed/
    2. Apply SMOTE to handle class imbalance
    3. Train Random Forest Classifier
    4. Print classification report
    5. Save model to models/rf_model.joblib
    6. Save feature column names to models/features.joblib
    """
    # Load processed data
    print("Loading processed data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Churn distribution in training set: {y_train.value_counts().to_dict()}")
    
    # Apply SMOTE to handle class imbalance
    print("\nApplying SMOTE to balance the training set...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Training set: {X_train_resampled.shape[0]} samples")
    print(f"Churn distribution after SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    
    # Train Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions on test set
    print("Making predictions on test set...")
    y_pred = rf_model.predict(X_test)
    
    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    print("="*60)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the trained model
    print("\nSaving trained model to models/rf_model.joblib...")
    joblib.dump(rf_model, 'models/rf_model.joblib')
    
    # Save feature column names (needed for the app later)
    print("Saving feature column names to models/features.joblib...")
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, 'models/features.joblib')
    
    print("\nTraining complete!")
    print("- Model saved to: models/rf_model.joblib")
    print("- Features saved to: models/features.joblib")
    
    return rf_model, feature_names


if __name__ == '__main__':
    train_model()
