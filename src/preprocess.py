import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def preprocess_data():
    """
    Preprocess the telco churn dataset.
    
    Steps:
    1. Load data from data/raw/telco_churn.csv
    2. Convert 'TotalCharges' to numeric (coerce errors)
    3. Fill NaNs in 'TotalCharges' with median
    4. Drop 'customerID' column
    5. Encode 'Churn' to 0/1 (No -> 0, Yes -> 1)
    6. One-hot encode categorical variables (drop_first=True)
    7. Split into train/test sets (test_size=0.2, random_state=42)
    8. Save processed sets to data/processed/
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/raw/telco_churn.csv')
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Convert 'TotalCharges' to numeric (coerce errors to NaN)
    print("Converting 'TotalCharges' to numeric...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill NaNs in 'TotalCharges' with median
    median_total_charges = df['TotalCharges'].median()
    print(f"Filling NaNs in 'TotalCharges' with median: {median_total_charges:.2f}")
    ddf['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)
    
    # Drop 'customerID'
    print("Dropping 'customerID' column...")
    df = df.drop('customerID', axis=1)
    
    # Encode 'Churn' to 0/1 (No -> 0, Yes -> 1)
    print("Encoding 'Churn' to 0/1...")
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify categorical columns (exclude numeric columns)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"Categorical columns to encode: {categorical_cols}")
    print(f"Numeric columns: {numeric_cols}")
    
    # One-hot encode categorical variables (drop_first=True)
    print("One-hot encoding categorical variables...")
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    
    print(f"After encoding: {X_encoded.shape[1]} features")
    
    # Split data into train/test sets
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Churn distribution in training set: {y_train.value_counts().to_dict()}")
    print(f"Churn distribution in test set: {y_test.value_counts().to_dict()}")
    
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed datasets
    print("Saving processed datasets...")
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("Preprocessing complete! Files saved to data/processed/")
    print("- X_train.csv")
    print("- X_test.csv")
    print("- y_train.csv")
    print("- y_test.csv")
    
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    preprocess_data()
