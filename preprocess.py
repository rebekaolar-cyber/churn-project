import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def preprocess_data():
    """
    Preprocess telco churn data:
    - Load data from data/raw/telco_churn.csv
    - Convert TotalCharges to numeric and fill NaNs with median
    - Drop customerID
    - Encode Churn to 0/1
    - One-hot encode categorical variables
    - Split into train/test sets
    - Save processed data to data/processed/
    """
    # Load data
    print("Loading data from data/raw/telco_churn.csv...")
    df = pd.read_csv('data/raw/telco_churn.csv')
    
    # Convert TotalCharges to numeric (coerce errors)
    print("Converting TotalCharges to numeric...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill NaNs with median
    print("Filling NaNs in TotalCharges with median...")
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_total_charges, inplace=True)
    
    # Drop customerID
    print("Dropping customerID column...")
    df = df.drop('customerID', axis=1)
    
    # Encode Churn to 0/1
    print("Encoding Churn to 0/1...")
    # Strip whitespace and normalize case to handle edge cases
    df['Churn'] = df['Churn'].str.strip().str.capitalize()
    
    # Check for unexpected values before encoding
    unique_churn_values = df['Churn'].unique()
    expected_values = {'Yes', 'No'}
    unexpected_values = set(unique_churn_values) - expected_values
    
    if unexpected_values:
        raise ValueError(
            f"Unexpected values found in 'Churn' column: {unexpected_values}. "
            f"Expected only 'Yes' or 'No'. Please clean the data first."
        )
    
    # Encode with validation - using replace() instead of map() for better error handling
    # Explicitly set dtype to int to avoid downcasting warnings
    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}).astype(int)
    
    # Verify no NaN values were introduced
    if df['Churn'].isna().any():
        nan_count = df['Churn'].isna().sum()
        raise ValueError(
            f"Encoding resulted in {nan_count} NaN values in 'Churn' column. "
            f"This indicates unexpected values that weren't caught. "
            f"Unique values found: {unique_churn_values}"
        )
    
    # Separate target variable
    y = df['Churn']
    X = df.drop('Churn', axis=1)
    
    # Identify categorical columns (exclude numeric columns)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")
    
    # One-hot encode categorical variables (drop_first=True)
    print("One-hot encoding categorical variables...")
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Split data into train/test sets
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create output directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed datasets
    print("Saving processed datasets to data/processed/...")
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("Preprocessing complete!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    preprocess_data()
