"""
Preprocessing script for Telco Customer Churn dataset.

This script loads raw data, performs preprocessing transformations,
and saves the processed training and test sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(filepath)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the telco churn dataset.
    
    Steps:
    1. Convert 'TotalCharges' to numeric (coerce errors)
    2. Fill NaNs in 'TotalCharges' with median
    3. Drop 'customerID' column
    4. Encode 'Churn' to 0/1
    5. One-hot encode categorical variables (drop_first=True)
    """
    df = df.copy()
    
    # Convert TotalCharges to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill NaN values with median
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)
    
    # Drop customerID column
    df = df.drop(columns=['customerID'])
    
    # Encode Churn to 0/1
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    
    # Identify categorical columns (object dtype) for one-hot encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categorical variables with drop_first=True
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df


def split_data(df: pd.DataFrame, target_col: str = 'Churn', 
               test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and test sets.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series,
                        output_dir: str) -> None:
    """Save processed training and test sets to CSV files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine features and target for saving
    train_data = X_train.copy()
    train_data['Churn'] = y_train
    
    test_data = X_test.copy()
    test_data['Churn'] = y_test
    
    # Save to CSV
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Training data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")


def main():
    """Main preprocessing pipeline."""
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'telco_churn.csv')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    
    print("Loading data...")
    df = load_data(raw_data_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    print("\nPreprocessing data...")
    df_processed = preprocess_data(df)
    print(f"After preprocessing: {len(df_processed)} rows and {len(df_processed.columns)} columns")
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    print("\nSaving processed data...")
    save_processed_data(X_train, X_test, y_train, y_test, processed_data_dir)
    
    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()
