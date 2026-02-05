import pandas as pd
import os

# Get the folder where this script lives (notebooks/)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root (churn-project/), then down to data
# This handles the path correctly no matter where you run it from
file_path = os.path.join(script_dir, '..', 'data', 'processed', 'X_train.csv')

# Resolve to an absolute path to be sure
file_path = os.path.abspath(file_path)

print(f"Checking for file at: {file_path}") # Debug print

if os.path.exists(file_path):
    print(f"✅ Success: File found!")
else:
    print(f"❌ Error: File not found.")
    # List what IS in the processed folder to help debug
    processed_dir = os.path.dirname(file_path)
    if os.path.exists(processed_dir):
        print(f"Files actually in {processed_dir}: {os.listdir(processed_dir)}")
    else:
        print(f"Folder {processed_dir} does not exist!")
    exit()

# 2. Load and Check Data
df = pd.read_csv(file_path)

# 3. Quick Stats (The "Sanity Check")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# 4. Check for Nulls (Requirement: Data should be clean)
missing_values = df.isnull().sum().sum()
if missing_values == 0:
    print("✅ Success: No missing values found.")
else:
    print(f"⚠️ Warning: Found {missing_values} missing values. Check preprocess logic.")

# 5. Check Column Types (Example)
# Ensure dates are actually dates, not strings
# print(df.dtypes) 
