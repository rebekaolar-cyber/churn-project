import pandas as pd
import os

# 1. Check if file exists
file_path = 'data/processed/your_processed_file.csv' # CHANGE THIS to your actual filename
if os.path.exists(file_path):
    print(f"✅ Success: File found at {file_path}")
else:
    print(f"❌ Error: File not found at {file_path}")
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
import pandas as pd
import os

# 1. Check if file exists
file_path = 'data/processed/your_processed_file.csv' # CHANGE THIS to your actual filename
if os.path.exists(file_path):
    print(f"✅ Success: File found at {file_path}")
else:
    print(f"❌ Error: File not found at {file_path}")
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
