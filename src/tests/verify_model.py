import joblib
import os
import pandas as pd
import numpy as np  # Added for safe dummy data creation

# --- 1. Robust Path Setup ---
# Get the directory where this script lives (e.g., .../churn-project/src/tests)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up TWO levels to get to the project root (churn-project/)
# Logic: src/tests/ -> src/ -> root/
project_root = os.path.join(script_dir, '..', '..')

# Define paths to the model files
model_path = os.path.join(project_root, 'models', 'rf_model.joblib')
features_path = os.path.join(project_root, 'models', 'features.joblib')

# Normalize paths (cleans up the '..' parts)
model_path = os.path.abspath(model_path)
features_path = os.path.abspath(features_path)

print(f"🔎 DEBUG: Looking for model file at: {model_path}")

# --- 2. Check File Existence ---
if os.path.exists(model_path) and os.path.exists(features_path):
    print(f"✅ Success: Model and feature files found.")
else:
    print(f"❌ Error: Model files missing.")

    # Debug helper: Check if 'models' folder even exists
    models_dir = os.path.dirname(model_path)
    if os.path.exists(models_dir):
        print(f"📂 Files actually in 'models' folder: {os.listdir(models_dir)}")
    else:
        print(f"⚠️ Warning: The folder '{models_dir}' does not exist!")
    exit()

# --- 3. Load the Model ---
try:
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    print(f"✅ Success: Model loaded correctly.")
    print(f"   - Model Type: {type(model).__name__}")
    print(f"   - Number of Features Expected: {len(features)}")
except Exception as e:
    print(f"❌ Error loading model content: {e}")
    exit()

# --- 4. Test Prediction (Smoke Test) ---
print("Running dummy prediction test...")
try:
    # Create a single row of zeros with the correct feature names
    # (This ensures the model accepts the input shape)
    dummy_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)

    # Make a prediction
    prediction = model.predict(dummy_data)
    probability = model.predict_proba(dummy_data)

    print(f"✅ Success: Model successfully generated a prediction.")
    print(f"   - Prediction: {prediction[0]} (0=No Churn, 1=Churn)")
    print(f"   - Probability: {probability[0]}")

except Exception as e:
    print(f"⚠️ Warning: Prediction test failed. Error: {e}")
