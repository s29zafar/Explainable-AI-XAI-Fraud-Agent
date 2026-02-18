import xgboost as xgb
import XGBoostModelParameters.json
import pandas as pd
import numpy as np

def predict(transaction_row, preprocessor):
    """
    Takes a single transaction row (dict or Series), preprocesses it,
    and returns the fraud probability using the trained model and preprocessor.
    """
    filename = 'XGBoostModelParameters.json'
    
    # Load parameters later
    with open(filename, 'r') as file:
        loaded_params = json.load(file)
    
    # Convert to DataFrame if it's a dict or Series
    if not isinstance(transaction_row, pd.DataFrame):
        df_input = pd.DataFrame([transaction_row])
    else:
        df_input = transaction_row.copy()
    # --- Preprocessing Steps matching Training ---
    
    # 1. Drop irrelevant columns
    cols_to_drop = ['month', 'device_fraud_count', 'fraud_bool']
    df_input = df_input.drop(columns=[c for c in cols_to_drop if c in df_input.columns], errors='ignore')
    
    # 2. Convert types (ensure numeric columns are numeric)
    for col in df_input.columns:
        if col not in ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']:
             df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
                
    # 3. Handle Missing Values (Convert -1 to NaN for specific columns)
    missing_cols = [
        "prev_address_months_count",
        "current_address_months_count",
        "bank_months_count",
        "session_length_in_minutes"
    ]
    for col in missing_cols:
        if col in df_input.columns:
             df_input[col] = df_input[col].replace(-1, np.nan)
    
    # 4. Apply OneHotEncoding using the fitted preprocessor
    # Transform returns a pandas DataFrame because of set_output(transform="pandas")
    try:
        X_transformed = preprocessor.transform(df_input)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None
    
    # 5. Convert to numpy array (model was trained on numpy)
    X_numpy = X_transformed.to_numpy()

    # Load Model
    model = xgb.XGBClassifier(**loaded_params)
    model.load_model('XGBoostModel.json')
    
    # --- Inference ---
    # Predict probability of class 1 (Fraud)
    try:
        probability = model.predict_proba(X_numpy)[0, 1]
        return probability
    except Exception as e:
        print(f"Prediction error: {e}")
        return None
    



