import xgboost as xgb
import json
import pandas as pd
import numpy as np
import shap

def preprocess_transaction(transaction_row, preprocessor=None):
    """
    Helper function to preprocess a single transaction row.
    Returns processed DataFrame.
    """
    # Convert to DataFrame if it's a dict or Series
    if not isinstance(transaction_row, pd.DataFrame):
        df_input = pd.DataFrame([transaction_row])
    else:
        df_input = transaction_row.copy()
        
    # 1. Drop irrelevant columns
    cols_to_drop = ['month', 'device_fraud_count', 'fraud_bool']
    df_input = df_input.drop(columns=[c for c in cols_to_drop if c in df_input.columns], errors='ignore')
    
    # 2. Convert types
    for col in df_input.columns:
        if col not in ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']:
             df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
                
    # 3. Handle Missing Values
    missing_cols = [
        "prev_address_months_count", "current_address_months_count",
        "bank_months_count", "session_length_in_minutes"
    ]
    for col in missing_cols:
        if col in df_input.columns:
             df_input[col] = df_input[col].replace(-1, np.nan)
    
    # 4. Apply OneHotEncoding
    if preprocessor:
        try:
            X_transformed = preprocessor.transform(df_input)
            return X_transformed
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    else:
        return df_input

def predict(transaction_row, preprocessor, model_params_path='XGBoostModelParameters.json', model_path='XGBoostModel.json'):
    """
    Takes a single transaction row, preprocesses it, and returns the fraud probability.
    """
    # Load parameters
    try:
        with open(model_params_path, 'r') as file:
            loaded_params = json.load(file)
    except FileNotFoundError:
        print(f"Error: {model_params_path} not found.")
        return None

    X_transformed = preprocess_transaction(transaction_row, preprocessor)
    if X_transformed is None:
        return None
    
    X_numpy = X_transformed.to_numpy()

    # Load Model (Note: This assumes model file exists)
    try:
        model = xgb.XGBClassifier(**loaded_params)
        model.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Inference
    try:
        probability = model.predict_proba(X_numpy)[0, 1]
        return float(probability)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def get_shap_explanation(transaction_data, model, preprocessor):
    """
    Generates a SHAP explanation for a single transaction.
    Returns a dictionary with fraud probability and top 3 contributing features.
    """
    # Preprocess
    X_transformed = preprocess_transaction(transaction_data, preprocessor)
    if X_transformed is None:
        return {"error": "Preprocessing failed"}
    
    # Ensure we use DataFrame for column names in SHAP
    feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else X_transformed.columns
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_df)
    
    # Get values for the first (and only) row
    # shap_values.values shape is (1, n_features)
    # Binary classification: some shap versions output values for both classes, some just one.
    # For XGBClassifier binary, it usually outputs log-odds for class 1.
    
    row_values = shap_values.values[0]
    # base_value = shap_values.base_values[0] # Not strictly needed for top 3
    data_values = X_df.iloc[0]
    
    # Calculate probability
    prob = model.predict_proba(X_df)[0, 1]
    
    # Identify top 3 features pushing score HIGHER (positive contribution to fraud class)
    # We want features that increase the probability of fraud.
    
    # Create list of (feature_name, shap_value, feature_value)
    contributions = []
    
    # Handle multi-class output shape if SHAP returns (1, n_features, 2)
    if len(row_values.shape) > 1:
        # Assuming class 1 is index 1
        row_values = row_values[:, 1]

    for name, val, feat_val in zip(X_df.columns, row_values, data_values):
        contributions.append((name, val, feat_val))
    
    # Sort by SHAP value descending (highest positive impact first)
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    top_3 = contributions[:3]
    
    top_reasons = []
    for name, val, feat_val in top_3:
        # Clean up feature name (remove 'cat__' etc if present)
        clean_name = str(name).replace('cat__', '').replace('remainder__', '')
        
        # Format based on value type
        if isinstance(feat_val, (int, float)):
             reason = f"{clean_name} = {feat_val:.2f}"
        else:
             reason = f"{clean_name} = {feat_val}"
        
        top_reasons.append(reason)
        
    return {
        "score": float(prob),
        "top_reasons": top_reasons
    }
