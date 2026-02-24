import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import sys

# Add the current directory to path so we can import agent_logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_logic import Fraud_Agent
except ImportError:
    # Try alternate relative path if needed
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Backend'))
    from agent_logic import Fraud_Agent

def test_fraud_agent():
    print("--- Starting Fraud_Agent Tests ---")
    
    # Paths to required files (adjusting because agent_logic expects them in cwd)
    # If running from Backend/, we might need to symlink or copy these files if they are in root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if required files exist in parent or current
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in ['preprocessor.joblib', 'XGBoostModel.json']:
        if not os.path.exists(f) and os.path.exists(os.path.join(root_dir, f)):
            print(f"Copying {f} from root to Backend/")
            import shutil
            shutil.copy(os.path.join(root_dir, f), f)

    agent = Fraud_Agent()
    print("✓ Agent instantiated")

    # 1. Test load_data
    try:
        print("Testing load_data...")
        df = agent.load_data()
        print(f"✓ load_data successful. Columns: {df.columns.tolist()[:5]}...")
        sample_row = df.iloc[0]
        user_id = sample_row['user_id']
    except Exception as e:
        print(f"✗ load_data failed: {e}")
        return

    # 2. Test ingest_pdf
    try:
        print("Testing ingest_pdf...")
        # Check if PDF exists
        if os.path.exists("Fraud_Detection_Policy.pdf"):
            res = agent.ingest_pdf("Fraud_Detection_Policy.pdf")
            print(f"✓ ingest_pdf result: {res}")
        else:
            print("! Skipping pdf ingestion (Fraud_Detection_Policy.pdf not found)")
    except Exception as e:
        print(f"✗ ingest_pdf failed: {e}")

    # 3. Test preprocess_transaction
    try:
        print("Testing preprocess_transaction...")
        processed = agent.preprocess_transaction(sample_row)
        print(f"✓ preprocess_transaction successful. Output shape: {processed.shape}")
    except Exception as e:
        print(f"✗ preprocess_transaction failed: {e}")

    # 4. Test predict
    try:
        print("Testing predict...")
        prob = agent.predict(sample_row)
        print(f"✓ predict successful. Probability: {prob}")
    except Exception as e:
        print(f"✗ predict failed: {e}")

    # 5. Test get_shap_explanation
    try:
        print("Testing get_shap_explanation...")
        explanation = agent.get_shap_explanation(sample_row)
        print(f"✓ get_shap_explanation successful. Score: {explanation['score']}, Reasons: {explanation['top_reasons']}")
    except Exception as e:
        print(f"✗ get_shap_explanation failed: {e}")

    # 6. Test tools
    try:
        print("Testing get_user_transactions tool...")
        history = agent.get_user_transactions(user_id)
        print(f"✓ get_user_transactions successful. Sample history length: {len(history)}")
        
        print("Testing search_bank_policy tool...")
        policy_snippet = agent.search_bank_policy("fraud detection")
        print(f"✓ search_bank_policy successful. Snippet: {policy_snippet[:100]}...")
    except Exception as e:
        print(f"✗ tools failed: {e}")

    # 7. Test run_fraud_investigation (Agent Flow)
    try:
        print("Testing run_fraud_investigation (Agent flow)...")
        # Check for API key
        if os.path.exists("GoogleAPI.env"):
            with open("GoogleAPI.env", "r") as f:
                if "GOOGLE_API_KEY=" in f.read():
                    memo = agent.run_fraud_investigation(user_id, sample_row)
                    print(f"✓ run_fraud_investigation successful. Memo: {memo}")
                else:
                    print("! Skipping agent flow (API key missing in GoogleAPI.env)")
        else:
            print("! Skipping agent flow (GoogleAPI.env not found)")
    except Exception as e:
        print(f"✗ run_fraud_investigation failed: {e}")

if __name__ == "__main__":
    test_fraud_agent()
