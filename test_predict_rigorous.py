import unittest
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import shutil
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import Predict  # The module to test

class TestPredictRigorous(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Setup a dummy environment for testing:
        1. Create a dummy model parameters file.
        2. Train a small dummy XGBoost model and save it.
        3. Create a fitted preprocessor.
        """
        print("Setting up test environment...")
        
        # 1. Dummy Parameters
        cls.params_file = 'XGBoostModelParameters_test.json'
        cls.model_file = 'XGBoostModel_test.json'
        
        cls.params = {
            "objective": "binary:logistic",
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "use_label_encoder": False
        }
        with open(cls.params_file, 'w') as f:
            json.dump(cls.params, f)
            
        # 2. Dummy Data & Preprocessor
        # Schema matching expected inputs roughly
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [10, 20, 30, 40],
            'payment_type': ['AA', 'BB', 'AA', 'CC'], # Categorical
            'month': [1, 2, 3, 4], # Should be dropped
            'fraud_bool': [0, 1, 0, 1] # Target/Should be dropped
        })
        
        # Define preprocessor (OneHotEncoder for categorical)
        categorical_cols = ['payment_type']
        ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        cls.preprocessor = ColumnTransformer(
            transformers=[("cat", ohe, categorical_cols)],
            remainder="passthrough"
        )
        cls.preprocessor.set_output(transform="pandas")
        
        # Fit preprocessor
        X = data.drop(columns=['fraud_bool', 'month'])
        cls.X_processed = cls.preprocessor.fit_transform(X)
        y = data['fraud_bool']
        
        # 3. Train & Save Dummy Model
        cls.model = xgb.XGBClassifier(**cls.params)
        cls.model.fit(cls.X_processed, y)
        cls.model.save_model(cls.model_file)
        
    @classmethod
    def tearDownClass(cls):
        """Cleanup generated files."""
        print("Cleaning up test environment...")
        if os.path.exists(cls.params_file):
            os.remove(cls.params_file)
        if os.path.exists(cls.model_file):
            os.remove(cls.model_file)

    def test_preprocess_transaction(self):
        """Test the preprocessing logic in isolation."""
        
        # Input with extra cols and wrong types
        row = {
            'feature1': '5.5', # String to float
            'feature2': 50,
            'payment_type': 'AA',
            'month': 5, # Should drop
            'device_fraud_count': 0, # Should drop
            'prev_address_months_count': -1 # Should become NaN
        }
        
        # Test helper directly if possible, else test via public functions
        # Predict.py exposes preprocess_transaction? Yes.
        processed = Predict.preprocess_transaction(row, self.preprocessor)
        
        self.assertIsInstance(processed, pd.DataFrame)
        
        # Check dropped columns
        self.assertNotIn('month', processed.columns)
        self.assertNotIn('device_fraud_count', processed.columns)
        
        # Check type conversion
        # Pass through features (feature1, feature2) + OneHot (payment_type_BB, payment_type_CC)
        # feature1 should be float
        # feature2 should be int/float
        # prev_address_months_count should be NaN
        
        # Note: Preprocessor output names depend on sklearn version
        # OneHot 'payment_type' (drop first 'AA') -> payment_type_BB, payment_type_CC
        # Remainder -> feature1, feature2, prev_address_months_count
        
        cols = processed.columns
        self.assertTrue(any('feature1' in c for c in cols) or 'remainder__feature1' in cols)
        
        # Check NaN handling (if column exists in output)
        # It might be in 'remainder' part
        if 'remainder__prev_address_months_count' in processed.columns:
            val = processed['remainder__prev_address_months_count'].iloc[0]
            self.assertTrue(np.isnan(val))

    def test_predict_function(self):
        """Test the predict function with valid input."""
        row = {
            'feature1': 1.0,
            'feature2': 10,
            'payment_type': 'AA'
        }
        
        # We need to pass our dummy file paths
        prob = Predict.predict(
            row, 
            self.preprocessor, 
            model_params_path=self.params_file, 
            model_path=self.model_file
        )
        
        self.assertIsNotNone(prob)
        self.assertIsInstance(prob, float)
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_shap_explanation(self):
        """Test SHAP explanation generation."""
        row = {
            'feature1': 2.0,
            'feature2': 20,
            'payment_type': 'BB'
        }
        
        explanation = Predict.get_shap_explanation(
            row, 
            self.model, 
            self.preprocessor
        )
        
        # Check structure
        self.assertIn('score', explanation)
        self.assertIn('top_reasons', explanation)
        
        # Check score matches model prediction
        # (Preprocessing creates a DataFrame row, our manual prediction matches)
        processed = Predict.preprocess_transaction(row, self.preprocessor)
        expected_prob = self.model.predict_proba(processed)[0, 1]
        self.assertAlmostEqual(explanation['score'], expected_prob, places=4)
        
        # Check reasons
        self.assertIsInstance(explanation['top_reasons'], list)
        self.assertTrue(len(explanation['top_reasons']) <= 3)
        self.assertTrue(len(explanation['top_reasons']) > 0)
        
        # Check formatting "Feature = Value"
        reason = explanation['top_reasons'][0]
        self.assertIn('=', reason)

    def test_missing_file_handling(self):
        """Test behavior when model file doesn't exist."""
        row = {'feature1': 1}
        prob = Predict.predict(
            row, 
            self.preprocessor, 
            model_params_path='non_existent.json', 
            model_path='non_existent.model'
        )
        self.assertIsNone(prob)

if __name__ == '__main__':
    unittest.main()
