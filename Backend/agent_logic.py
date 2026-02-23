import pandas as pd, numpy as np, json, joblib, xgboost as xgb, shap, sqlite3, os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
import chromadb, json, logging, sys
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import sqlite3, kagglehub
from faker import Faker
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Fraud_Agent():
    def __init__(self):
        self.CHROMA_PATH = "./chroma_db"
        self.CHROMA_COLLECTION_NAME = "bank_policies"
        self.preprocessor = joblib.load('preprocessor.joblib')
        self.model = xgb.XGBClassifier()
        self.model.load_model('XGBoostModel.json')
    
    # Phase 1: Preprocessing
    def preprocess_transaction(transaction_row, preprocessor=self.preprocessor):
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

    # Phase 1: Prediction
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
    
    # Phase 1: Explanation
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
    
    # Phase 2: Data Environment
    def load_data(self):
        """
        Load data from SQLite database.
        """
        # Setup SQLLite connection 
        connection = sqlite3.connect("Fraud_Agent.db")
        # Download latest version
        path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")
        # ensure we point to a .csv file (dataset_download may return a path without extension)
        csv_path = str(path) + "/Base.csv"

        # read the CSV into a DataFrame and setup the final test data
        df_OG = pd.read_csv(csv_path)
        mask = df_OG["month"] == 7
        full_test_data = df_OG[mask].sample(frac=0.5).reset_index(drop=True).drop('month',axis=1) 
        df = full_test_data

        # Add 
        fake = Faker()
        num_users = len(df) // 8
        user_ids = [f"USER_{i:04d}" for i in range(num_users)]

        df['user_id'] = (user_ids * 9)[:len(df)]

        # Setup a Table in SQL
        table_name = "transaction_history"
        full_test_data.to_sql(table_name, connection, if_exists='replace', index=False)

        # Verify the data was written by reading it back into a new DataFrame
        query = f"SELECT * FROM {table_name}"
        result_df = pd.read_sql_query(query, connection)

        # Close the database connection
        connection.close()
        return result_df

    def ingest_pdf():
        # 1. Load PDF
        if not os.path.exists(PDF_PATH):
            print(f"Error: {PDF_PATH} not found.")
            return

        print(f"Loading {PDF_PATH}...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        # 2. Split Text
        print("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunked_documents = text_splitter.split_documents(documents)
    
        print(f"Created {len(chunked_documents)} chunks.")

        # 3. Initialize Embeddings
        print("Initializing embeddings...")
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 4. Initialize Chroma Client
        print("Initializing ChromaDB client...")
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        # 5. Add to Chroma
        print(f"Adding documents to collection '{CHROMA_COLLECTION_NAME}'...")
        Chroma.from_documents(
            documents=chunked_documents,
            embedding=embedding_function,
            collection_name=CHROMA_COLLECTION_NAME,
            client=chroma_client,
        )
        
        print(f"Successfully added {len(chunked_documents)} chunks to ChromaDB at {CHROMA_PATH}")

        return "PDF ingested successfully."


    # Phase 3: Tools
    @tool
    def get_user_transactions(user_id: str):
        """
        Randomly select 5 transactions for this user.
        Use this tool when you need to verify if a user has a history of fraud
        transactions.
        """
        db_path = "Fraud_Agent.db"

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # SELECT any 5 transactions of the user at random
        query = f"SELECT * FROM transaction_history WHERE user_id = '{user_id}' LIMIT 5"
        result_df = pd.read_sql_query(query, conn)
        print("\nData read from SQLite table:")

        conn.close()

        return result_df

    @tool
    def search_bank_policy(query: str) -> str:
        """
        Searches the official Bank Anti-Fraud Policy documentation. 
        Use this tool when you need to verify if a flagged transaction 
        violates specific banking regulations or internal risk thresholds.
        """
        # Consistency: Use same embeddings and paths as ingestion
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize Persistent Client
        chroma_client = chromadb.PersistentClient(path=self.CHROMA_PATH)

        vector_db = Chroma(
            client=chroma_client,
            embedding_function=embeddings,
            collection_name=self.CHROMA_COLLECTION_NAME
        )
        
        # Perform the Similarity Search
        # k=1 returns only the single most relevant paragraph
        docs = vector_db.similarity_search(query, k=1)
        
        if not docs:
            return "No relevant policy found for this query."
        
        # Return the text content of the best match
        return docs[0].page_content

    # Test the tool manually
    try:
        # result = search_bank_policy.invoke("What is the limit for overseas wire transfers?")
        result = search_bank_policy.invoke("housing status")
        print(f"Policy Found: {result}")
    except Exception as e:
        print(f"Error during search: {e}")

    # Phase 4: Agent
    def run_fraud_investigation(self, transaction_data):
        
        pass
