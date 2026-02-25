import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
import shap
import sqlite3
import os
import logging
from faker import Faker
import kagglehub
from dotenv import load_dotenv

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    import chromadb
    from langchain_chroma import Chroma
    HAS_CHROMA = True
except (ImportError, Exception) as e:
    print(f"Warning: ChromaDB or related modules failed to load. Vector features will be disabled. Error: {e}")
    HAS_CHROMA = False
    Chroma = None

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool 
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

class Fraud_Agent():
    def __init__(self):
        self.CHROMA_PATH = "./chroma_db"
        self.CHROMA_COLLECTION_NAME = "bank_policies"
        self.preprocessor = joblib.load('preprocessor.joblib')
        self.model = xgb.XGBClassifier()
        self.model.load_model('XGBoostModel.json')
    
    # Phase 1: Preprocessing
    def preprocess_transaction(self, transaction_row, preprocessor=None):
        """
        Helper function to preprocess a single transaction row.
        Returns processed DataFrame.
        """
        if preprocessor is None:
            preprocessor = self.preprocessor

        # Convert to DataFrame if it's a dict or Series
        if not isinstance(transaction_row, pd.DataFrame):
            df_input = pd.DataFrame([transaction_row])
        else:
            df_input = transaction_row.copy()
        
        # 1. Drop irrelevant columns
        cols_to_drop = ['month', 'device_fraud_count', 'fraud_bool', 'user_id']
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
                # If preprocessor returns a sparse matrix or numpy array, convert to DataFrame
                if not isinstance(X_transformed, pd.DataFrame):
                    feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else None
                    X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
                return X_transformed
            except Exception as e:
                print(f"Preprocessing error: {e}")
                return None
        else:
            return df_input

    # Phase 1: Prediction
    def predict(self, transaction_row, preprocessor=None, model=None):
        """
        Takes a single transaction row, preprocesses it, and returns the fraud probability.
        """
        if preprocessor is None:
            preprocessor = self.preprocessor
        if model is None:
            model = self.model

        X_transformed = self.preprocess_transaction(transaction_row, preprocessor)
        if X_transformed is None:
            return None
        
        X_numpy = X_transformed.to_numpy()

        # Inference
        try:
            probability = model.predict_proba(X_numpy)[0, 1]
            return float(probability)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    # Phase 1: Explanation
    def get_shap_explanation(self, transaction_data, model=None, preprocessor=None):
        """
        Generates a SHAP explanation for a single transaction.
        Returns a dictionary with fraud probability and top 3 contributing features.
        """
        if model is None:
            model = self.model
        if preprocessor is None:
            preprocessor = self.preprocessor

        # Preprocess
        X_transformed = self.preprocess_transaction(transaction_data, preprocessor)
        if X_transformed is None:
            return {"error": "Preprocessing failed"}
        
        # Ensure we use DataFrame for column names in SHAP
        feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else X_transformed.columns
        X_df = pd.DataFrame(X_transformed, columns=feature_names)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_df)
        
        row_values = shap_values.values[0]
        data_values = X_df.iloc[0]
        
        # Calculate probability
        prob = model.predict_proba(X_df)[0, 1]
        
        # Identify top 3 features pushing score HIGHER (positive contribution to fraud class)
        contributions = []
        
        # Handle multi-class output shape if SHAP returns (n_rows, n_features, 2)
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
        # ensure we point to a .csv file
        csv_path = os.path.join(path, "Base.csv")

        # read the CSV into a DataFrame and setup the final test data
        df_OG = pd.read_csv(csv_path)
        mask = df_OG["month"] == 7
        full_test_data = df_OG[mask].sample(frac=0.5, random_state=42).reset_index(drop=True)
        df = full_test_data.copy()

        # Add Fake User IDs and Transaction IDs
        fake = Faker()
        num_users = 100 # Let's say we have 100 unique users
        user_ids = [f"USER_{i:04d}" for i in range(num_users)]
        df['user_id'] = np.random.choice(user_ids, size=len(df))
        df['tx_id'] = [f"TXN_{i:08d}" for i in range(len(df))]

        # Setup a Table in SQL
        table_name = "transaction_history"
        df.to_sql(table_name, connection, if_exists='replace', index=False)

        # Verify the data was written by reading it back into a new DataFrame
        query = f"SELECT * FROM {table_name}"
        result_df = pd.read_sql_query(query, connection)

        # Close the database connection
        connection.close()
        return result_df

    def ingest_pdf(self, pdf_path="Fraud_Detection_Policy.pdf"):
        if not HAS_CHROMA:
            return "ChromaDB not available. PDF ingestion skipped."
        # 1. Load PDF
        if not os.path.exists(pdf_path):
            print(f"Error: {pdf_path} not found.")
            return

        print(f"Loading {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
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
        chroma_client = chromadb.PersistentClient(path=self.CHROMA_PATH)

        # 5. Add to Chroma
        print(f"Adding documents to collection '{self.CHROMA_COLLECTION_NAME}'...")
        Chroma.from_documents(
            documents=chunked_documents,
            embedding=embedding_function,
            collection_name=self.CHROMA_COLLECTION_NAME,
            client=chroma_client,
        )
        
        print(f"Successfully added {len(chunked_documents)} chunks to ChromaDB at {self.CHROMA_PATH}")

        return "PDF ingested successfully."

    def get_transaction_by_id(self, tx_id: str):
        """
        Fetch a single transaction by its ID from the database.
        """
        db_path = "Fraud_Agent.db"
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM transaction_history WHERE tx_id = '{tx_id}'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return None
        return df.iloc[0]


    def get_user_transactions(self, user_id: str):
        """
        Randomly select 5 transactions for this user.
        Use this tool when you need to verify if a user has a history of fraud transactions.
        """
        db_path = "Fraud_Agent.db"
        conn = sqlite3.connect(db_path)
        
        # SELECT any 5 transactions of the user at random
        query = f"SELECT * FROM transaction_history WHERE user_id = '{user_id}' LIMIT 5"
        result_df = pd.read_sql_query(query, conn)
        conn.close()

        return result_df.to_string() # LangChain tools usually prefer strings or serializable objects

    def search_bank_policy(self, query: str) -> str:
        """
        Searches the official Bank Anti-Fraud Policy documentation. 
        Use this tool when you need to verify if a flagged transaction 
        violates specific banking regulations or internal risk thresholds.
        """
        if not HAS_CHROMA:
            return "ChromaDB not available. Unable to search policies."
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        chroma_client = chromadb.PersistentClient(path=self.CHROMA_PATH)

        vector_db = Chroma(
            client=chroma_client,
            embedding_function=embeddings,
            collection_name=self.CHROMA_COLLECTION_NAME
        )
        
        docs = vector_db.similarity_search(query, k=1)
        if not docs:
            return "No relevant policy found for this query."
        
        return docs[0].page_content

    # Phase 4: Agent
    def run_fraud_investigation(self, user_id, transaction_row):
        explanation = self.get_shap_explanation(transaction_row, self.model, self.preprocessor)
        
        # Load environment variables
        load_dotenv("Backend/GoogleAPI.env") 
        google_api_key = os.getenv("GOOGLE_API_KEY")

        # System Prompt
        system_prompt = """You are a Senior Fraud Investigator.
        You will be given a transaction and its SHAP explanations.
        Use your tools to check the user's history and bank policy.
        Then, write a 3-sentence memo concluding if it is fraud or not."""

        # 1. Define Tools with self bound
        @tool
        def get_user_transactions_tool(user_id: str):
            """Get transaction history for a user."""
            return self.get_user_transactions(user_id)

        @tool
        def search_bank_policy_tool(query: str):
            """Search bank anti-fraud policies."""
            return self.search_bank_policy(query)

        tools = [get_user_transactions_tool, search_bank_policy_tool]

        # 2. Initialize the model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            api_key=google_api_key,
            temperature=0
        )

        # 3. Initialize the agent (using langgraph prebuilt)
        agent = create_react_agent(llm, tools, prompt=system_prompt)

        # 4. Run the agent
        reasons_str = "\n".join([f"- {r}" for r in explanation.get('top_reasons', [])])
        query = f"""
        Investigate transaction for {user_id}.
        Model Fraud Probability: {explanation.get('score', 0):.2f}
        Top SHAP Contributing Factors:
        {reasons_str}

        Please check the user's transaction history and cross-reference with bank policy to generate a final memo.
        """
        
        result = agent.invoke({"messages": [("user", query)]})
        
        # 5. Extract additional info for frontend
        agent_memo = result["messages"][-1].content
        
        # Try to find if the agent called search_bank_policy tool to get policy reference
        policy_reference = "No specific policy citations found."
        for msg in reversed(result["messages"]):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc['name'] == 'search_bank_policy_tool':
                        # The next message in the list should be the ToolMessage with the result
                        msg_idx = result["messages"].index(msg)
                        if msg_idx + 1 < len(result["messages"]):
                            policy_reference = result["messages"][msg_idx + 1].content
                        break
        
        # Prepare SHAP values for chart (top 5 reasons)
        high_level_shap = {r.split(' = ')[0]: float(r.split(' = ')[1]) if ' = ' in r and r.split(' = ')[1].replace('.','').isdigit() else 1.0 
                          for r in explanation.get('top_reasons', [])}

        return {
            "probability": explanation.get('score', 0),
            "shap_values": high_level_shap,
            "agent_memo": agent_memo,
            "policy_reference": policy_reference
        }
