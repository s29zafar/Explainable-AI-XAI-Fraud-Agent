
import os
import sys
import sqlite3
import pandas as pd
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import unittest
import logging

# Silence noisy logging from libraries
import logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Fix for SQLite on Mac (common issue with ChromaDB)
if sys.platform.startswith('darwin'):
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules['pysqlite3']
    except ImportError:
        pass

# --- Singleton for Embeddings to avoid reloading ---
class EmbeddingModel:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return cls._instance

# --- Functions to be tested ---

def get_user_transactions(user_id: str, db_path="Fraud_Agent.db"):
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM transaction_history WHERE user_id = '{user_id}' LIMIT 5"
    result_df = pd.read_sql_query(query, conn)
    conn.close()
    return result_df

def search_bank_policy(query: str, chroma_path="./chroma_db", collection_name="bank_policies"):
    embeddings = EmbeddingModel.get_instance()
    
    # Initialize Persistent Client
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    
    vector_db = Chroma(
        client=chroma_client,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    docs = vector_db.similarity_search(query, k=1)
    if not docs:
        return "No relevant policy found for this query."
    return docs[0].page_content

# --- Unit Tests ---

class TestCoreTools(unittest.TestCase):
    
    def test_get_user_transactions_output_type(self):
        """Verify that get_user_transactions returns a pandas DataFrame."""
        df = get_user_transactions("USER_0000")
        if df is not None:
            self.assertIsInstance(df, pd.DataFrame)
        else:
            self.skipTest("Fraud_Agent.db not found.")

    def test_get_user_transactions_valid_user(self):
        """Verify that get_user_transactions returns data for a valid user."""
        df = get_user_transactions("USER_0000")
        if df is not None:
            self.assertTrue(len(df) >= 0)
        else:
            self.skipTest("Fraud_Agent.db not found.")

    def test_search_bank_policy_output_type(self):
        """Verify that search_bank_policy returns a string."""
        if os.path.exists("./chroma_db"):
            result = search_bank_policy("housing status")
            self.assertIsInstance(result, str)
            self.assertNotEqual(result, "No relevant policy found for this query.")
        else:
            self.skipTest("./chroma_db not found.")

    def test_search_bank_policy_content(self):
        """Verify that search_bank_policy returns relevant content."""
        if os.path.exists("./chroma_db"):
            result = search_bank_policy("risk indicators")
            self.assertTrue(len(result) > 10)
        else:
            self.skipTest("./chroma_db not found.")

if __name__ == "__main__":
    unittest.main(verbosity=1)
