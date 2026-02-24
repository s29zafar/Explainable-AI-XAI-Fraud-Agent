from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_logic import run_fraud_investigation # Import your Phase 4 function

app = FastAPI(title="Fraud Agent API")

# Define what the input should look like
class TransactionID(BaseModel):
    tx_id: str

@app.get("/")
def home():
    return {"message": "Fraud Agent Backend is running"}

@app.post("/investigate")
async def investigate(item: TransactionID):
    try:
        # This calls your LangGraph agent
        result = run_fraud_investigation(item.tx_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))