import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from prometheus_client import make_asgi_app

# OTel Configuration
resource = Resource.create({"service.name": "fraud-agent-backend"})
tracer_provider = TracerProvider(resource=resource)
# Check if OTLP endpoint is provided
if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    otlp_exporter = OTLPSpanExporter()
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

trace.set_tracer_provider(tracer_provider)

# Instrument LangChain
LangChainInstrumentor().instrument()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_logic import Fraud_Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Agent API")

# Expose Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Initialize global Fraud_Agent
# Note: In a production app, you might want to use lifespan events or DI
fraud_agent = Fraud_Agent()

# Define what the input should look like
class TransactionRequest(BaseModel):
    tx_id: str

@app.get("/")
def home():
    return {"message": "Fraud Agent Backend is running"}

@app.post("/investigate")
async def investigate(item: TransactionRequest):
    logger.info(f"Investigating transaction: {item.tx_id}")
    
    # 1. Fetch transaction details from the database
    transaction = fraud_agent.get_transaction_by_id(item.tx_id)
    
    if transaction is None:
        logger.warning(f"Transaction {item.tx_id} not found")
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # 2. Extract user_id
    user_id = transaction['user_id']
    
    try:
        # 3. Call the LangGraph agent for investigation
        result = fraud_agent.run_fraud_investigation(user_id, transaction)
        
        # Return the rich dictionary directly
        return {
            "tx_id": item.tx_id,
            "user_id": user_id,
            "probability": result["probability"],
            "shap_values": result["shap_values"],
            "agent_memo": result["agent_memo"],
            "policy_reference": result["policy_reference"]
        }
    except Exception as e:
        logger.error(f"Agent investigation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Investigation failed: {str(e)}")