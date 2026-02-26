# Explainable-AI-XAI-Fraud-Agent

## 🛡️ Agentic XAI Fraud Investigator

An End-to-End Explainable AI System for Automated Financial Risk Oversight. I may host it via netify in the future, but for now this is a GitHub repo. 

###The Problem

Traditional fraud detection models are "Black Boxes." While they may have high accuracy, human investigators often struggle to understand why a transaction was flagged, leading to high manual review times and customer friction (False Positives).

#### The Solution
This project builds an Agentic AI System that doesn't just score fraud—it investigates it. By combining an XGBoost classification engine with SHAP explainability and a LangGraph-driven LLM Agent, the system provides a natural language investigative memo for every high-risk alert, cross-referencing internal bank policies and historical user behavior.🚀 

## Key Technical Highlights
Precision-Engineered Model: Optimized an XGBoost classifier for a highly imbalanced dataset (1% fraud rate), achieving an $F_{2}$ score of 0.3463 to prioritize recall and minimize financial loss.
Explainable AI (XAI): Integrated SHAP (SHapley Additive exPlanations) to decompose model predictions into feature-level contributions.
RAG-Powered Compliance: Built a Retrieval-Augmented Generation (RAG) pipeline using ChromaDB to ensure the AI Agent's decisions align with official Bank Anti-Fraud Policies.
Synthetic Identity Layer: Engineered a relational schema in SQLite to simulate longitudinal user behavior, allowing the agent to perform historical pattern matching.
Production-Ready Stack: Deployed as a microservice architecture using FastAPI and Streamlit, fully containerized via Docker.🛠️ 

## Tech Stack
ML/Math: Python, XGBoost, Scikit-learn, SHAPAgentic 
AI: LangGraph, LangChain, OpenAI GPT-4oData/Vector: SQLite, ChromaDB
Engineering: FastAPI, Streamlit, Docker, Docker Compose

## 📖 System Architecture

Detection: A transaction is processed by the XGBoost model.
Attribution: If the score exceeds a threshold, SHAP values identify the "Top 3" risk drivers.
Investigation: The LangGraph Agent receives the alert and autonomously triggers
  tools:get_user_history: Queries SQLite for behavioral anomalies.
  search_bank_policy: Queries ChromaDB for regulatory alignment.
Reporting: The agent synthesizes a professional investigative memo.

## Quick Start (Docker)

Ensure you have an .env file with your OPENAI_API_KEY.

Bash:

git clone https://github.com/yourusername/fraud-agent.git 

cd fraud-agent

docker-compose up --build

Frontend: http://localhost:8501API
Docs: http://localhost:8000/docs

<img width="1436" height="816" alt="Screenshot 2026-02-25 at 6 55 10 PM" src="https://github.com/user-attachments/assets/6032109e-ad69-455d-a77e-22bbcc3250a1" />
