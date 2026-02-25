import streamlit as st
import requests
import os

st.set_page_config(page_title="Fraud Investigator Portal", layout="wide")

st.title("🛡️ AI Fraud Investigation Console")
st.markdown("---")

# 1. Input Section
tx_id = st.text_input("Enter Transaction ID to scan:", value="TXN_00000000")

if st.button("Start AI Investigation"):
    with st.spinner("Agent is analyzing history and policies..."):
        # 2. Call the FastAPI Backend
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        try:
            response = requests.post(f"{backend_url}/investigate", json={"tx_id": tx_id})
            
            if response.status_code == 200:
                data = response.json()
                
                # 3. Display Results in Columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Model Prediction")
                    st.metric("Fraud Probability", f"{data['probability']:.2%}")
                    
                    st.write("Top SHAP Contributing Factors")
                    if data['shap_values']:
                        st.bar_chart(data['shap_values']) 
                    else:
                        st.write("No SHAP values available.")

                with col2:
                    st.subheader("Agent Reasoning (XAI)")
                    st.info(data['agent_memo'])
                    
                    with st.expander("View Policy Citations"):
                        st.markdown(data['policy_reference'])
            elif response.status_code == 404:
                st.warning(f"Transaction ID {tx_id} not found in the database. Please check and try again.")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")