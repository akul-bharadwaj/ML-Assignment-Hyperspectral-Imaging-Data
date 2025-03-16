import streamlit as st
import pandas as pd
import requests

# API Endpoint
API_URL = "http://127.0.0.1:8000/predict"

st.title("DON Concentration Prediction App")
st.write("Upload a CSV file containing spectral data to get predictions.")

# File Upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.head())
    
    # Ensure 'hsi_id' column exists
    if "hsi_id" in df.columns:
        selected_id = st.selectbox("Select an hsi_id for prediction", df["hsi_id"].unique())
        
        if st.button("Get Prediction"):
            response = requests.post(API_URL, json={"hsi_id": selected_id})
            if response.status_code == 200:
                prediction = response.json()
                st.success("Prediction Results")
                st.json(prediction)
            else:
                st.error("Error: Unable to fetch predictions.")
    else:
        st.error("CSV must contain an 'hsi_id' column.")
