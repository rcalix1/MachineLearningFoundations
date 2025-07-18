import streamlit as st
import pandas as pd
import requests

st.title("Steel Plant Silicon Prediction")

file = st.file_uploader("Upload your CSV file")
if file:
    df = pd.read_csv(file)
    st.dataframe(df)

    if st.button("Predict"):
        try:
            # Wrap file correctly for requests
            files = {"file": (file.name, file.getvalue())}
            res = requests.post(
                "http://10.0.0.139:8000/predict",  # Your GPU server IP
                files=files,
                headers={"x-api-key": "secret123"}  # Required API key
            )

            # Show response status and text for debugging
            st.write("Status Code:", res.status_code)
            st.write("Raw Response:", res.text)

            if res.status_code == 200:
                st.write("Predictions:", res.json()["predictions"])
            else:
                st.error("Request failed. Check status code and response above.")
        except Exception as e:
            st.error(f"Error: {e}")

