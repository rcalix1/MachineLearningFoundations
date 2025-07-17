import streamlit as st
import pandas as pd
import requests

st.title("Steel Plant Silicon Prediction")

file = st.file_uploader("Upload your CSV file")
if file:
    df = pd.read_csv(file)
    st.dataframe(df)

    if st.button("Predict"):
        res = requests.post(
            "http://localhost:8000/predict",
            files={"file": file},
            headers={"x-api-key": "secret123"}
        )
        st.write("Predictions:", res.json()["predictions"])
