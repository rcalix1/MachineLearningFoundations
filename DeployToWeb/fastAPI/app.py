from fastapi import FastAPI, UploadFile
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
async def predict(file: UploadFile):
    df = pd.read_csv(file.file)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}