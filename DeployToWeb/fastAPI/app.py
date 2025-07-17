from fastapi import FastAPI, UploadFile, Header, HTTPException
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("model.pkl")
API_KEY = "secret123"  # Change this to your desired key

@app.post("/predict")
async def predict(file: UploadFile, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    df = pd.read_csv(file.file)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
