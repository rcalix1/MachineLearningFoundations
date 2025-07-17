# AI Model Deployment MVP

This is a minimal full-stack AI product template that:
- Serves your ML model using FastAPI
- Provides a UI using Streamlit
- Works locally or can be deployed to Render.com

## 🔧 Setup

```bash
pip install -r requirements.txt
```

### Train a model (optional)
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "model.pkl")
```

## ▶️ Run FastAPI server
```bash
uvicorn app:app --reload
```

## ▶️ Run Streamlit UI
```bash
streamlit run streamlit_app.py
```

## 🌐 Test it
Upload a CSV with feature columns matching your model.

## 🚀 Deploy (Optional)
Use Render.com:
- Add `app.py`, `streamlit_app.py`, `requirements.txt`, `model.pkl`
- Create a new web service
