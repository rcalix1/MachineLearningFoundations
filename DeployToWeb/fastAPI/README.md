# AI Model Deployment MVP (with API Key)

This is a minimal full-stack AI product template that:
- Serves your ML model using FastAPI
- Provides a UI using Streamlit
- Adds simple API key protection to secure public endpoints



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
Upload a CSV with feature columns matching your model. Auth header required:
```bash
curl -X POST http://localhost:8000/predict \
  -H "x-api-key: secret123" \
  -F "file=@yourfile.csv"
```

## 🚀 Deploy (Optional)
Use Render.com:
- Add `app.py`, `streamlit_app.py`, `requirements.txt`, `model.pkl`
- Create a new web service

---


"""
# AI Model Deployment MVP (with API Key)

This is a minimal full-stack AI product template that:
- Serves your ML model using FastAPI
- Provides a UI using Streamlit
- Adds simple API key protection to secure public endpoints

## ✅ Deploying Your FastAPI Server on a GPU Box

### 🧱 1. Create a Conda Environment on the GPU box
```bash
conda create -n ai-deploy python=3.10 -y
conda activate ai-deploy
```

### 📦 2. Install Requirements
Place the `requirements.txt` file in the directory, then:
```bash
pip install -r requirements.txt
```
If you're using a GPU model (e.g., PyTorch or TensorFlow), install those too:
```bash
pip install torch torchvision torchaudio  # or other ML libraries
```

### 🚀 3. Run the FastAPI Server
Make sure you're in the directory with `app.py` and `model.pkl`:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Now the server is listening on all interfaces (your laptop can reach it).

## 🌍 On Your Laptop: Send Data

### 🔍 1. Find the GPU server’s IP
Let’s say it’s `192.168.1.100` (or a public IP if cloud-hosted).

### 📤 2. Send CSV via curl
```bash
curl -X POST http://192.168.1.100:8000/predict \
  -H "x-api-key: secret123" \
  -F "file=@yourfile.csv"
```

### 🐍 Or with Python:
```python
import requests
files = {"file": open("yourfile.csv", "rb")}
headers = {"x-api-key": "secret123"}
res = requests.post("http://192.168.1.100:8000/predict", files=files, headers=headers)
print(res.json())
```

## 🔐 Optional: Security Tips
If you're using this outside your local network (e.g., public IP), consider:
- Running behind **nginx** with **HTTPS**
- Using **gunicorn + uvicorn workers** for performance
- Adding **basic auth** or **JWT tokens**

"""

# Procfile (for Render.com or Railway.app)
web: uvicorn app:app --host 0.0.0.0 --port 8000

