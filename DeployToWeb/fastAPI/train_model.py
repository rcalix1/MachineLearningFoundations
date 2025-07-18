from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y  = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "model.pkl")
