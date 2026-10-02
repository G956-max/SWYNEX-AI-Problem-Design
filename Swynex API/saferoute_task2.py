import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
DATASET = BASE / "SafeRoute_AI_Road_Risk_Dataset.csv"
MODEL_FILE = BASE / "saferoute_risk_model.joblib"

def train_model():
    df = pd.read_csv(DATASET)
    features = ["weather","traffic_density","road_condition","visibility",
                "vehicle_speed_kmh","time_of_day","road_type"]
    X, y = df[features], df["risk_level"]
    categorical = ["weather","traffic_density","road_condition","visibility",
                   "time_of_day","road_type"]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", ["vehicle_speed_kmh"])
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("===== SafeRoute AI - Task 2 =====")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, pred, average='weighted', zero_division=0):.4f}")
    joblib.dump(model, MODEL_FILE)
    print("Model saved successfully.")
    return model

model = train_model()

app = FastAPI(title="SafeRoute AI - SWYNEX Task 2", version="1.0")

class RoadInput(BaseModel):
    weather: str
    traffic_density: str
    road_condition: str
    visibility: str
    vehicle_speed_kmh: float = Field(..., ge=0, le=200)
    time_of_day: str
    road_type: str

@app.get("/")
def home():
    return {"project": "SafeRoute AI", "task": "SWYNEX Task 2", "status": "running"}

@app.post("/predict")
def predict(data: RoadInput):
    try:
        row = pd.DataFrame([data.model_dump()])
        risk = model.predict(row)[0]
        confidence = float(max(model.predict_proba(row)[0]))
        return {"risk_level": risk, "confidence": round(confidence, 4)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
