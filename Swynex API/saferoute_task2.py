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
REPO_ROOT = BASE.parent
DATASET = REPO_ROOT / "dataset" / "road_risk_data.csv"
MODEL_FILE = BASE / "saferoute_risk_model.joblib"

FEATURES = [
    "weather", "traffic_density", "road_condition", "visibility",
    "vehicle_speed_kmh", "time_of_day", "road_type"
]

def train_model():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)
    X, y = df[FEATURES], df["risk_level"]
    categorical = [
        "weather", "traffic_density", "road_condition", "visibility",
        "time_of_day", "road_type"
    ]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", ["vehicle_speed_kmh"])
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    print("===== SafeRoute AI - Task 2 =====")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, pred, average='weighted', zero_division=0):.4f}")

    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")
    return pipeline

model = train_model()
app = FastAPI(title="SafeRoute AI - SWYNEX Task 2", version="1.1")

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
        # dict() works with Pydantic v1 and v2.
        row = pd.DataFrame([data.dict()], columns=FEATURES)
        risk = model.predict(row)[0]
        confidence = float(max(model.predict_proba(row)[0]))
        return {"risk_level": str(risk), "confidence": round(confidence, 4)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
