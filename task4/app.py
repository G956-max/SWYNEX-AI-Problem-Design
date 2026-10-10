from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATASET = ROOT / "dataset" / "road_risk_data.csv"
FEATURES = ["weather","traffic_density","road_condition","visibility","vehicle_speed_kmh","time_of_day","road_type"]
CATEGORIES = {
 "weather":["Clear","Cloudy","Light Rain","Heavy Rain","Fog"],
 "traffic_density":["Low","Medium","High"],
 "road_condition":["Dry","Wet","Damaged"],
 "visibility":["High","Medium","Low"],
 "time_of_day":["Morning","Afternoon","Evening","Night"],
 "road_type":["City","Highway","Residential","Rural"]
}
df = pd.read_csv(DATASET)
X, y = df[FEATURES], df["risk_level"]
prep = ColumnTransformer([
 ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORIES.keys())),
 ("num", "passthrough", ["vehicle_speed_kmh"])
])
model = Pipeline([("prep", prep), ("rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
model.fit(X_train,y_train)
pred = model.predict(X_test)
metrics = {
 "accuracy":round(float(accuracy_score(y_test,pred)),4),
 "precision":round(float(precision_score(y_test,pred,average="weighted",zero_division=0)),4),
 "recall":round(float(recall_score(y_test,pred,average="weighted",zero_division=0)),4),
 "f1_score":round(float(f1_score(y_test,pred,average="weighted",zero_division=0)),4)
}
app = FastAPI(title="SafeRoute AI", description="SWYNEX Task 4 educational road-risk prototype.", version="1.0")
app.mount("/static", StaticFiles(directory=BASE/"static"), name="static")

class Scenario(BaseModel):
 weather: str
 traffic_density: str
 road_condition: str
 visibility: str
 vehicle_speed_kmh: float = Field(..., ge=0, le=200)
 time_of_day: str
 road_type: str

@app.get("/")
def home():
 return FileResponse(BASE/"static"/"index.html")

@app.get("/api/health")
def health():
 return {"status":"ok","project":"SafeRoute AI","task":4}

@app.get("/api/metrics")
def get_metrics():
 return {**metrics,"dataset_type":"synthetic","note":"Scores reflect generated labels, not real-world accident outcomes."}

@app.post("/api/predict")
def predict(s: Scenario):
 data = s.dict()
 for key, allowed in CATEGORIES.items():
  if data[key] not in allowed:
   raise HTTPException(422, f"Invalid {key}. Allowed: {', '.join(allowed)}")
 row = pd.DataFrame([{k:data[k] for k in FEATURES}], columns=FEATURES)
 risk = str(model.predict(row)[0])
 probs = model.predict_proba(row)[0]
 class_probs = {str(c):round(float(p),4) for c,p in zip(model.classes_,probs)}
 messages = {
  "Low":"The model classifies this scenario as low risk. Continue following local road rules.",
  "Medium":"The model flags conditions that may need extra attention. Adapt driving to conditions.",
  "High":"The model classifies this scenario as high risk. Consider delaying travel or choosing a safer route if practical."
 }
 return {
  "risk_level":risk,"confidence":round(float(max(probs)),4),
  "class_probabilities":class_probs,
  "observed_conditions":[f"Weather: {data['weather']}",f"Traffic: {data['traffic_density']}",f"Road: {data['road_condition']}",f"Visibility: {data['visibility']}",f"Speed: {data['vehicle_speed_kmh']} km/h",f"Time: {data['time_of_day']}",f"Road type: {data['road_type']}"],
  "message":messages.get(risk,"Review conditions carefully."),
  "disclaimer":"Synthetic-data educational prototype; not a real-world safety assessment."
 }
