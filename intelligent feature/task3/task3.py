import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE = Path(__file__).resolve().parent.parent.parent
DATASET = BASE / "dataset" / "road_risk_data.csv.csv"

FEATURES = ["weather","traffic_density","road_condition","visibility",
            "vehicle_speed_kmh","time_of_day","road_type"]

ALLOWED = {
    "weather": ["Clear","Cloudy","Light Rain","Heavy Rain","Fog"],
    "traffic_density": ["Low","Medium","High"],
    "road_condition": ["Dry","Wet","Damaged"],
    "visibility": ["High","Medium","Low"],
    "time_of_day": ["Morning","Afternoon","Evening","Night"],
    "road_type": ["City","Highway","Residential","Rural"],
}

SAFE = {
    "weather":"Clear","traffic_density":"Low","road_condition":"Dry",
    "visibility":"High","vehicle_speed_kmh":40,"time_of_day":"Morning",
    "road_type":"City"
}

def build_model():
    df = pd.read_csv(DATASET)
    X, y = df[FEATURES], df["risk_level"]
    cats = list(ALLOWED.keys())
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", "passthrough", ["vehicle_speed_kmh"])
    ])
    model = Pipeline([
        ("prep", prep),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
    ])
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    model.fit(Xtr,ytr)
    pred = model.predict(Xte)
    metrics = {
        "accuracy": accuracy_score(yte,pred),
        "precision": precision_score(yte,pred,average="weighted",zero_division=0),
        "recall": recall_score(yte,pred,average="weighted",zero_division=0),
        "f1": f1_score(yte,pred,average="weighted",zero_division=0),
        "confusion_matrix": confusion_matrix(yte,pred).tolist()
    }
    return model, metrics

MODEL, METRICS = build_model()

def validate(data):
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    for f, vals in ALLOWED.items():
        if data[f] not in vals:
            raise ValueError(f"Invalid {f}. Allowed: {', '.join(vals)}")
    speed = data["vehicle_speed_kmh"]
    if isinstance(speed,bool) or not isinstance(speed,(int,float)):
        raise ValueError("vehicle_speed_kmh must be a number.")
    if not 0 <= speed <= 200:
        raise ValueError("vehicle_speed_kmh must be between 0 and 200 km/h.")

def predict(data):
    validate(data)
    x = pd.DataFrame([data], columns=FEATURES)
    risk = MODEL.predict(x)[0]
    conf = float(max(MODEL.predict_proba(x)[0]))
    return risk, conf

def explain(data):
    risk, conf = predict(data)
    factors = []
    for f in FEATURES:
        if data[f] == SAFE[f]:
            continue
        changed = dict(data)
        changed[f] = SAFE[f]
        try:
            safer_risk, _ = predict(changed)
            if safer_risk != risk:
                factors.append(f"{f}: {data[f]} -> safer reference {SAFE[f]} changes prediction to {safer_risk}")
        except ValueError:
            pass
    if not factors:
        factors.append("No single tested feature changed the predicted class; combined conditions drive the result.")
    return {"risk_level": risk, "confidence": round(conf,4), "explanation": factors}

if __name__ == "__main__":
    print("===== SafeRoute AI - Task 3 =====")
    for k in ["accuracy","precision","recall","f1"]:
        print(f"{k.title():10}: {METRICS[k]:.4f}")
    print("Confusion matrix:", METRICS["confusion_matrix"])

    examples = {
        "normal_case": {"weather":"Clear","traffic_density":"Low","road_condition":"Dry","visibility":"High","vehicle_speed_kmh":40,"time_of_day":"Morning","road_type":"City"},
        "high_risk_case": {"weather":"Heavy Rain","traffic_density":"High","road_condition":"Wet","visibility":"Low","vehicle_speed_kmh":80,"time_of_day":"Night","road_type":"Highway"},
        "failure_missing_field": {"weather":"Clear","traffic_density":"Low","road_condition":"Dry","visibility":"High","time_of_day":"Morning","road_type":"City"},
        "failure_invalid_speed": {"weather":"Clear","traffic_density":"Low","road_condition":"Dry","visibility":"High","vehicle_speed_kmh":250,"time_of_day":"Morning","road_type":"City"}
    }
    print("\nEvaluation examples:")
    for name, item in examples.items():
        try:
            print(name, "->", explain(item))
        except ValueError as e:
            print(name, "-> HANDLED ERROR:", e)
