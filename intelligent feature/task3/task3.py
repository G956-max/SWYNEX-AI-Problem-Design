from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.inspection import permutation_importance

BASE = Path(__file__).resolve().parents[2]
DATASET = BASE / "dataset" / "road_risk_data.csv"

FEATURES = [
    "weather", "traffic_density", "road_condition", "visibility",
    "vehicle_speed_kmh", "time_of_day", "road_type"
]
ALLOWED = {
    "weather": ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Fog"],
    "traffic_density": ["Low", "Medium", "High"],
    "road_condition": ["Dry", "Wet", "Damaged"],
    "visibility": ["High", "Medium", "Low"],
    "time_of_day": ["Morning", "Afternoon", "Evening", "Night"],
    "road_type": ["City", "Highway", "Residential", "Rural"],
}
SAFE_REFERENCE = {
    "weather": "Clear",
    "traffic_density": "Low",
    "road_condition": "Dry",
    "visibility": "High",
    "vehicle_speed_kmh": 40,
    "time_of_day": "Morning",
    "road_type": "City",
}

def build_model():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    df = pd.read_csv(DATASET)
    missing_columns = [c for c in FEATURES + ["risk_level"] if c not in df.columns]
    if missing_columns:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing_columns))

    X, y = df[FEATURES], df["risk_level"]
    categorical = list(ALLOWED.keys())
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
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    # Global feature importance measured by shuffling each original input column.
    importance_result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=5, random_state=42,
        scoring="f1_weighted"
    )
    importance = {
        feature: float(value)
        for feature, value in zip(FEATURES, importance_result.importances_mean)
    }
    return pipeline, metrics, importance

MODEL, METRICS, FEATURE_IMPORTANCE = build_model()

def validate(data):
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary/object.")
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))

    for feature, allowed_values in ALLOWED.items():
        if not isinstance(data[feature], str) or data[feature] not in allowed_values:
            raise ValueError(
                f"Invalid {feature}. Allowed values: {', '.join(allowed_values)}"
            )

    speed = data["vehicle_speed_kmh"]
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ValueError("vehicle_speed_kmh must be a number.")
    if not 0 <= speed <= 200:
        raise ValueError("vehicle_speed_kmh must be between 0 and 200 km/h.")
    return {feature: data[feature] for feature in FEATURES}

def predict(data):
    clean_data = validate(data)
    row = pd.DataFrame([clean_data], columns=FEATURES)
    risk = str(MODEL.predict(row)[0])
    confidence = float(max(MODEL.predict_proba(row)[0]))
    return risk, confidence

def explain(data):
    clean_data = validate(data)
    risk, confidence = predict(clean_data)

    # Honest summary of the observed input; not a causal claim.
    observed_conditions = [
        f"Weather is {clean_data['weather']}.",
        f"Traffic density is {clean_data['traffic_density']}.",
        f"Road condition is {clean_data['road_condition']}.",
        f"Visibility is {clean_data['visibility']}.",
        f"Vehicle speed is {clean_data['vehicle_speed_kmh']} km/h.",
        f"Time of day is {clean_data['time_of_day']}.",
        f"Road type is {clean_data['road_type']}."
    ]

    # Test one-feature-at-a-time changes to a defined safer reference profile.
    counterfactual_checks = []
    for feature in FEATURES:
        if clean_data[feature] == SAFE_REFERENCE[feature]:
            continue
        changed = dict(clean_data)
        changed[feature] = SAFE_REFERENCE[feature]
        alternative_risk, _ = predict(changed)
        counterfactual_checks.append({
            "feature": feature,
            "current_value": clean_data[feature],
            "reference_value": SAFE_REFERENCE[feature],
            "risk_after_change": alternative_risk,
            "changed_predicted_class": alternative_risk != risk
        })

    top_global_features = sorted(
        FEATURE_IMPORTANCE.items(), key=lambda item: item[1], reverse=True
    )[:3]

    return {
        "risk_level": risk,
        "confidence": round(confidence, 4),
        "observed_conditions": observed_conditions,
        "top_global_model_features": [
            {"feature": feature, "permutation_importance_f1": round(score, 4)}
            for feature, score in top_global_features
        ],
        "safer_reference_checks": counterfactual_checks,
        "interpretation_note": (
            "Observed conditions are a summary of the input. Global feature importance "
            "describes model behaviour across the test set, not proof of causation. "
            "Reference checks are model what-if tests, not safety guarantees."
        )
    }

def run_examples():
    print("===== SafeRoute AI - Task 3 =====")
    print(f"Accuracy  : {METRICS['accuracy']:.4f}")
    print(f"Precision : {METRICS['precision']:.4f}")
    print(f"Recall    : {METRICS['recall']:.4f}")
    print(f"F1        : {METRICS['f1']:.4f}")
    print("Confusion matrix:", METRICS["confusion_matrix"])
    print("\nTop global model features (permutation importance):")
    for item in sorted(FEATURE_IMPORTANCE.items(), key=lambda pair: pair[1], reverse=True)[:3]:
        print(f"- {item[0]}: {item[1]:.4f}")

    examples = {
        "normal_case": {
            "weather": "Clear", "traffic_density": "Low", "road_condition": "Dry",
            "visibility": "High", "vehicle_speed_kmh": 40,
            "time_of_day": "Morning", "road_type": "City"
        },
        "high_risk_case": {
            "weather": "Heavy Rain", "traffic_density": "High", "road_condition": "Wet",
            "visibility": "Low", "vehicle_speed_kmh": 80,
            "time_of_day": "Night", "road_type": "Highway"
        },
        "failure_missing_field": {
            "weather": "Clear", "traffic_density": "Low", "road_condition": "Dry",
            "visibility": "High", "time_of_day": "Morning", "road_type": "City"
        },
        "failure_invalid_speed": {
            "weather": "Clear", "traffic_density": "Low", "road_condition": "Dry",
            "visibility": "High", "vehicle_speed_kmh": 250,
            "time_of_day": "Morning", "road_type": "City"
        },
        "failure_invalid_category": {
            "weather": "Stormy", "traffic_density": "Low", "road_condition": "Dry",
            "visibility": "High", "vehicle_speed_kmh": 40,
            "time_of_day": "Morning", "road_type": "City"
        },
    }

    print("\nEvaluation examples:")
    for name, item in examples.items():
        try:
            print(f"\n{name} ->")
            result = explain(item)
            print("Risk:", result["risk_level"], "| Confidence:", result["confidence"])
            print("Observed conditions:")
            for condition in result["observed_conditions"]:
                print(" -", condition)
            print("Top global model features:", result["top_global_model_features"])
            print("Safer-reference checks:")
            print(result["safer_reference_checks"])
        except ValueError as exc:
            print("HANDLED ERROR:", exc)

if __name__ == "__main__":
    run_examples()
