# SafeRoute AI - SWYNEX Task 2

Single-file continuation of Task 1.

## Includes
- Random Forest classification
- One-hot encoding
- Model evaluation
- FastAPI `/predict` endpoint

## Files
- `saferoute_task2.py`
- `SafeRoute_AI_Road_Risk_Dataset.csv`
- `README.md`

## Run

```bash
pip install pandas scikit-learn joblib fastapi uvicorn
python saferoute_task2.py
```

Open `http://127.0.0.1:8000/docs`

Example input:

```json
{
  "weather": "Heavy Rain",
  "traffic_density": "High",
  "road_condition": "Wet",
  "visibility": "Low",
  "vehicle_speed_kmh": 80,
  "time_of_day": "Night",
  "road_type": "Highway"
}
```

The dataset is synthetic and this is an internship prototype.
