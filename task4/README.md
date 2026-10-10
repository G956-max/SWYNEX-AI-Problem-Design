# SWYNEX Task 4 — SafeRoute AI Final Application

Responsive HTML/CSS/JavaScript web UI with a FastAPI + Scikit-learn Random Forest backend.

## Includes
- Dashboard with model evaluation metrics
- Interactive road-risk analyzer
- Low / Medium / High prediction, confidence and class probabilities
- Quick demo scenarios
- Model/method explanation
- Limitations and ethics notes
- FastAPI Swagger docs at `/docs`

## Run on Windows
Open Command Prompt in the repository root:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r task4\requirements.txt
python -m uvicorn task4.app:app --reload
```

Open `http://127.0.0.1:8000`. API docs: `http://127.0.0.1:8000/docs`.

The dataset is synthetic. The evaluation metrics reflect generated labels, not real-world accident prediction performance. This is an educational prototype, not a real road-safety or navigation system.
