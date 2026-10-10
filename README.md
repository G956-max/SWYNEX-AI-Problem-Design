# SWYNEX - SafeRoute AI

SafeRoute AI is a learning prototype that classifies road-condition records into Low, Medium, or High risk categories.

## Repository progression

- **Task 1 — AI Problem Design:** problem statement and synthetic dataset.
- **Task 2 — Model/API Integration:** Random Forest classifier and FastAPI `/predict` endpoint.
- **Task 3 — Intelligent Feature:** evaluation metrics, input validation, failure cases, observed-condition summary, global permutation importance, and safer-reference what-if checks.

## Structure

```text
SWYNEX-AI-Problem-Design/
├── dataset/
│   └── road_risk_data.csv
├── Swynex API/
│   ├── saferoute_task2.py
│   ├── saferoute_risk_model.joblib
│   └── README.md
└── intelligent feature/
    └── task3/
        ├── task3.py
        ├── README.md
        └── evaluation_examples.csv
```

## Task 2: run the API

From the repository root:

```bash
pip install pandas scikit-learn joblib fastapi uvicorn
python "Swynex API/saferoute_task2.py"
```

Open `http://127.0.0.1:8000/docs` and test `POST /predict`.

## Task 3: run evaluation and intelligent analysis

From the repository root:

```bash
pip install pandas scikit-learn
python "intelligent feature/task3/task3.py"
```

## Important limitations

The dataset and its labels are synthetic. Evaluation scores show performance against those generated labels; they must not be represented as validated real-world road-safety performance. The prototype is not suitable for real-world safety decisions.


## Task 4 — Final AI Application

The responsive web app is in `task4/`. See `task4/README.md` for setup instructions.

```bash
python -m pip install -r task4/requirements.txt
python -m uvicorn task4.app:app --reload
```

Open `http://127.0.0.1:8000`.
