# SWYNEX Task 3 - SafeRoute AI

## Intelligent Feature

Task 3 continues the SafeRoute AI prototype from Tasks 1 and 2.

### Intelligent feature
The prototype does more than return a risk label. It tests each input against a safer reference value and reports features whose change can alter the predicted risk class.

### Error handling
The prototype handles:
- missing required fields
- invalid category values
- non-numeric speed
- speed outside 0-200 km/h

### Evaluation
The script reports:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

It also demonstrates:
1. normal input
2. high-risk input
3. missing-field failure
4. invalid-speed failure

### Run

From repository root:

```bash
pip install pandas scikit-learn
python task3/task3.py
```

The dataset is synthetic and the explanation is a prototype-level counterfactual explanation, not proof of causation or a real-world safety decision.
