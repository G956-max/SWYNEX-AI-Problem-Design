# SWYNEX Task 3 - SafeRoute AI: Intelligent Feature

This continues the existing Task 1 and Task 2 repository.

## Added in Task 3

- Model evaluation: accuracy, precision, recall, F1 score, confusion matrix.
- Input validation: required fields, category values, numeric speed, speed range.
- Failure demonstrations: missing field, invalid speed, invalid category.
- Intelligent analysis:
  - human-readable summary of all observed conditions;
  - global feature importance measured by permutation importance on the held-out test set;
  - one-feature-at-a-time what-if checks against a defined safer reference profile.

## Run from the repository root

```bash
pip install pandas scikit-learn
python "intelligent feature/task3/task3.py"
```

The dataset path is resolved from the script location, so the script expects the repository structure to remain intact.

## Interpretation and limitations

The dataset is synthetic. Reported metrics measure agreement with synthetic labels, not real-world accident prediction performance. Global feature importance is not causal evidence. Safer-reference checks are model what-if tests and are not safety guarantees.
