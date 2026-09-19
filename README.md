# Health Guard ML

Predicts heart disease risk from clinical data using XGBoost. But the real point is SHAP — understanding *why* each prediction was made, not just getting an accuracy number.

Would you trust a model that says "70% heart disease risk" without explaining why? In healthcare, interpretability isn't optional. That's what this project is about.

## How it works

```
data/heart.csv → 80/20 split (random_state=42) → XGBClassifier → classification report
                                                              → SHAP TreeExplainer → feature_importance.png
```

- `main.py` is the whole pipeline: load CSV, split, train, print a `classification_report`, then explain with SHAP and save the summary plot.
- 4 features: `age`, `chol` (cholesterol), `trestbps` (resting blood pressure), `thalach` (max heart rate). Target is binary.
- The bundled `data/heart.csv` is a 100-row sample — enough to run end to end in seconds, not enough to claim anything about real-world performance.

## Run it

```bash
pip install -r requirements.txt
python main.py
# prints precision/recall/F1 per class, saves feature_importance.png
```

## Tech stack

Python, pandas, XGBoost, scikit-learn, SHAP, matplotlib. CI lints with flake8.

## Limitations — read before citing any number from this repo

- **No metrics are recorded.** The script prints a classification report but I never saved a run, so there is no accuracy/F1 figure to quote. Until I paste a real run below, assume there isn't one. *(TODO: run on the full dataset, record the report.)*
- **The bundled data is a 100-row sample of unknown provenance.** Replace it with the full Cleveland/UCI heart disease set (303 rows, 13 features) before drawing conclusions.
- `use_label_encoder=False` in `main.py` is deprecated in current XGBoost — harmless, but should be removed.
- No saved model artifact, no train/validation/test split discipline beyond one split, no calibration. This is a teaching pipeline, not a clinical tool.
- **Not medical advice.** Nothing here is validated for real decisions.

## What I'd improve

- [ ] Swap in the full 13-feature dataset and record the classification report in this README
- [ ] Drop the deprecated `use_label_encoder` flag, pin dependency versions
- [ ] Save the trained model (`joblib`) plus the SHAP expected value, so explanations reproduce without retraining
- [ ] Add calibration curves — raw classifier probabilities shouldn't be read as risk percentages
- [ ] Per-patient force plots, not just the global summary plot

## What I learned

XGBoost is absurdly fast. SHAP values are elegant. And cleaning real-world clinical data takes more code than the ML itself — the four clean columns in the sample hide the messy part.
