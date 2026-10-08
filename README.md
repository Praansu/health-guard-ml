# Health Guard ML

Predicts heart disease risk from clinical data using XGBoost. But the real point is SHAP — understanding *why* each prediction was made, not just getting an accuracy number.

Would you trust a model that says "70% heart disease risk" without explaining why? In healthcare, interpretability isn't optional. That's what this project is about.

## How it works

```
data/cleveland.csv → 80/20 stratified split (random_state=42) → XGBClassifier → classification report
                                                                              → metrics.json + model.joblib
                                                                              → SHAP TreeExplainer → feature_importance.png
```

- `main.py` is the whole pipeline: load CSV, split, train, print a `classification_report`, save the model + metrics, then explain with SHAP and save the summary plot. Everything lands in `outputs/` (committed, so you can inspect a real run without training).
- Default data is the UCI Cleveland set: 297 rows after cleaning, 13 features, binary target. See `data/README.md` for provenance and the cleaning steps.
- The bundled `data/heart.csv` is a 100-row, 4-feature sample — `python main.py --data data/heart.csv` runs it end to end in seconds. Good for a smoke test, not for conclusions (it scores ~0.60 accuracy on 20 test rows — small data, honest numbers).

## Run it

```bash
pip install -r requirements.txt
python main.py
# prints precision/recall/F1 per class, saves model.joblib, metrics.json,
# feature_importance.png to outputs/
```

## Recorded run (UCI Cleveland, 297 rows, 13 features)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 — no disease | 0.85 | 0.91 | 0.88 | 32 |
| 1 — disease | 0.88 | 0.82 | 0.85 | 28 |
| **Accuracy** | | | **0.87** | 60 |
| Macro avg | 0.87 | 0.86 | 0.87 | 60 |

Run it yourself — same seed, same split, same numbers. Raw report in `outputs/metrics.json`.

## Tech stack

Python, pandas, XGBoost, scikit-learn, SHAP, matplotlib. CI lints with flake8.

## Limitations — read before citing any number from this repo

- Metrics above come from one 80/20 split (n=60 test). Respectable, not gospel — no cross-validation yet, no calibration, and raw classifier probabilities shouldn't be read as risk percentages.
- The bundled `data/heart.csv` sample is still of unknown provenance. Conclusions rest on Cleveland only.
- Single split discipline only (now stratified). No calibration curves. This is a teaching pipeline, not a clinical tool.
- **Not medical advice.** Nothing here is validated for real decisions.

## What I'd improve

- [x] Swap in the full 13-feature dataset and record the classification report in this README
- [x] Drop the deprecated `use_label_encoder` flag, pin dependency versions
- [x] Save the trained model (`joblib`) plus the SHAP expected value, so explanations reproduce without retraining
- [ ] Add calibration curves — raw classifier probabilities shouldn't be read as risk percentages
- [ ] Per-patient force plots, not just the global summary plot
- [ ] Cross-validation instead of a single split

## What I learned

XGBoost is absurdly fast. SHAP values are elegant. And cleaning real-world clinical data takes more code than the ML itself — the four clean columns in the sample hide the messy part.
