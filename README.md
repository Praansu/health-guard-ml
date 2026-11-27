# Health Guard ML

[![CI](https://img.shields.io/github/actions/workflow/status/Praansu/health-guard-ml/ci.yml?style=flat-square)](https://github.com/Praansu/health-guard-ml/actions)
[![License](https://img.shields.io/badge/License-MIT-1F2328?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-1F2328?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-1F2328?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-explainable-1F2328?style=flat-square)](https://shap.readthedocs.io)

Predicts heart disease risk from clinical data using XGBoost. But the real point is **SHAP** — understanding *why* each prediction was made, not just getting an accuracy number.

Would you trust a model that says "70% heart disease risk" without explaining why? In healthcare, interpretability isn't optional. That's what this project is about.

## How it works

```
data/cleveland.csv → 80/20 stratified split (random_state=42) → XGBClassifier → classification report
                                                                               → metrics_single.json + model.joblib
                                                                               → SHAP TreeExplainer → feature_importance.png
                                                                               → SHAP force plots (per patient)
                                                                               → Calibration curve (probability reliability)
```

Cross-validation:
```
data/cleveland.csv → 5-fold stratified CV → per-fold metrics → mean ± std → metrics_cv.json
```

- `main.py` is the whole pipeline: load CSV, split, train, print a `classification_report`, save the model + metrics, then explain with SHAP and save the summary plot. Everything lands in `outputs/` (committed, so you can inspect a real run without training).
- Run `python main.py` for the single-split pipeline. Add `--cv` for 5-fold stratified cross-validation: `python main.py --cv`.
- Default data is the UCI Cleveland set: 297 rows after cleaning, 13 features, binary target. See `data/README.md` for provenance and the cleaning steps.
- The bundled `data/heart.csv` is a 100-row, 4-feature sample — `python main.py --data data/heart.csv` runs it end to end in seconds. Good for a smoke test, not for conclusions (it scores ~0.60 accuracy on 20 test rows — small data, honest numbers).

## Recorded runs (UCI Cleveland, 297 rows, 13 features)

### Single 80/20 stratified split (n=60 test)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 — no disease | 0.85 | 0.91 | 0.88 | 32 |
| 1 — disease | 0.88 | 0.82 | 0.85 | 28 |
| **Accuracy** | | | **0.87** | 60 |
| Macro avg | 0.87 | 0.86 | 0.87 | 60 |
| ROC-AUC | | | **0.89** | — |

Raw report in `outputs/metrics_single.json`.

### 5-fold stratified cross-validation

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.79 ± 0.05 |
| Precision | 0.79 ± 0.07 |
| Recall | 0.76 ± 0.09 |
| F1 | 0.77 ± 0.06 |
| ROC-AUC | 0.86 ± 0.05 |

Raw per-fold results in `outputs/metrics_cv.json`.

### Single split vs Cross-validation

The single 80/20 split reported 0.87 accuracy — cross-validation gives a more honest 0.79 ± 0.05. One lucky split isn't generalization. That gap is exactly why CV matters.

| Metric | Single split | CV mean ± std |
|--------|-------------|---------------|
| Accuracy | 0.87 | 0.79 ± 0.05 |
| Precision | 0.89 | 0.79 ± 0.07 |
| Recall | 0.82 | 0.76 ± 0.09 |
| F1 | 0.85 | 0.77 ± 0.06 |
| ROC-AUC | 0.89 | 0.86 ± 0.05 |

## Outputs

| File | What it is |
|------|-----------|
| `outputs/model.joblib` | Trained XGBoost model |
| `outputs/metrics_single.json` | Single-split metrics |
| `outputs/metrics_cv.json` | 5-fold CV metrics (with `--cv`) |
| `outputs/feature_importance.png` | Global SHAP summary plot |
| `outputs/force_plot_patient_0..2.png` | Per-patient SHAP force plots |
| `outputs/calibration_curve.png` | Reliability diagram — do predicted probabilities match actual outcomes? |

## Tech stack

Python, pandas, XGBoost, scikit-learn, SHAP, matplotlib. CI lints with **ruff**.

## Limitations — read before citing any number from this repo

- Metrics come from 297 rows of the UCI Cleveland set. Respectable, not gospel — small dataset, no external validation.
- Raw classifier probabilities shouldn't be read as risk percentages without calibration — see `outputs/calibration_curve.png`.
- This is a teaching pipeline, not a clinical tool.
- **Not medical advice.** Nothing here is validated for real decisions.

## What I'd improve

- [x] Swap in the full 13-feature dataset and record the classification report in this README
- [x] Drop the deprecated `use_label_encoder` flag, pin dependency versions
- [x] Save the trained model (`joblib`) plus the SHAP expected value, so explanations reproduce without retraining
- [x] Add calibration curves — raw classifier probabilities shouldn't be read as risk percentages
- [x] Per-patient force plots, not just the global summary plot
- [x] Cross-validation instead of a single split

## What I learned

XGBoost is absurdly fast. SHAP values are elegant. And cleaning real-world clinical data takes more code than the ML itself — the four clean columns in the sample hide the messy part.

Biggest lesson: a single train/test split can flatter you. The same model scored 0.87 accuracy on one split but 0.79 ± 0.05 across five folds. Cross-validation isn't extra credit — it's how you know your number isn't luck.