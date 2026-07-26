# Health Guard ML

Predicts heart disease risk from clinical data using XGBoost. But the real point is SHAP — understanding WHY each prediction was made, not just getting an accuracy number.

```
clinical data → clean → train XGBoost → evaluate → SHAP explain → done
```

```bash
pip install -r requirements.txt
python main.py
# outputs a feature importance plot
```

Would you trust a model that says "70% heart disease risk" without explaining why? Didn't think so. That's why SHAP matters.

XGBoost is absurdly fast. SHAP values are elegant. Real-world data is messy — cleaning takes more code than the ML itself. In healthcare, interpretability isn't optional.
