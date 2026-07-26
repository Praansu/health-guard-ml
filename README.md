# Health Guard ML — Heart Disease Prediction

A machine learning model that predicts heart disease risk from clinical data. Not just accuracy — I wanted to understand WHY the model makes its decisions. That's where SHAP comes in.

## What it does

Give it patient data (age, cholesterol, blood pressure, the usual clinical stuff), and it tells you whether there's a heart disease risk. More importantly, it shows you which factors influenced the decision — so you're not just trusting a black box.

## How it works

```
Clinical data → clean & preprocess → train XGBoost → evaluate → SHAP explanation → done
```

XGBoost does the heavy lifting for predictions, and SHAP (SHapley Additive Explanations) breaks down which features matter most. Turns out, in heart disease prediction, knowing WHY is actually the whole point.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

This trains the model and saves a feature importance plot as `feature_importance.png`.

## What this taught me

- XGBoost is fast. Scarily fast for how accurate it is.
- SHAP values are elegant. Understanding individual predictions is way more useful than just looking at overall accuracy.
- In healthcare ML, interpretability isn't optional. Would you trust a model that says "you have a 70% risk" without explanation?
- Real-world data is messy. Cleaning it takes more code than the actual ML part.

## Tech

Python, XGBoost, scikit-learn, SHAP, Pandas, Matplotlib

## License

MIT
