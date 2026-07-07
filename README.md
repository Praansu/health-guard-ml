# Health Guard ML

A heart disease prediction model using XGBoost. Also tried to understand why the model makes the predictions it does using SHAP.

## What it does

Takes clinical data about a patient (age, cholesterol, blood pressure, etc.) and predicts whether they have heart disease. Trained on a real dataset.

## How it works

1. Loads the heart disease dataset using Pandas
2. Splits it into training and testing sets
3. Trains an XGBoost classifier on the training data
4. Evaluates the model using precision, recall, and F1-score
5. Uses SHAP to show which features matter most for predictions

## Run it

```
pip install -r requirements.txt
python main.py
```

This will train the model and save a feature importance plot as feature_importance.png.

## What I learned

- How gradient boosting works (XGBoost specifically)
- That model accuracy isnt everything - understanding WHY matters too
- SHAP values and how they explain individual predictions
- Basic ML workflow: load, clean, train, evaluate, interpret

## Stack

XGBoost, scikit-learn, SHAP, Pandas, Matplotlib
