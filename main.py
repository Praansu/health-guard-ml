"""Heart disease risk + SHAP explanations.

Teaching pipeline, not a clinical tool: trains XGBoost on a CSV with a
binary `target` column, prints a classification report, saves the trained
model + metrics, and writes a SHAP summary plot.

Default data is the UCI Cleveland set (297 rows after cleaning, 13
features) in data/cleveland.csv. The bundled data/heart.csv is a small
100-row, 4-feature sample that runs in seconds — fine for a smoke test,
not for conclusions.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "cleveland.csv"


def train_model(data_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    print(f"Loaded {len(df)} rows, {X.shape[1]} features from {data_path}")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))

    metrics = {
        "data": str(data_path),
        "n_rows": len(df),
        "n_features": X.shape[1],
        "accuracy": report["accuracy"],
        "macro_avg": report["macro avg"],
        "class_0": report["0"],
        "class_1": report["1"],
    }

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    try:
        metrics["shap_expected_value"] = float(explainer.expected_value)
    except (TypeError, ValueError):
        metrics["shap_expected_value"] = [float(v) for v in explainer.expected_value]

    joblib.dump(model, out_dir / "model.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close()

    print(f"Saved model.joblib, metrics.json, feature_importance.png to {out_dir}")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train heart-disease XGBoost + SHAP explanations")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV with a binary target column")
    parser.add_argument("--out", default="outputs", help="Directory for model.joblib, metrics.json, plots")
    args = parser.parse_args()
    train_model(args.data, args.out)
    print("Model training complete.")
