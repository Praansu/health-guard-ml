"""Heart disease risk + SHAP explanations.

Teaching pipeline, not a clinical tool: trains XGBoost on a CSV with a
binary `target` column, prints a classification report, saves the trained
model + metrics, and writes SHAP explanations.

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
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    classification_report,

    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "cleveland.csv"
N_SPLITS = 5


def train_model(data_path, out_dir):
    """Single 80/20 stratified split — original pipeline, kept for comparability."""
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
    probs = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))

    metrics = {
        "data": str(data_path),
        "n_rows": len(df),
        "n_features": X.shape[1],
        "split": "single_80_20",
        "accuracy": report["accuracy"],
        "macro_avg": report["macro avg"],
        "class_0": report["0"],
        "class_1": report["1"],
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    try:
        metrics["shap_expected_value"] = float(explainer.expected_value)
    except (TypeError, ValueError):
        metrics["shap_expected_value"] = [float(v) for v in explainer.expected_value]

    joblib.dump(model, out_dir / "model.joblib")
    with open(out_dir / "metrics_single.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close()

    # SHAP force plots for 3 representative patients (one per outcome mix)
    force_idx = [0, len(X_test) // 2, len(X_test) - 1]
    for i, idx in enumerate(force_idx):
        plt.figure(figsize=(10, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[idx],
            X_test.iloc[idx],
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            out_dir / f"force_plot_patient_{i}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    # Calibration curve — do predicted probabilities match actual outcomes?
    plt.figure(figsize=(6, 6))
    CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=10)
    plt.title("Calibration Curve — XGBoost (single split)")
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve.png", dpi=150)
    plt.close()

    print(
        f"Saved model.joblib, metrics_single.json, SHAP plots, calibration_curve.png to {out_dir}"
    )
    return model, metrics


def cross_validate(data_path, out_dir):
    """5-fold stratified cross-validation — estimates generalization, not luck of one split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_metrics = []
    fold = 0
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        fold_metrics.append(
            {
                "fold": fold,
                "accuracy": float(np.mean(preds == y_test)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall": float(recall_score(y_test, preds, zero_division=0)),
                "f1": float(f1_score(y_test, preds, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probs)),
                "n_test": len(y_test),
            }
        )
        print(
            f"Fold {fold}: acc={fold_metrics[-1]['accuracy']:.3f}, "
            f"f1={fold_metrics[-1]['f1']:.3f}, roc_auc={fold_metrics[-1]['roc_auc']:.3f}"
        )

    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_summary = {
        "data": str(data_path),
        "n_rows": len(df),
        "n_features": X.shape[1],
        "split": f"{N_SPLITS}-fold_stratified_cv",
        "n_splits": N_SPLITS,
    }
    for key in keys:
        vals = [m[key] for m in fold_metrics]
        cv_summary[f"mean_{key}"] = float(np.mean(vals))
        cv_summary[f"std_{key}"] = float(np.std(vals))
    cv_summary["folds"] = fold_metrics

    with open(out_dir / "metrics_cv.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    # Side-by-side comparison table (printed, also in README)
    print("\n=== Single split vs Cross-validation ===")
    print(f"{'Metric':<14} {'Single split':>14} {'CV mean ± std':>20}")
    single = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }
    for key in keys:
        print(
            f"{key:<14} {single[key]:>14.3f} "
            f"{cv_summary[f'mean_{key}']:.3f} ± {cv_summary[f'std_{key}']:.3f}"
        )

    print(f"\nSaved CV metrics to {out_dir / 'metrics_cv.json'}")
    return cv_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train heart-disease XGBoost + SHAP explanations"
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA), help="CSV with a binary target column"
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Directory for model.joblib, metrics.json, plots",
    )
    parser.add_argument(
        "--cv", action="store_true", help="Also run 5-fold stratified cross-validation"
    )
    args = parser.parse_args()

    model, metrics = train_model(args.data, args.out)

    if args.cv:
        cross_validate(args.data, args.out)

    print("Model training complete.")
