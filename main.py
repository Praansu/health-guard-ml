import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("feature_importance.png")
    return model

if __name__ == "__main__":
    trained_model = train_model("data/heart.csv")
    print("Model training complete.")
