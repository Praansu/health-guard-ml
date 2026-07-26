# ❤️ Health Guard ML — Heart Disease Prediction

A machine learning model that predicts heart disease risk using clinical data, with **SHAP explainability** to understand why each prediction is made.

## 🔍 Overview

This project takes patient clinical data (age, cholesterol, blood pressure, etc.) and predicts heart disease likelihood using **XGBoost**, then explains each prediction using **SHAP** (SHapley Additive exPlanations). It's not just about accuracy — it's about **interpretability**.

## ✨ Features

- **XGBoost Classifier** — gradient boosting for high-accuracy predictions
- **SHAP Explainability** — understand which features drive each prediction
- **Feature Importance Visualization** — see what matters most
- **Full ML Pipeline** — load → clean → train → evaluate → interpret

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py
```

This trains the model and saves a feature importance plot as `feature_importance.png`.

## 📊 How It Works

```
Clinical Data → Pandas Preprocessing → Train/Test Split
                                        ↓
                              XGBoost Training
                                        ↓
                              Evaluation (Precision, Recall, F1)
                                        ↓
                              SHAP Explanation → Feature Importance Plot
```

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-5C3EE8?style=flat-square&logo=&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=&logoColor=white)

## 📚 What I Learned

- How **gradient boosting** works under the hood (XGBoost internals)
- That accuracy isn't everything — **understanding WHY** a model predicts what it does is critical, especially in healthcare
- **SHAP values** and how they explain individual predictions
- End-to-end ML workflow: load → clean → train → evaluate → interpret

## 📄 License

MIT © 2026 Praansu Karmacharya
