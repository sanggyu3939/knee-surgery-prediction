# Prediction of Surgical Intervention in Acute Knee Trauma

This repository contains the code used in the study:

**"Prediction of Surgical Intervention in Patients with Acute Knee Trauma: A Threshold-Based Clinical Decision Approach"**

## 🔍 Overview
This study develops and evaluates prediction models to identify patients who are likely to require surgical intervention using routinely available clinical variables.

Key features:
- Logistic regression, Random Forest, XGBoost
- Threshold-based performance evaluation
- Decision Curve Analysis (DCA)
- SHAP interpretability

---

## 📊 Models
- Logistic Regression
- Random Forest
- XGBoost

---

## 📈 Evaluation
- AUROC
- AUPRC
- Brier score
- Calibration plot
- Threshold-based metrics
- Decision Curve Analysis

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/model.py
