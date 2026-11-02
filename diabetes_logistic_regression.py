"""
LOGISTIC REGRESSION PROJECT
Project: Diabetes Prediction using Kaggle Data (Pima Indians Diabetes Database)

Workflow:
  load CSV → inspect data → replace invalid zeros → train-test split
  → median imputation → StandardScaler → balanced Logistic Regression
  → evaluate → interpret coefficients
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay, ConfusionMatrixDisplay
)

# -------------------------------------------------
# 1. Load the dataset
# -------------------------------------------------
# Public mirror of the classic Pima Indians Diabetes dataset
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(URL, header=None, names=COLUMN_NAMES)

print("=" * 60)
print("1. DATASET LOADED")
print("=" * 60)
print(f"Shape: {df.shape}")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe())

# -------------------------------------------------
# 2. Inspect & replace invalid zeros
# -------------------------------------------------
# Physiologically impossible zeros (Glucose, BloodPressure, SkinThickness,
# Insulin, BMI) are treated as missing values.
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\n" + "=" * 60)
print("2. REPLACING INVALID ZEROS WITH NaN")
print("=" * 60)
print("Zero counts before replacement:")
print((df[cols_with_zeros] == 0).sum())

df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

print("\nMissing values after replacement:")
print(df.isnull().sum())

# -------------------------------------------------
# 3. Train-test split (before imputation to avoid leakage)
# -------------------------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,          # fixed split for reproducibility
    stratify=y
)

print("\n" + "=" * 60)
print("3. TRAIN-TEST SPLIT")
print("=" * 60)
print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
print(f"Class distribution (train):\n{y_train.value_counts(normalize=True).round(3)}")
print(f"Class distribution (test):\n{y_test.value_counts(normalize=True).round(3)}")

# -------------------------------------------------
# 4. Pipeline: Median Imputation → StandardScaler → Balanced Logistic Regression
# -------------------------------------------------
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",   # handles class imbalance
        max_iter=1000,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# -------------------------------------------------
# 5. Predictions & Evaluation
# -------------------------------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred)
rec   = recall_score(y_test, y_pred)
f1    = f1_score(y_test, y_pred)
auc   = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 60)
print("5. EVALUATION METRICS")
print("=" * 60)
print(f"{'Metric':<12} {'Result':>8}   Meaning")
print("-" * 50)
print(f"{'Accuracy':<12} {acc:>8.3f}   Overall correct predictions")
print(f"{'Precision':<12} {prec:>8.3f}   Reliability of positive predictions")
print(f"{'Recall':<12} {rec:>8.3f}   Actual positive cases detected")
print(f"{'F1-score':<12} {f1:>8.3f}   Balance of precision and recall")
print(f"{'ROC-AUC':<12} {auc:>8.3f}   Overall class-separation ability")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Diabetes (0)", "Diabetes (1)"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"                 Predicted 0    Predicted 1")
print(f"Actual 0           TN = {tn:<4}     FP = {fp}")
print(f"Actual 1           FN = {fn:<4}     TP = {tp}")

# -------------------------------------------------
# 6. Coefficient Interpretation
# -------------------------------------------------
feature_names = X.columns.tolist()
coefs = pipeline.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefs
}).sort_values("Coefficient", ascending=False)

print("\n" + "=" * 60)
print("6. MODEL COEFFICIENTS (after scaling)")
print("=" * 60)
print(coef_df.to_string(index=False))
print("\nInterpretation:")
print("  • Positive coefficient → increases probability of diabetes")
print("  • Negative coefficient → decreases probability of diabetes")
print("  • Glucose and BMI typically show the strongest positive coefficients")

# -------------------------------------------------
# 7. Visualizations
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix heatmap
ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"]).plot(
    ax=axes[0], cmap="Blues", colorbar=False
)
axes[0].set_title("Confusion Matrix")

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
axes[1].set_title(f"ROC Curve (AUC = {auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)

plt.tight_layout()
plt.savefig("evaluation_plots.png", dpi=150, bbox_inches="tight")
print("\nPlots saved as: evaluation_plots.png")

# Coefficient bar plot
plt.figure(figsize=(8, 5))
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["Coefficient"]]
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coefficient value (standardized features)")
plt.title("Logistic Regression Coefficients")
plt.tight_layout()
plt.savefig("coefficients.png", dpi=150, bbox_inches="tight")
print("Coefficient plot saved as: coefficients.png")

# -------------------------------------------------
# 8. Key Findings (matching the report)
# -------------------------------------------------
print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)
print(f"""
The model obtains a ROC-AUC of about {auc:.3f} and detects approximately
{rec*100:.1f}% of the positive cases in the fixed test split.
Glucose and BMI have the strongest positive model coefficients.
Recall is especially important in a screening-style classification task
because false negatives represent missed positive cases.

Limitation: This small historical dataset represents a specific population.
The model is suitable for learning classification concepts, not for medical diagnosis.
""")

print("Script finished successfully.")
