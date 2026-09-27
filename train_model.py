import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("ai4i2020.csv")

print("========================================")
print("PREDICTIVE MAINTENANCE - ML MODEL")
print("========================================")

print("\nDataset shape:", df.shape)


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]


print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nTarget: Machine failure")


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. PREPROCESSING
# ============================================================

categorical_features = ["Type"]

numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# Preprocessing for Random Forest
rf_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Preprocessing for Logistic Regression
lr_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    ]
)


# ============================================================
# 5. CREATE MODELS
# ============================================================

random_forest = Pipeline(
    steps=[
        ("preprocessor", rf_preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


logistic_regression = Pipeline(
    steps=[
        ("preprocessor", lr_preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# ============================================================
# 6. TRAIN RANDOM FOREST
# ============================================================

print("\n----------------------------------------")
print("Training Random Forest...")
print("----------------------------------------")

random_forest.fit(X_train, y_train)

rf_pred = random_forest.predict(X_test)
rf_prob = random_forest.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
rf_auc = roc_auc_score(y_test, rf_prob)


# ============================================================
# 7. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n----------------------------------------")
print("Training Logistic Regression...")
print("----------------------------------------")

logistic_regression.fit(X_train, y_train)

lr_pred = logistic_regression.predict(X_test)
lr_prob = logistic_regression.predict_proba(X_test)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, zero_division=0)
lr_recall = recall_score(y_test, lr_pred, zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, zero_division=0)
lr_auc = roc_auc_score(y_test, lr_prob)


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Logistic Regression"
    ],
    "Accuracy": [
        rf_accuracy,
        lr_accuracy
    ],
    "Precision": [
        rf_precision,
        lr_precision
    ],
    "Recall": [
        rf_recall,
        lr_recall
    ],
    "F1 Score": [
        rf_f1,
        lr_f1
    ],
    "ROC-AUC": [
        rf_auc,
        lr_auc
    ]
})

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(results.round(4).to_string(index=False))


# ============================================================
# 9. RANDOM FOREST CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("RANDOM FOREST CLASSIFICATION REPORT")
print("========================================")

print(classification_report(
    y_test,
    rf_pred,
    zero_division=0
))


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

rf_cm = confusion_matrix(y_test, rf_pred)

print("Random Forest Confusion Matrix:")
print(rf_cm)


plt.figure(figsize=(6, 5))

ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=["No Failure", "Failure"]
).plot()

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig("random_forest_confusion_matrix.png", dpi=300)

plt.show()


# ============================================================
# 11. ROC CURVE
# ============================================================

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Model Comparison")

plt.legend()

plt.tight_layout()

plt.savefig("roc_curve_comparison.png", dpi=300)

plt.show()


# ============================================================
# 12. FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================

rf_model = random_forest.named_steps["model"]
rf_preprocessor_fitted = random_forest.named_steps["preprocessor"]

feature_names = rf_preprocessor_fitted.get_feature_names_out()

importances = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n========================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("========================================")

print(feature_importance.to_string(index=False))


# Plot top features

top_features = feature_importance.head(10)

plt.figure(figsize=(8, 5))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("Top Features for Machine Failure Prediction")

plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300)

plt.show()


# ============================================================
# 13. SELECT FINAL MODEL
# ============================================================

# Random Forest is selected here as the final model
# because it handles nonlinear relationships well and
# provides feature importance.

final_model = random_forest


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    final_model,
    "predictive_maintenance_model.pkl"
)

print("\n========================================")
print("FINAL MODEL SAVED")
print("========================================")

print("File: predictive_maintenance_model.pkl")

print("\nAdditional files created:")
print("- random_forest_confusion_matrix.png")
print("- roc_curve_comparison.png")
print("- feature_importance.png")

print("\n========================================")
print("MACHINE LEARNING PART COMPLETED")
print("========================================")