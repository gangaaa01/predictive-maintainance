import pandas as pd
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

from xgboost import XGBClassifier


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


tree_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


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
        ("preprocessor", tree_preprocessor),
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


xgboost_model = Pipeline(
    steps=[
        ("preprocessor", tree_preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
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
# 8. TRAIN XGBOOST
# ============================================================

print("\n----------------------------------------")
print("Training XGBoost...")
print("----------------------------------------")

xgboost_model.fit(X_train, y_train)

xgb_pred = xgboost_model.predict(X_test)
xgb_prob = xgboost_model.predict_proba(X_test)[:, 1]

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred, zero_division=0)
xgb_recall = recall_score(y_test, xgb_pred, zero_division=0)
xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
xgb_auc = roc_auc_score(y_test, xgb_prob)


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Logistic Regression",
        "XGBoost"
    ],
    "Accuracy": [
        rf_accuracy,
        lr_accuracy,
        xgb_accuracy
    ],
    "Precision": [
        rf_precision,
        lr_precision,
        xgb_precision
    ],
    "Recall": [
        rf_recall,
        lr_recall,
        xgb_recall
    ],
    "F1 Score": [
        rf_f1,
        lr_f1,
        xgb_f1
    ],
    "ROC-AUC": [
        rf_auc,
        lr_auc,
        xgb_auc
    ]
})

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(results.round(4).to_string(index=False))


# ============================================================
# 10. RANDOM FOREST CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("RANDOM FOREST CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        rf_pred,
        zero_division=0
    )
)


# ============================================================
# 11. XGBOOST CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("XGBOOST CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        xgb_pred,
        zero_division=0
    )
)


# ============================================================
# 12. RANDOM FOREST CONFUSION MATRIX
# ============================================================

rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=["No Failure", "Failure"]
).plot(ax=ax)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "random_forest_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. XGBOOST CONFUSION MATRIX
# ============================================================

xgb_cm = confusion_matrix(y_test, xgb_pred)

print("\nXGBoost Confusion Matrix:")
print(xgb_cm)

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay(
    confusion_matrix=xgb_cm,
    display_labels=["No Failure", "Failure"]
).plot(ax=ax)

plt.title("XGBoost - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "xgboost_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. ROC CURVE
# ============================================================

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)

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
    xgb_fpr,
    xgb_tpr,
    label=f"XGBoost (AUC = {xgb_auc:.3f})"
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

plt.savefig(
    "roc_curve_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. FEATURE IMPORTANCE - XGBOOST
# ============================================================

xgb_model = xgboost_model.named_steps["model"]

xgb_preprocessor_fitted = (
    xgboost_model.named_steps["preprocessor"]
)

feature_names = (
    xgb_preprocessor_fitted.get_feature_names_out()
)

importances = xgb_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========================================")
print("XGBOOST FEATURE IMPORTANCE")
print("========================================")

print(
    feature_importance.to_string(index=False)
)


top_features = feature_importance.head(10)

plt.figure(figsize=(8, 5))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Top Features for Machine Failure Prediction - XGBoost"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. SELECT FINAL MODEL
# ============================================================

final_model = xgboost_model


# ============================================================
# 17. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    final_model,
    "predictive_maintenance_model.pkl"
)

print("\n========================================")
print("FINAL MODEL SAVED")
print("========================================")

print("Final model: XGBoost")

print(
    "File: predictive_maintenance_model.pkl"
)

print("\nAdditional files created:")
print("- random_forest_confusion_matrix.png")
print("- xgboost_confusion_matrix.png")
print("- roc_curve_comparison.png")
print("- feature_importance.png")

print("\n========================================")
print("MACHINE LEARNING PART COMPLETED")
print("========================================")