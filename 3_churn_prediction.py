"""
Step 3: Churn Definition, Labeling, and Supervised Churn Prediction
Author: Student
Course: Retail Analytics Project

This script builds models to predict whether a customer will churn.
Key steps:
1. Define Churn: Customer has not purchased in the last 180 days (churned = 1, active = 0).
2. Split dataset into train and test sets (stratified).
3. Encode categorical features (preferred payment method, customer segment) using OneHotEncoder.
4. Target Leakage Avoidance: Exclude the raw 'recency' feature from the model, as it directly defines churn.
5. Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) on training data.
6. Train and compare models: Logistic Regression, Random Forest, and XGBoost.
7. Track parameters, metrics, and models using MLflow.
8. Evaluate using ROC-AUC, F1, and Precision-Recall curves.
9. Generate SHAP values for explainability on the XGBoost model.
10. Save the final model and encoders.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML / preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)

# MLflow and SHAP
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap

# ----------------------------------------------------
# 1. Load Segmented Customers Data
# ----------------------------------------------------
print("--- Step 1: Loading customer data ---")
data_path = "data/processed/segmented_customers.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing {data_path}. Please run 2_segmentation.py first.")

df = pd.read_csv(data_path)
print(f"Loaded segmented customer data with shape: {df.shape}")

# ----------------------------------------------------
# 2. Define Churn Label
# ----------------------------------------------------
print("\n--- Step 2: Defining and labeling churn ---")
# Churn: No purchase in the last 180 days (recency > 180)
df['churned'] = (df['recency'] > 180).astype(int)

churn_counts = df['churned'].value_counts()
churn_rate = df['churned'].mean() * 100
print(f"Active (0): {churn_counts.get(0, 0)} | Churned (1): {churn_counts.get(1, 0)}")
print(f"Churn rate in dataset: {churn_rate:.2f}%")

# ----------------------------------------------------
# 3. Train-Test Split (with Leakage Prevention)
# ----------------------------------------------------
print("\n--- Step 3: Splitting features and target (Leakage Prevention) ---")
# Features to exclude:
# - customer_unique_id (just an identifier)
# - recency (TARGET LEAKAGE: Churn is defined by recency. If we include it, the model gets 100% accuracy instantly!)
# - cluster (redundant, we use the 'segment' text column instead)
features_to_drop = ['customer_unique_id', 'recency', 'cluster', 'churned']
if 'pca_1' in df.columns:
    features_to_drop.extend(['pca_1', 'pca_2'])

X = df.drop(columns=features_to_drop)
y = df['churned']

print(f"Feature columns: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")

# ----------------------------------------------------
# 4. Encoding Categorical Features
# ----------------------------------------------------
print("\n--- Step 4: Encoding categorical variables ---")
categorical_cols = ['preferred_payment_method', 'segment']
numeric_cols = ['frequency', 'monetary', 'avg_order_value', 'avg_review_score', 'cancellation_rate']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])

# Convert encoded features to DataFrame
encoded_col_names = encoder.get_feature_names_out(categorical_cols)
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_col_names, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_col_names, index=X_test.index)

# Combine numeric features and encoded features
X_train_final = pd.concat([X_train[numeric_cols], X_train_encoded_df], axis=1)
X_test_final = pd.concat([X_test[numeric_cols], X_test_encoded_df], axis=1)

print(f"Encoded features. New feature space size: {X_train_final.shape[1]}")

# ----------------------------------------------------
# 5. Class Imbalance Handling (SMOTE)
# ----------------------------------------------------
print("\n--- Step 5: Handling class imbalance with SMOTE on training set ---")
print(f"Training class distribution before SMOTE: {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_final, y_train)
print(f"Training class distribution after SMOTE: {y_train_res.value_counts().to_dict()}")

# ----------------------------------------------------
# 6. Model Training & MLflow Tracking
# ----------------------------------------------------
print("\n--- Step 6: Training models with MLflow tracking ---")
# Setup local MLflow experiment
mlflow.set_experiment("Olist_Churn_Prediction")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
}

trained_models = {}
metrics_results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    with mlflow.start_run(run_name=model_name):
        # Fit model
        model.fit(X_train_res, y_train_res)
        trained_models[model_name] = model
        
        # Predictions
        y_pred = model.predict(X_test_final)
        y_prob = model.predict_proba(X_test_final)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        metrics_results[model_name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        }
        
        print(f"  {model_name} metrics:")
        for k, v in metrics_results[model_name].items():
            print(f"    {k}: {v:.4f}")
            mlflow.log_metric(k.lower().replace("-", "_"), v)
            
        # Log parameters
        mlflow.log_params(model.get_params())
        
        # Log model
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, "xgboost_model")
        else:
            mlflow.sklearn.log_model(model, f"{model_name.lower()}_model")

# ----------------------------------------------------
# 7. Generate Evaluation Plots (ROC & Precision-Recall)
# ----------------------------------------------------
print("\n--- Step 7: Generating evaluation curves ---")
plt.figure(figsize=(14, 6))

# ROC Curve
plt.subplot(1, 2, 1)
for model_name, model in trained_models.items():
    y_prob = model.predict_proba(X_test_final)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics_results[model_name]['ROC-AUC']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Precision-Recall Curve (Very important for imbalanced data)
plt.subplot(1, 2, 2)
for model_name, model in trained_models.items():
    y_prob = model.predict_proba(X_test_final)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=f"{model_name}", linewidth=2)
plt.title('Precision-Recall Curves')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
eval_plot_path = "reports/model_comparison.png"
plt.savefig(eval_plot_path, dpi=150)
plt.close()
print(f"Saved model evaluation curves to: {eval_plot_path}")

# ----------------------------------------------------
# 8. SHAP Explainability (XGBoost)
# ----------------------------------------------------
print("\n--- Step 8: Computing SHAP values for XGBoost ---")
xgb_model = trained_models["XGBoost"]

# Calculate SHAP values on a subset of the test set (1000 samples) to run fast
shap_sample_size = 1000
X_test_sample = X_test_final.sample(n=shap_sample_size, random_state=42)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test_sample)

# Save SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title("XGBoost SHAP Feature Importance Summary", fontsize=14, pad=15)
plt.tight_layout()

shap_plot_path = "reports/shap_importance.png"
plt.savefig(shap_plot_path, dpi=150)
plt.close()
print(f"Saved SHAP importance summary to: {shap_plot_path}")

# ----------------------------------------------------
# 9. Save Best Model and Encoder
# ----------------------------------------------------
print("\n--- Step 9: Saving artifacts ---")
os.makedirs("models", exist_ok=True)

# Save the primary XGBoost model
with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# Save the fitted one-hot encoder
with open("models/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Save feature list to ensure correct feature order in API
feature_list = list(X_train_final.columns)
with open("models/feature_list.pkl", "wb") as f:
    pickle.dump(feature_list, f)

print("Saved XGBoost model to models/xgboost_model.pkl")
print("Saved Encoder to models/encoder.pkl")
print("Saved feature columns list to models/feature_list.pkl")
print("\nChurn prediction modeling finished successfully!")
