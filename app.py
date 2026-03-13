import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from io import StringIO
# For data upload
# sklearn and related
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import shap

# Streamlit layout
st.title("Heart Disease Prediction - Colab to Streamlit")

# Upload data
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload heart.csv", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Please upload heart.csv to continue.")
    data = None

if data is not None:
    st.subheader("Data Preview")
    st.write(data.head())

    st.subheader("Missing Values per Column")
    st.write(data.isnull().sum())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

    # Age histogram
    st.subheader("Age Distribution")
    plt.figure(figsize=(6, 4))
    sns.histplot(data["age"], kde=True)
    st.pyplot(plt)

    # Features and target
    if "target" not in data.columns:
        st.error("Column 'target' not found in data.")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_scaled, y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_smote, y_smote, test_size=0.2, random_state=42, stratify=y_smote
        )

        # Models
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)

        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        xgb.fit(X_train, y_train)

        lr = LogisticRegression(max_iter=1000, n_jobs=-1)
        lr.fit(X_train, y_train)

        # Predictions and evaluation (using the RF as example)
        y_pred_rf = rf.predict(X_test)
        y_proba_rf = rf.predict_proba(X_test)[:, 1]
        roc_rf = roc_auc_score(y_test, y_proba_rf)

        st.subheader("Model Evaluation (RandomForest)")
        st.write(f"ROC AUC: {roc_rf:.4f}")

        st.subheader("Classification Report (RF)")
        report = classification_report(y_test, y_pred_rf, output_dict=True)
        st.table(pd.DataFrame(report).transpose())

        # SHAP values for RF (summary)
        try:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_test)
            st.subheader("SHAP Summary (RF)")
            # SHAP plots can be heavy; render a summary plot
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(plt)
        except Exception as e:
            st.write("SHAP plotting failed:", e)

        st.subheader("Raw Data Shape")
        st.write(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

else:
    st.info("Awaiting data upload to run the pipeline.")

