import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("📊 ML Dashboard: XGBoost vs Bagging")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Expected columns
    feature_cols = [f"feature_{i}" for i in range(1, 21)]
    target_col = "target"

    # Check columns
    if not all(col in df.columns for col in feature_cols + [target_col]):
        st.error("CSV must contain feature_1 ... feature_20 and target column")
    else:
        # Split data
        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        st.subheader("🔍 Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Summary Statistics")
            st.write(df.describe())

        with col2:
            st.write("Missing Values")
            st.write(df.isnull().sum())

        # Correlation heatmap
        st.subheader("📌 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Train models
        st.subheader("🤖 Model Training")

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss"
        )

        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)

        # Bagging
        bag_model = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=50,
            random_state=42
        )

        bag_model.fit(X_train, y_train)
        bag_preds = bag_model.predict(X_test)

        # Metrics
        xgb_acc = accuracy_score(y_test, xgb_preds)
        bag_acc = accuracy_score(y_test, bag_preds)

        st.subheader("📈 Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("XGBoost Accuracy", f"{xgb_acc:.4f}")

        with col2:
            st.metric("Bagging Accuracy", f"{bag_acc:.4f}")

        # Classification Reports
        st.subheader("📋 Classification Reports")

        st.write("### XGBoost")
        st.text(classification_report(y_test, xgb_preds))

        st.write("### Bagging")
        st.text(classification_report(y_test, bag_preds))

        # Feature Importance (XGBoost)
        st.subheader("🔥 Feature Importance (XGBoost)")

        importance = xgb_model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            ax=ax2
        )
        st.pyplot(fig2)

        # Prediction comparison
        st.subheader("📊 Prediction Comparison")

        comparison_df = pd.DataFrame({
            "Actual": y_test,
            "XGBoost": xgb_preds,
            "Bagging": bag_preds
        }).reset_index(drop=True)

        st.dataframe(comparison_df.head(50))

        # Accuracy comparison chart
        st.subheader("📉 Accuracy Comparison")

        fig3, ax3 = plt.subplots()
        ax3.bar(["XGBoost", "Bagging"], [xgb_acc, bag_acc], color=["blue", "green"])
        ax3.set_ylabel("Accuracy")
        st.pyplot(fig3)

else:
    st.info("Please upload a CSV file to begin.")
