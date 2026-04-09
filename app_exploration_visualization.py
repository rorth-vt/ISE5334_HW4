import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score
)

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingRegressor, BaggingClassifier,
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Data generation
# -------------------------
@st.cache_data
def generate_data():
    X_reg, y_reg = make_regression(
        n_samples=500, n_features=50, n_informative=10,
        noise=10, random_state=42
    )

    X_clf, y_clf = make_classification(
        n_samples=500, n_features=50, n_informative=10,
        n_redundant=10, random_state=42
    )

    cols = [f"x{i}" for i in range(50)]

    df_reg = pd.DataFrame(X_reg, columns=cols)
    df_reg["target"] = y_reg

    df_clf = pd.DataFrame(X_clf, columns=cols)
    df_clf["target"] = y_clf

    return df_reg, df_clf

# -------------------------
# Models
# -------------------------
def get_models(task):
    if task == "Regression":
        return {
            "Decision Tree": DecisionTreeRegressor(),
            "Bagging": BaggingRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
        }
    else:
        return {
            "Decision Tree": DecisionTreeClassifier(),
            "Bagging": BaggingClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
        }

# -------------------------
# Cross-validation
# -------------------------
def cross_validate_models(X, y, task):
    models = get_models(task)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():

        if task == "Regression":
            scoring = ["neg_root_mean_squared_error", "r2"]
        else:
            scoring = ["accuracy"]

        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

        if task == "Regression":
            results[name] = {
                "RMSE (mean)": -scores["test_neg_root_mean_squared_error"].mean(),
                "RMSE (std)": scores["test_neg_root_mean_squared_error"].std(),
                "R2 (mean)": scores["test_r2"].mean(),
            }
        else:
            results[name] = {
                "Accuracy (mean)": scores["test_accuracy"].mean(),
                "Accuracy (std)": scores["test_accuracy"].std(),
            }

    return pd.DataFrame(results).T

# -------------------------
# EDA section
# -------------------------
def show_eda(df, task):
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Target Distribution")

    fig, ax = plt.subplots()

    if task == "Regression":
        sns.histplot(df["target"], kde=True, ax=ax)
    else:
        sns.countplot(x="target", data=df, ax=ax)

    st.pyplot(fig)

    st.subheader("Correlation Heatmap (Top 10 features)")

    corr = df.corr().abs()["target"].sort_values(ascending=False)
    top_features = corr.index[1:11]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[top_features].corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------
# Streamlit UI
# -------------------------
st.title("Tree-Based Pipelines Dashboard with CV & EDA")

task = st.radio("Select Task", ["Regression", "Classification"])

df_reg, df_clf = generate_data()

df = df_reg if task == "Regression" else df_clf

# Split features/target
X = df.drop(columns=["target"])
y = df["target"]

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
run_eda = st.sidebar.checkbox("Show Data Exploration", True)
run_cv = st.sidebar.checkbox("Run Cross Validation", True)

# -------------------------
# EDA
# -------------------------
if run_eda:
    show_eda(df, task)

# -------------------------
# Modeling
# -------------------------
if run_cv:
    st.subheader("Cross-Validation Results")
    results = cross_validate_models(X, y, task)

    st.dataframe(results)

    st.subheader("Performance Comparison")
    st.bar_chart(results)

# -------------------------
# Notes
# -------------------------
st.markdown("""
### Interpretation Guide

- **Bagging / Random Forest** → reduces variance  
- **Gradient Boosting** → reduces bias  
- **Cross-validation** → more reliable performance estimate  

### Tip
If results vary a lot (high std), your model is unstable.
""")
