import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingRegressor, BaggingClassifier,
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)

# -------------------------
# Data generation
# -------------------------
@st.cache_data
def generate_data():
    X_reg, y_reg = make_regression(
        n_samples=500, n_features=50, n_informative=10, noise=10, random_state=42
    )
    X_clf, y_clf = make_classification(
        n_samples=500, n_features=50, n_informative=10, n_redundant=10, random_state=42
    )
    return X_reg, y_reg, X_clf, y_clf

# -------------------------
# Model pipelines
# -------------------------
def get_models(task="regression"):
    if task == "regression":
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
# Evaluation
# -------------------------
def evaluate_models(X, y, task):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = get_models(task)
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task == "regression":
            results[name] = {
                "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                "R2": r2_score(y_test, preds),
            }
        else:
            results[name] = {
                "Accuracy": accuracy_score(y_test, preds)
            }

    return pd.DataFrame(results).T

# -------------------------
# Streamlit UI
# -------------------------
st.title("Tree-Based Model Pipelines Dashboard")

task = st.radio("Select Task", ["Regression", "Classification"])

X_reg, y_reg, X_clf, y_clf = generate_data()

if task == "Regression":
    results = evaluate_models(X_reg, y_reg, "regression")
else:
    results = evaluate_models(X_clf, y_clf, "classification")

st.subheader("Model Performance")
st.dataframe(results)

st.subheader("Performance Chart")
st.bar_chart(results)

st.markdown("""
### Pipelines Included:
- Decision Tree
- Bagging
- Random Forest
- Gradient Boosting

### Notes:
- Bagging reduces variance
- Boosting reduces bias
- Random Forest improves bagging via feature randomness
""")
