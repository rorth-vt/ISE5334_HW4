from __future__ import annotations

import io
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)

# Optional third-party models
OPTIONAL_IMPORTS: Dict[str, bool] = {}
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    OPTIONAL_IMPORTS["xgboost"] = True
except Exception:
    OPTIONAL_IMPORTS["xgboost"] = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
    OPTIONAL_IMPORTS["lightgbm"] = True
except Exception:
    OPTIONAL_IMPORTS["lightgbm"] = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
    OPTIONAL_IMPORTS["catboost"] = True
except Exception:
    OPTIONAL_IMPORTS["catboost"] = False

st.set_page_config(
    page_title="Tree Model Cross-Validation Studio",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
    .metric-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: linear-gradient(180deg, rgba(240,242,246,0.45), rgba(255,255,255,0.85));
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    .section-card {
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        background: rgba(255,255,255,0.75);
        margin-bottom: 0.9rem;
    }
    .small-note {
        color: #5f6570;
        font-size: 0.92rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@dataclass
class DatasetBundle:
    name: str
    frame: pd.DataFrame


CLASSIFICATION_SCORERS = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
}

REGRESSION_SCORERS = {
    "r2": "r2",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
}


MODEL_DESCRIPTIONS = {
    "CART": "Standard greedy decision tree implemented with scikit-learn.",
    "Random Forest": "Bootstrap aggregation of decision trees with feature randomness.",
    "Extra Trees": "Extremely randomized trees using random split thresholds.",
    "Gradient Boosting": "Sequential residual-fitting tree ensemble.",
    "AdaBoost": "Adaptive boosting with shallow tree weak learners.",
    "XGBoost": "Regularized gradient boosting. Optional dependency.",
    "LightGBM": "Leaf-wise gradient boosting. Optional dependency.",
    "CatBoost": "Boosting with native categorical handling. Optional dependency.",
    "C4.5": "Direct production-grade Python support is not included here; not benchmarked in this app.",
    "GUIDE": "Direct production-grade Python support is not included here; not benchmarked in this app.",
}


def infer_task(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique <= 10 and nunique / max(len(y), 1) < 0.05:
            return "classification"
        return "regression"
    return "classification"


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_bytes: bytes, filename: str) -> List[DatasetBundle]:
    suffix = os.path.splitext(filename)[1].lower()
    data_stream = io.BytesIO(file_bytes)
    bundles: List[DatasetBundle] = []

    if suffix == ".csv":
        df = pd.read_csv(data_stream)
        bundles.append(DatasetBundle(name=filename, frame=df))
    elif suffix in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(data_stream)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            bundles.append(DatasetBundle(name=f"{filename} :: {sheet_name}", frame=df))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return bundles


@st.cache_resource(show_spinner=False)
def build_model_catalog(random_state: int) -> Dict[str, Dict[str, object]]:
    models: Dict[str, Dict[str, object]] = {
        "classification": {
            "CART": DecisionTreeClassifier(random_state=random_state, max_depth=None),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, random_state=random_state, n_jobs=-1
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=300, random_state=random_state, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "AdaBoost": AdaBoostClassifier(random_state=random_state),
        },
        "regression": {
            "CART": DecisionTreeRegressor(random_state=random_state, max_depth=None),
            "Random Forest": RandomForestRegressor(
                n_estimators=300, random_state=random_state, n_jobs=-1
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=300, random_state=random_state, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
            "AdaBoost": AdaBoostRegressor(random_state=random_state),
        },
    }

    if OPTIONAL_IMPORTS["xgboost"]:
        models["classification"]["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=4,
        )
        models["regression"]["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=4,
        )

    if OPTIONAL_IMPORTS["lightgbm"]:
        models["classification"]["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
            verbose=-1,
        )
        models["regression"]["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
            verbose=-1,
        )

    if OPTIONAL_IMPORTS["catboost"]:
        models["classification"]["CatBoost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
        )
        models["regression"]["CatBoost"] = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
        )

    return models



def build_preprocessor(X: pd.DataFrame, use_one_hot: bool = True) -> ColumnTransformer:
    categorical_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            if use_one_hot
            else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")



def select_preprocessor_for_model(X: pd.DataFrame, model_name: str) -> ColumnTransformer:
    if model_name == "CatBoost" and OPTIONAL_IMPORTS["catboost"]:
        # Keep ordinal encoding for generic sklearn pipeline compatibility.
        return build_preprocessor(X, use_one_hot=False)
    return build_preprocessor(X, use_one_hot=True)



def summarize_cv_results(cv_results: Dict[str, np.ndarray], task: str) -> Dict[str, float]:
    output: Dict[str, float] = {}
    output["fit_time_mean_s"] = float(np.mean(cv_results["fit_time"]))
    output["score_time_mean_s"] = float(np.mean(cv_results["score_time"]))

    if task == "classification":
        output["accuracy_mean"] = float(np.mean(cv_results["test_accuracy"]))
        output["accuracy_std"] = float(np.std(cv_results["test_accuracy"]))
        output["f1_macro_mean"] = float(np.mean(cv_results["test_f1_macro"]))
        output["f1_weighted_mean"] = float(np.mean(cv_results["test_f1_weighted"]))
    else:
        output["r2_mean"] = float(np.mean(cv_results["test_r2"]))
        output["r2_std"] = float(np.std(cv_results["test_r2"]))
        output["rmse_mean"] = float(-np.mean(cv_results["test_neg_root_mean_squared_error"]))
        output["mae_mean"] = float(-np.mean(cv_results["test_neg_mean_absolute_error"]))

    return output



def fit_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    model_name: str,
    estimator,
) -> pd.DataFrame:
    preprocessor = select_preprocessor_for_model(X, model_name)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", clone(estimator)),
    ])

    y_fit = y.copy()
    if task == "classification" and not pd.api.types.is_numeric_dtype(y_fit):
        le = LabelEncoder()
        y_fit = pd.Series(le.fit_transform(y_fit.astype(str)), index=y_fit.index)

    pipe.fit(X, y_fit)

    model = pipe.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(len(importances))])

    feat_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=False).head(20).reset_index(drop=True)
    return feat_df



def make_excel_report(results_df: pd.DataFrame, metadata_df: pd.DataFrame, feature_map: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="CV Summary", index=False)
        metadata_df.to_excel(writer, sheet_name="Dataset Metadata", index=False)
        for sheet_name, feat_df in feature_map.items():
            safe_name = sheet_name[:31]
            feat_df.to_excel(writer, sheet_name=safe_name, index=False)
    output.seek(0)
    return output.getvalue()



def make_json_report(results_df: pd.DataFrame, metadata_df: pd.DataFrame) -> bytes:
    payload = {
        "summary": results_df.to_dict(orient="records"),
        "metadata": metadata_df.to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2).encode("utf-8")


st.title("🌲 Tree Model Cross-Validation Studio")
st.caption(
    "Upload one or more CSV / Excel datasets, benchmark professional tree-based models, and export a polished cross-validation report."
)

with st.sidebar:
    st.header("Configuration")
    random_state = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    folds = st.slider("Cross-validation folds", min_value=3, max_value=10, value=5, step=1)
    use_stratified = st.toggle("Use stratified CV for classification", value=True)
    st.divider()
    st.subheader("Model availability")
    availability_rows = []
    for name in ["xgboost", "lightgbm", "catboost"]:
        availability_rows.append({
            "package": name,
            "status": "Available" if OPTIONAL_IMPORTS[name] else "Not installed",
        })
    st.dataframe(pd.DataFrame(availability_rows), use_container_width=True, hide_index=True)
    st.markdown(
        "<div class='small-note'>C4.5 and GUIDE are noted in the app for completeness, but not benchmarked because mainstream Python production support is limited and inconsistent.</div>",
        unsafe_allow_html=True,
    )

model_catalog = build_model_catalog(int(random_state))

uploaded_files = st.file_uploader(
    "Upload datasets",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="You can upload multiple CSV files and multi-sheet Excel workbooks.",
)

if not uploaded_files:
    st.info("Upload at least one dataset to begin.")
    st.stop()

all_datasets: List[DatasetBundle] = []
errors: List[str] = []
for uploaded in uploaded_files:
    try:
        bundles = load_uploaded_dataset(uploaded.getvalue(), uploaded.name)
        all_datasets.extend(bundles)
    except Exception as exc:
        errors.append(f"{uploaded.name}: {exc}")

if errors:
    st.warning("Some files could not be loaded:\n\n- " + "\n- ".join(errors))

if not all_datasets:
    st.error("No valid datasets were loaded.")
    st.stop()

st.success(f"Loaded {len(all_datasets)} dataset table(s).")

dataset_names = [bundle.name for bundle in all_datasets]
selected_dataset_names = st.multiselect(
    "Select datasets to analyze",
    options=dataset_names,
    default=dataset_names,
)

selected_bundles = [bundle for bundle in all_datasets if bundle.name in selected_dataset_names]

if not selected_bundles:
    st.warning("Select at least one dataset.")
    st.stop()

with st.expander("Tree model coverage notes", expanded=False):
    coverage_df = pd.DataFrame(
        [{"Model": k, "Implementation note": v} for k, v in MODEL_DESCRIPTIONS.items()]
    )
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)

results_records: List[Dict[str, object]] = []
metadata_records: List[Dict[str, object]] = []
feature_map: Dict[str, pd.DataFrame] = {}

for bundle in selected_bundles:
    df = bundle.frame.copy()

    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader(bundle.name)
        col1, col2, col3 = st.columns([1.2, 1.1, 1.7])
        with col1:
            st.write(f"**Rows:** {df.shape[0]:,}")
            st.write(f"**Columns:** {df.shape[1]:,}")
        with col2:
            st.write(f"**Missing cells:** {int(df.isna().sum().sum()):,}")
            st.write(f"**Duplicate rows:** {int(df.duplicated().sum()):,}")
        with col3:
            st.dataframe(df.head(5), use_container_width=True)

        columns = list(df.columns)
        target_col = st.selectbox(
            f"Target column — {bundle.name}",
            options=columns,
            key=f"target_{bundle.name}",
        )
        feature_cols_default = [c for c in columns if c != target_col]
        feature_cols = st.multiselect(
            f"Feature columns — {bundle.name}",
            options=[c for c in columns if c != target_col],
            default=feature_cols_default,
            key=f"features_{bundle.name}",
        )

        if not feature_cols:
            st.error("At least one feature column must be selected.")
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        y = df[target_col]
        suggested_task = infer_task(y)
        task = st.radio(
            f"Task type — {bundle.name}",
            options=["classification", "regression"],
            index=0 if suggested_task == "classification" else 1,
            horizontal=True,
            key=f"task_{bundle.name}",
        )

        available_models = list(model_catalog[task].keys())
        default_models = [m for m in ["CART", "Random Forest", "Extra Trees", "Gradient Boosting"] if m in available_models]
        selected_models = st.multiselect(
            f"Models — {bundle.name}",
            options=available_models,
            default=default_models,
            key=f"models_{bundle.name}",
        )

        if not selected_models:
            st.warning("Select at least one model for benchmarking.")
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        run_now = st.button(f"Run cross-validation — {bundle.name}", key=f"run_{bundle.name}")
        if run_now:
            work_df = df[feature_cols + [target_col]].copy().dropna(axis=0, how="all")
            work_df = work_df.loc[:, ~work_df.columns.duplicated()].copy()
            work_df = work_df.dropna(subset=[target_col])

            X = work_df[feature_cols].copy()
            y = work_df[target_col].copy()

            metadata_records.append({
                "dataset": bundle.name,
                "rows_used": int(len(work_df)),
                "feature_count": int(X.shape[1]),
                "target": target_col,
                "task": task,
                "missing_feature_cells": int(X.isna().sum().sum()),
                "class_count": int(y.nunique()) if task == "classification" else np.nan,
            })

            if task == "classification":
                if not pd.api.types.is_numeric_dtype(y):
                    le = LabelEncoder()
                    y_for_cv = le.fit_transform(y.astype(str))
                else:
                    y_for_cv = y.to_numpy()
                cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(random_state)) if use_stratified else KFold(n_splits=folds, shuffle=True, random_state=int(random_state))
                scoring = CLASSIFICATION_SCORERS
            else:
                y_for_cv = pd.to_numeric(y, errors="coerce")
                valid_mask = ~pd.isna(y_for_cv)
                X = X.loc[valid_mask].copy()
                y_for_cv = y_for_cv.loc[valid_mask].to_numpy()
                cv = KFold(n_splits=folds, shuffle=True, random_state=int(random_state))
                scoring = REGRESSION_SCORERS

            progress = st.progress(0.0)
            dataset_results: List[Dict[str, object]] = []

            for idx, model_name in enumerate(selected_models):
                estimator = model_catalog[task][model_name]
                preprocessor = select_preprocessor_for_model(X, model_name)
                pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", clone(estimator)),
                ])

                with st.spinner(f"Running {model_name} on {bundle.name}..."):
                    cv_results = cross_validate(
                        pipe,
                        X,
                        y_for_cv,
                        cv=cv,
                        scoring=scoring,
                        return_train_score=False,
                        error_score="raise",
                        n_jobs=1,
                    )
                    summary = summarize_cv_results(cv_results, task)
                    result_record = {
                        "dataset": bundle.name,
                        "task": task,
                        "model": model_name,
                        **summary,
                    }
                    results_records.append(result_record)
                    dataset_results.append(result_record)

                    feat_df = fit_feature_importance(X, pd.Series(y_for_cv), task, model_name, estimator)
                    feature_map[f"{bundle.name} - {model_name}"] = feat_df

                progress.progress((idx + 1) / len(selected_models))

            progress.empty()

            dataset_results_df = pd.DataFrame(dataset_results)
            st.markdown("### Cross-validation summary")
            st.dataframe(dataset_results_df, use_container_width=True, hide_index=True)

            if not dataset_results_df.empty:
                sort_col = "accuracy_mean" if task == "classification" else "r2_mean"
                best_row = dataset_results_df.sort_values(sort_col, ascending=False).iloc[0]
                a, b, c = st.columns(3)
                with a:
                    st.metric("Best model", best_row["model"])
                with b:
                    st.metric("Datasets rows used", f"{int(len(X)):,}")
                with c:
                    st.metric("Primary CV score", f"{best_row[sort_col]:.4f}")

                for model_name in selected_models:
                    feat_key = f"{bundle.name} - {model_name}"
                    feat_df = feature_map.get(feat_key, pd.DataFrame())
                    if not feat_df.empty:
                        st.markdown(f"#### Top feature importances — {model_name}")
                        st.bar_chart(feat_df.set_index("feature")["importance"])
                        break

        st.markdown("</div>", unsafe_allow_html=True)

summary_tab, export_tab = st.tabs(["Portfolio Summary", "Export Center"])

with summary_tab:
    if results_records:
        results_df = pd.DataFrame(results_records)
        metadata_df = pd.DataFrame(metadata_records)
        st.subheader("All benchmarked results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        if not results_df.empty:
            primary_metric = "accuracy_mean" if (results_df["task"] == "classification").all() else None
            cls_results = results_df[results_df["task"] == "classification"]
            reg_results = results_df[results_df["task"] == "regression"]

            if not cls_results.empty:
                st.markdown("### Classification leaderboard")
                st.dataframe(
                    cls_results.sort_values(["dataset", "accuracy_mean"], ascending=[True, False]),
                    use_container_width=True,
                    hide_index=True,
                )
            if not reg_results.empty:
                st.markdown("### Regression leaderboard")
                st.dataframe(
                    reg_results.sort_values(["dataset", "r2_mean"], ascending=[True, False]),
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("Run at least one dataset benchmark to populate the summary.")

with export_tab:
    if results_records:
        results_df = pd.DataFrame(results_records)
        metadata_df = pd.DataFrame(metadata_records)
        excel_bytes = make_excel_report(results_df, metadata_df, feature_map)
        json_bytes = make_json_report(results_df, metadata_df)
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Excel report",
            data=excel_bytes,
            file_name="tree_cv_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download CSV summary",
            data=csv_bytes,
            file_name="tree_cv_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download JSON summary",
            data=json_bytes,
            file_name="tree_cv_summary.json",
            mime="application/json",
        )
    else:
        st.info("Run at least one benchmark first, then export the report files here.")
