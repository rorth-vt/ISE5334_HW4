"""
ISE 5334 HW 4 Code

@author Reagan Orth
Disclaimer: I used ChatGPT to generate code, but have verified the results myself.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from plyfile import PlyData
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull

# -------------------------
# PATHS
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
FEASIBLE_PATH = BASE_DIR / "Data" / "feasible"
INFEASIBLE_PATH = BASE_DIR / "Data" / "infeasible"

st.set_page_config(layout="wide")
st.title("3D Point Cloud Explorer")

# -------------------------
# SIDEBAR NOTE
# -------------------------
st.sidebar.info(
    "Note: This app uses a limited dataset due to storage and performance constraints."
)

# -------------------------
# SESSION STATE
# -------------------------
for key in ["data", "X", "y", "results", "Xte", "yte", "models"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------
# LOAD + BUILD
# -------------------------
def load_xyz_from_ply(path):
    ply = PlyData.read(path)
    v = ply['vertex'].data
    return pd.DataFrame({"X": v["x"], "Y": v["y"], "Z": v["z"]})

def load_and_build(nf, ni):
    data = {}

    feasible_files = sorted(FEASIBLE_PATH.glob("*.ply"))[:nf]
    infeasible_files = sorted(INFEASIBLE_PATH.glob("*.ply"))[:ni]

    for i, f in enumerate(feasible_files):
        data[f"feasible_{i}"] = load_xyz_from_ply(str(f))

    for i, f in enumerate(infeasible_files):
        data[f"infeasible_{i}"] = load_xyz_from_ply(str(f))

    X, y = [], []
    for k, df in data.items():
        df = df[df["Z"] > df["Z"].min() + 2]
        X.append(extract_features(df))
        y.append(k.startswith("feasible"))

    return data, np.array(X), np.array(y)

# -------------------------
# FEATURES
# -------------------------
def extract_features(df):
    xyz = df[["X","Y","Z"]].values
    feats = []

    xyz_norm = (xyz - xyz.mean(axis=0)) / (xyz.std(axis=0) + 1e-8)

    feats.extend(xyz.mean(axis=0))
    feats.extend(xyz.std(axis=0))

    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    bbox = maxs - mins
    feats.extend(bbox)
    feats.append(np.prod(bbox))

    pca = PCA(n_components=3).fit(xyz_norm)
    l1, l2, l3 = np.sort(pca.explained_variance_)[::-1]

    feats.extend([l1, l2, l3])
    feats.extend([(l1-l2)/(l1+1e-8),(l2-l3)/(l1+1e-8),l3/(l1+1e-8)])

    nn = NearestNeighbors(n_neighbors=10).fit(xyz_norm)
    dists, _ = nn.kneighbors(xyz_norm)

    feats.append(dists[:,1].mean())
    feats.append(dists[:,1].std())

    try:
        clustering = DBSCAN(eps=0.05, min_samples=10).fit(xyz_norm)
        labels = clustering.labels_
        clusters = set(labels)
        clusters.discard(-1)
        sizes = [np.sum(labels == c) for c in clusters]

        if sizes:
            feats.extend([len(sizes), max(sizes)/len(xyz), np.mean(sizes), np.var(sizes)])
        else:
            feats.extend([0,0,0,0])
    except:
        feats.extend([0,0,0,0])

    try:
        kmeans = KMeans(n_clusters=3, n_init=10).fit(xyz_norm)
        feats.extend(kmeans.cluster_centers_.flatten())
    except:
        feats.extend([0]*9)

    try:
        hull = ConvexHull(xyz)
        feats.extend([hull.area, hull.volume, (hull.area**3)/(hull.volume**2 + 1e-8)])
    except:
        feats.extend([0,0,0])

    return np.array(feats)

# -------------------------
# MODELS
# -------------------------
def generate_models():
    return {
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "adaboost": AdaBoostClassifier(),
        "extra_trees": ExtraTreesClassifier(),
        "bagging": BaggingClassifier(),
    }

# -------------------------
# TRAIN
# -------------------------
def run_models(X, y):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    models = generate_models()
    results = {}
    trained = {}

    for name, m in models.items():
        m.fit(Xtr, ytr)
        preds = m.predict(Xte)

        results[name] = {
            "accuracy": accuracy_score(yte, preds),
            "f1": f1_score(yte, preds)
        }
        trained[name] = m

    return results, trained, Xte, yte

# -------------------------
# HELPERS
# -------------------------
def downsample(df, n=5000):
    if len(df) > n:
        return df.sample(n)
    return df

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Controls")

nf = st.sidebar.slider("Feasible samples", 1, 5, 3)
ni = st.sidebar.slider("Infeasible samples", 1, 5, 3)

if st.sidebar.button("1. Load + Build Dataset"):
    data, X, y = load_and_build(nf, ni)
    st.session_state.data = data
    st.session_state.X = X
    st.session_state.y = y
    st.success("Dataset ready")

if st.sidebar.button("2. Train Models"):
    if st.session_state.X is None:
        st.warning("Build dataset first")
    else:
        with st.spinner("Training models..."):
            results, models, Xte, yte = run_models(
                st.session_state.X,
                st.session_state.y
            )

            st.session_state.results = results
            st.session_state.models = models
            st.session_state.Xte = Xte
            st.session_state.yte = yte

        st.success("Training complete")

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["Point Cloud Viewer", "Model Viewer", "Comparison"])

# -------------------------
# TAB 1: POINT CLOUD
# -------------------------
with tab1:
    if st.session_state.data:
        key = st.selectbox("Select point cloud", list(st.session_state.data.keys()))
        df = st.session_state.data[key]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("3D View")
            df_plot = downsample(df)

            fig = px.scatter_3d(
                df_plot,
                x="X", y="Y", z="Z",
                color="Z",
                opacity=0.6,
                height=400
            )
            st.plotly_chart(fig, use_container_width=False)

        with col2:
            st.subheader("PCA Projection")
            xyz = df[["X","Y","Z"]].values
            reduced = PCA(n_components=2).fit_transform(xyz)

            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.scatter(reduced[:,0], reduced[:,1], s=2)
            st.pyplot(fig2)

# -------------------------
# TAB 2: MODEL VIEWER
# -------------------------
with tab2:
    if st.session_state.results:
        model_name = st.selectbox("Model", list(st.session_state.results.keys()))
        metrics = st.session_state.results[model_name]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Metrics")
            st.write(metrics)

        with col2:
            st.subheader("Confusion Matrix")
            model = st.session_state.models[model_name]
            preds = model.predict(st.session_state.Xte)

            cm = confusion_matrix(st.session_state.yte, preds)

            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

# -------------------------
# TAB 3: COMPARISON
# -------------------------
with tab3:
    if st.session_state.results:
        results = st.session_state.results

        selected = st.multiselect(
            "Select models",
            list(results.keys()),
            default=list(results.keys())
        )

        col1, col2 = st.columns(2)

        with col1:
            accs = [results[m]["accuracy"] for m in selected]

            fig, ax = plt.subplots(figsize=(5,4))
            ax.bar(selected, accs)
            ax.set_title("Accuracy")
            ax.set_xticklabels(selected, rotation=45, ha="right")
            st.pyplot(fig)

        with col2:
            f1s = [results[m]["f1"] for m in selected]

            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.bar(selected, f1s)
            ax2.set_title("F1 Score")
            ax2.set_xticklabels(selected, rotation=45, ha="right")
            st.pyplot(fig2)
