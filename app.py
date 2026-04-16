"""
ISE 5334 HW 4 Code

@author Reagan Orth
Disclaimer: I used ChatGPT to generate code, but have verified the results myself.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plyfile import PlyData
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial import ConvexHull

# -------------------------
# PATHS
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
FEASIBLE_PATH = BASE_DIR / "Data" / "feasible"
INFEASIBLE_PATH = BASE_DIR / "Data" / "infeasible"

st.set_page_config(layout="wide")
st.title("3D Point Cloud Model Explorer")

# -------------------------
# SESSION STATE INIT
# -------------------------
for key in ["data", "X", "y", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------
# LOAD
# -------------------------
def load_xyz_from_ply(path):
    ply = PlyData.read(path)
    v = ply['vertex'].data
    return pd.DataFrame({"X": v["x"], "Y": v["y"], "Z": v["z"]})

def read_data(nf, ni):
    data = {}
    feasible_files = sorted(FEASIBLE_PATH.glob("*.ply"))
    infeasible_files = sorted(INFEASIBLE_PATH.glob("*.ply"))

    for i, f in enumerate(feasible_files[:nf]):
        data[f"feasible_{i}"] = load_xyz_from_ply(str(f))

    for i, f in enumerate(infeasible_files[:ni]):
        data[f"infeasible_{i}"] = load_xyz_from_ply(str(f))

    return data

# -------------------------
# FEATURES (UNCHANGED)
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

def build_dataset(data, filtered=True):
    X, y = [], []
    for k, df in data.items():
        if filtered:
            df = df[df["Z"] > df["Z"].min() + 2]

        X.append(extract_features(df))
        y.append(k.startswith("feasible"))

    return np.array(X), np.array(y)

# -------------------------
# MODELS (UNCHANGED)
# -------------------------
def generate_models():
    models = {}
    scalers = [None, StandardScaler()]

    classifiers = {
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "adaboost": AdaBoostClassifier(),
        "extra_trees": ExtraTreesClassifier(),
        "bagging": BaggingClassifier(),
    }

    count = 0
    for scaler in scalers:
        for name, clf in classifiers.items():
            steps = []
            if scaler:
                steps.append(("scaler", scaler))
            steps.append(("clf", clf))

            models[f"{name}_{'scaled' if scaler else 'noscale'}"] = Pipeline(steps)
            count += 1

    return models

# -------------------------
# TRAIN
# -------------------------
def run_models(X, y):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    models = generate_models()
    results = {}

    for name, m in models.items():
        try:
            m.fit(Xtr, ytr)
            preds = m.predict(Xte)
        except:
            preds = np.zeros_like(yte)

        results[name] = {
            "accuracy": accuracy_score(yte, preds),
            "f1": f1_score(yte, preds)
        }

    return results

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Controls")

nf = st.sidebar.slider("Feasible samples", 1, 10, 5)
ni = st.sidebar.slider("Infeasible samples", 1, 10, 5)

# -------------------------
# BUTTONS (CRITICAL CHANGE)
# -------------------------
if st.sidebar.button("1. Load Data"):
    st.session_state.data = read_data(nf, ni)
    st.success("Data loaded")

if st.sidebar.button("2. Build Dataset"):
    if st.session_state.data is None:
        st.warning("Load data first")
    else:
        X, y = build_dataset(st.session_state.data)
        st.session_state.X = X
        st.session_state.y = y
        st.success("Dataset built")

if st.sidebar.button("3. Train Models"):
    if st.session_state.X is None:
        st.warning("Build dataset first")
    else:
        st.session_state.results = run_models(
            st.session_state.X,
            st.session_state.y
        )
        st.success("Training complete")

# =========================
# MAIN UI
# =========================

tab1, tab2 = st.tabs(["Model Viewer", "Comparison"])

# -------------------------
# TAB 1: SIDE-BY-SIDE MODELS
# -------------------------
with tab1:
    if st.session_state.results:
        models = list(st.session_state.results.keys())

        col1, col2 = st.columns(2)

        with col1:
            m1 = st.selectbox("Model A", models, key="m1")
        with col2:
            m2 = st.selectbox("Model B", models, key="m2")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(m1)
            st.write(st.session_state.results[m1])

        with col2:
            st.subheader(m2)
            st.write(st.session_state.results[m2])

# -------------------------
# TAB 2: COMPARISON PLOT
# -------------------------
with tab2:
    if st.session_state.results:
        results = st.session_state.results

        selected = st.multiselect(
            "Select models to compare",
            list(results.keys()),
            default=list(results.keys())[:4]
        )

        if selected:
            accs = [results[m]["accuracy"] for m in selected]

            fig, ax = plt.subplots()
            ax.bar(selected, accs)
            ax.set_ylabel("Accuracy")
            ax.set_xticklabels(selected, rotation=45, ha="right")

            st.pyplot(fig)
