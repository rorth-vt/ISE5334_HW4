"""
ISE 5334 HW 4 Code

@author Reagan Orth
Disclaimer: I used ChatGPT to generate code, but have verified the results myself.
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plyfile import PlyData

from scipy.stats import skew, kurtosis
from scipy.spatial import ConvexHull

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from pathlib import Path
# import umap

# -------------------------
# PATHS
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
FEASIBLE_PATH = BASE_DIR / "Data" / "feasible"
INFEASIBLE_PATH = BASE_DIR / "Data" / "infeasible"


# -------------------------
# LOAD PLY
# -------------------------
def load_xyz_from_ply(path):
    ply = PlyData.read(path)
    v = ply['vertex'].data
    return pd.DataFrame({"X": v["x"], "Y": v["y"], "Z": v["z"]})


# -------------------------
# 3D VISUALIZATION
# -------------------------
def visualize_3d(df, title):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["X"], df["Y"], df["Z"], s=1, alpha=0.3)
    ax.set_title(title)
    plt.show()


# -------------------------
# PCA VALIDATION
# -------------------------
def pca_plot(df, title):
    xyz = df[["X","Y","Z"]].values
    reduced = PCA(n_components=2).fit_transform(xyz)

    plt.scatter(reduced[:,0], reduced[:,1], s=1, alpha=0.3)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


# -------------------------
# PLANE FILTER
# -------------------------
def filter_plane(df, z_thresh):
    return df[df["Z"] > z_thresh]


# -------------------------
# ADVANCED FEATURES
# -------------------------
def extract_features(df):

    xyz = df[["X","Y","Z"]].values
    feats = []

    # -------------------------
    # Normalize
    # -------------------------
    xyz_norm = (xyz - xyz.mean(axis=0)) / (xyz.std(axis=0) + 1e-8)

    # -------------------------
    # GLOBAL FEATURES
    # -------------------------
    feats.extend(xyz.mean(axis=0))
    feats.extend(xyz.std(axis=0))

    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    bbox = maxs - mins
    feats.extend(bbox)
    feats.append(np.prod(bbox))

    # -------------------------
    # PCA (global)
    # -------------------------
    pca = PCA(n_components=3).fit(xyz_norm)
    l1, l2, l3 = np.sort(pca.explained_variance_)[::-1]

    feats.extend([l1, l2, l3])
    feats.extend([
        (l1-l2)/(l1+1e-8),
        (l2-l3)/(l1+1e-8),
        l3/(l1+1e-8)
    ])

    # -------------------------
    # LOCAL FEATURES
    # -------------------------
    nn = NearestNeighbors(n_neighbors=10).fit(xyz_norm)
    dists, idxs = nn.kneighbors(xyz_norm)

    feats.append(dists[:,1].mean())
    feats.append(dists[:,1].std())

    # -------------------------
    # DBSCAN (features of interest, very slow for large datasets)
    # -------------------------
    try:
        clustering = DBSCAN(eps=0.05, min_samples=10).fit(xyz_norm)
        labels = clustering.labels_

        clusters = set(labels)
        clusters.discard(-1)

        sizes = [np.sum(labels == c) for c in clusters]

        if sizes:
            feats.extend([
                len(sizes),
                max(sizes)/len(xyz),
                np.mean(sizes),
                np.var(sizes)
            ])
        else:
            feats.extend([0,0,0,0])
    except:
        feats.extend([0,0,0,0])

    # -------------------------
    # UMAP FEATURES (attempted, but runtime was >30 hours for 40 samples)
    # -------------------------
    # try:
    #     reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    #     embedding = reducer.fit_transform(xyz_norm)
    #
    #     # Use embedding statistics
    #     feats.extend(embedding.mean(axis=0))
    #     feats.extend(embedding.std(axis=0))
    #
    #     # Spread / structure in embedding
    #     mins_e = embedding.min(axis=0)
    #     maxs_e = embedding.max(axis=0)
    #     feats.extend(maxs_e - mins_e)
    #
    # except:
    #     feats.extend([0]*6)

    # -------------------------
    # KMEANS (unsupervised)
    # -------------------------
    try:
        kmeans = KMeans(n_clusters=3, n_init=10).fit(xyz_norm)
        feats.extend(kmeans.cluster_centers_.flatten())
    except:
        feats.extend([0]*9)

    # -------------------------
    # CONVEX HULL
    # -------------------------
    try:
        hull = ConvexHull(xyz)
        feats.extend([
            hull.area,
            hull.volume,
            (hull.area**3)/(hull.volume**2 + 1e-8)
        ])
    except:
        feats.extend([0,0,0])

    return np.array(feats)


# -------------------------
# READ DATA
# -------------------------
def read_data(nf, ni):
    data = {}

    feasible_files = list(FEASIBLE_PATH.glob("*.ply"))
    infeasible_files = list(INFEASIBLE_PATH.glob("*.ply"))

    for i, f in enumerate(feasible_files):
        df = load_xyz_from_ply(str(f))
        data[f"feasible_{i}"] = df
        if i >= nf:
            break

    for i, f in enumerate(infeasible_files):
        df = load_xyz_from_ply(str(f))
        data[f"infeasible_{i}"] = df
        if i >= ni:
            break

    return data


# -------------------------
# BUILD DATASET
# -------------------------
def build_dataset(data, filtered=False, z_thresh=0.0):
    X, y = [], []
    for k, df in data.items():
        if filtered:
            df_f = filter_plane(df, df["Z"].min() + 2)
            df = df_f

        X.append(extract_features(df))
        y.append(k.startswith("feasible"))
        print(f"Building dataset, {len(y)}")

    return np.array(X), np.array(y)


# -------------------------
# 50 MODELS
# -------------------------
def generate_models():

    models = {}
    scalers = [None, StandardScaler()]

    classifiers = {

        # -------------------------
        # Linear Models
        # -------------------------
        "logreg_l2": LogisticRegression(penalty="l2", max_iter=500),
        "logreg_l1": LogisticRegression(penalty="l1", solver="liblinear", max_iter=500),
        "ridge": LogisticRegression(penalty="l2", solver="lbfgs", max_iter=500),

        # -------------------------
        # SVM (kernel = fundamentally different)
        # -------------------------
        "svm_linear": SVC(kernel="linear", probability=True, max_iter=500),
        "svm_rbf": SVC(kernel="rbf", probability=True, max_iter=500),
        "svm_poly": SVC(kernel="poly", degree=3, probability=True, max_iter=500),
        "svm_sigmoid": SVC(kernel="sigmoid", probability=True, max_iter=500),

        # -------------------------
        # kNN (distance metric = different geometry)
        # -------------------------
        "knn_euclidean": KNeighborsClassifier(metric="euclidean"),
        "knn_manhattan": KNeighborsClassifier(metric="manhattan"),
        "knn_chebyshev": KNeighborsClassifier(metric="chebyshev"),
        "knn_minkowski": KNeighborsClassifier(metric="minkowski"),

        # -------------------------
        # Tree-based
        # -------------------------
        "decision_tree_gini": DecisionTreeClassifier(criterion="gini"),
        "decision_tree_entropy": DecisionTreeClassifier(criterion="entropy"),

        "random_forest": RandomForestClassifier(),  # single version only

        "gradient_boosting": GradientBoostingClassifier(),

        # -------------------------
        # Probabilistic
        # -------------------------
        "naive_bayes": GaussianNB(),

        # -------------------------
        # Discriminant
        # -------------------------
        "lda": LinearDiscriminantAnalysis(),

        # -------------------------
        # Additional distinct models
        # -------------------------
        "extra_tree": DecisionTreeClassifier(splitter="random"),
        "adaboost": AdaBoostClassifier(),
        "extra_trees": ExtraTreesClassifier(),
        "bagging": BaggingClassifier(),
        "mlp": MLPClassifier(max_iter=500),
        "sgd_hinge": SGDClassifier(loss="hinge"),
        "sgd_log": SGDClassifier(loss="log_loss"),
        "passive_aggressive": PassiveAggressiveClassifier(),
        "gaussian_process": GaussianProcessClassifier(1.0 * RBF(1.0)),
        "radius_neighbors": RadiusNeighborsClassifier(radius=1.0),
    }

    count = 0
    for scaler in scalers:
        for name, clf in classifiers.items():
            steps = []
            if scaler is not None:
                steps.append(("scaler", scaler))
            steps.append(("clf", clf))
            models[f"model_{count}_{name}_{'scaled' if scaler else 'noscale'}"] = Pipeline(steps)
            count += 1

    print(f"Total pipelines generated: {len(models)}")
    return models


# -------------------------
# RUN EXPERIMENT
# -------------------------
def run(X, y, label):

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    models = generate_models()
    results = {}

    print(f"\n==== {label} ====")

    for name, m in models.items():
        # print(f"=> Training {name}... ", end="")

        if "bagging" not in name and "boosting" not in name and "tree" not in name and "forest" not in name:
            continue  # only use subset for Streamlit's sake
        
        try:
            m.fit(Xtr, ytr)
            preds = m.predict(Xte)
        except:
            preds = np.zeros_like(yte)  # if training fails we'll use 0

        acc = accuracy_score(yte, preds)
        f1 = f1_score(yte, preds)

        results[name] = {
            "accuracy": acc,
            "f1": f1
        }

        print(f"Acc: {acc:.3f}, F1: {f1:.3f}")

    # -------------------------
    # Sort by F1 (better metric)
    # -------------------------
    best = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)[:5]

    print("\nTop 5 (by F1):")
    for name, metrics in best:
        print(f"{name} | Acc: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")

    return results


# =========================
# MAIN
# =========================

# set to 10 total for the poor streamlit server's sake
NUM_FEASIBLE = 5
NUM_INFEASIBLE = 5

data = read_data(NUM_FEASIBLE, NUM_INFEASIBLE)

# -------------------------
# Pick one example of each
# -------------------------
feasible_key = next(k for k in data if k.startswith("feasible"))
infeasible_key = next(k for k in data if k.startswith("infeasible"))

df_feasible = data[feasible_key]
df_infeasible = data[infeasible_key]

# -------------------------
# Visualization BEFORE pipeline
# -------------------------
print("\n=== VISUAL VALIDATION ===")

"""
# ---- Feasible sample
visualize_3d(df_feasible, f"{feasible_key} - ORIGINAL")
pca_plot(df_feasible, f"{feasible_key} - PCA BEFORE")

df_feasible_filt = filter_plane(df_feasible, z_thresh=df_feasible["Z"].min() + 5)

visualize_3d(df_feasible_filt, f"{feasible_key} - FILTERED")
pca_plot(df_feasible_filt, f"{feasible_key} - PCA AFTER")

# ---- Infeasible sample
visualize_3d(df_infeasible, f"{infeasible_key} - ORIGINAL")
pca_plot(df_infeasible, f"{infeasible_key} - PCA BEFORE")

df_infeasible_filt = filter_plane(df_infeasible, z_thresh=df_infeasible["Z"].min() + 5)

visualize_3d(df_infeasible_filt, f"{infeasible_key} - FILTERED")
pca_plot(df_infeasible_filt, f"{infeasible_key} - PCA AFTER")
"""

# -------------------------
# Build datasets
# -------------------------
# X_u, y_u = build_dataset(data, filtered=False)
X_f, y_f = build_dataset(data, filtered=True)

# -------------------------
# Run experiments
# -------------------------
# res_u = run(X_u, y_u, "UNFILTERED")
res_f = run(X_f, y_f, "FILTERED")
