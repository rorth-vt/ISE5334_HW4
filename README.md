This code was written by ChatGPT, and I make no claim to its construction, accuracy, or utility for any real use case.

# 3D Point Cloud Classification Explorer

An interactive Streamlit app for exploring machine learning models on 3D point cloud data. This project extracts geometric features from .ply files and evaluates a wide range of classifiers (~50 pipelines) to distinguish between feasible and infeasible samples.

## 🚀 Features
### 🔍 Model Explorer
Select from ~50+ model pipelines
View:
Accuracy
F1 score
Quickly compare model performance
### 📊 Model Comparison
Visualize accuracy across model families:
Tree-based models
Bagging methods
Boosting methods
Identify which approaches perform best on your data
### 🧠 Feature Engineering Insights
Built-in explanation panel describing all computed features
Covers:
Global geometry
PCA-based shape descriptors
Local neighborhood structure
Clustering behavior
Convex hull metrics
## 📁 Project Structure
.
├── app.py
├── Data/
│   ├── feasible/
│   │   ├── *.ply
│   └── infeasible/
│       ├── *.ply
├── README.md
## 📦 Installation

Install dependencies:

pip install -r requirements.txt

Or manually:

pip install streamlit numpy pandas matplotlib scipy scikit-learn plyfile umap-learn
## ▶️ Running the App
streamlit run app.py
## 📊 Data Format
Input files must be .ply point clouds
Each file should contain:
x, y, z vertex coordinates

Directory structure is required:

Data/
├── feasible/
└── infeasible/

The label is inferred from the folder name.

## ⚙️ Feature Engineering

Each point cloud is converted into a feature vector using:

Global Features
Mean (X, Y, Z)
Standard deviation
Bounding box dimensions
Bounding box volume
PCA-Based Shape Features
Eigenvalues of covariance matrix
Linearity, planarity, scattering ratios
Local Geometry
Nearest neighbor distances (mean, std)
Clustering (DBSCAN)
Number of clusters
Largest cluster ratio
Cluster size statistics
KMeans Features
Cluster centers (k=3)
Convex Hull Metrics
Surface area
Volume
Compactness ratio
## 🤖 Models Evaluated

The app evaluates a wide variety of classifiers, including:

Logistic Regression
Support Vector Machines (multiple kernels)
k-Nearest Neighbors (multiple distance metrics)
Decision Trees
Random Forest
Gradient Boosting
AdaBoost
Bagging
Extra Trees
Naive Bayes
LDA
Neural Networks (MLP)
SGD / Passive-Aggressive
Gaussian Processes
Radius Neighbors

Each model is tested:

With and without feature scaling
Using a consistent train/test split
## 🧪 Experimental Setup
Train/test split: 70/30
Stratified sampling
Metrics:
Accuracy
F1 score (primary ranking metric)
## ⚠️ Notes on Performance
Feature extraction can be computationally expensive:
DBSCAN
Convex Hull
KMeans
Use small datasets (10–20 samples per class) for best performance on Streamlit Cloud
Results are cached to improve responsiveness
## 🌐 Deployment

This app is designed to run on Streamlit Cloud.

To deploy:

Push this repo to GitHub
Connect it to Streamlit Cloud
Ensure the Data/ folder is included in the repo
🔮 Future Improvements
Interactive 3D visualization (Plotly)
Confusion matrices and ROC curves
Feature importance analysis
Precomputed feature caching
Larger-scale dataset support

## License
Please never use this code for anything, for your own sake.
