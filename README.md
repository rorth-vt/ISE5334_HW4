# Tree Model Cross-Validation Studio

A professional Streamlit dashboard for benchmarking tree-based machine learning models across uploaded datasets.

## What the app does

- Upload multiple CSV files and Excel workbooks
- Select target and feature columns for each dataset
- Run cross-validation for tree-based models
- Support both classification and regression workflows
- Export benchmark summaries to Excel, CSV, and JSON
- Display feature importance for fitted tree models

## Included model families

Implemented and benchmarked in the dashboard:
- CART
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- XGBoost (optional dependency)
- LightGBM (optional dependency)
- CatBoost (optional dependency)

Included as documentation notes only, not benchmarked automatically:
- C4.5
- GUIDE

Reason: production-grade Python support for C4.5 and GUIDE is limited and inconsistent compared with the models above.

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud / GitHub deployment

1. Create a new GitHub repository.
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Push to GitHub.
4. In Streamlit Community Cloud, create a new app from that repository.
5. Set the main file path to:

```bash
app.py
```

## Recommended repository structure

```text
tree_cv_dashboard/
├── app.py
├── requirements.txt
└── README.md
```

## Notes

- The app automatically handles missing values with imputation.
- Categorical variables are encoded automatically.
- Classification uses stratified K-fold by default.
- Regression uses shuffled K-fold.
- For large datasets, boosted models may take longer to run.
- If Streamlit Cloud memory is tight, remove optional packages you do not need from `requirements.txt`.

## Suggested next extensions

- SHAP explainability panel
- Hyperparameter tuning tab
- Model comparison plots with statistical tests
- PDF report export
- Per-fold predictions and residual diagnostics
