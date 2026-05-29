## Environment

- Python 3.10+
- Scikit-learn 1.x
- XGBoost 2.x+
- SHAP 0.4+
- Pandas 2.x
- NumPy 1.26+
- 32–256 GB RAM recommended (depends on feature combinations)
- Multi-core CPU strongly recommended for RandomizedSearchCV

---

# Cover Crop Biomass Modeling Using Machine Learning

## Overview

This project implements a modular machine learning workflow for predicting biomass using multi-source feature sets.  
It supports multiple feature combinations, automated feature selection, hyperparameter tuning, model evaluation, and explainability using SHAP.

---

# Project Overview

The pipeline performs:

- Feature combination from multiple groups (e.g., VIs, SFs, SBs, TFs)
- Multicollinearity removal using VIF
- Data scaling (X and y standardization)
- Model training using multiple regressors:
  - Random Forest (RF)
  - Support Vector Regression (SVR)
  - Partial Least Squares Regression (PLSR)
  - XGBoost (XGB)
- Hyperparameter optimization using RandomizedSearchCV
- Cross-validation evaluation
- Test evaluation
- Model explainability using SHAP
- Export of results to Excel

---

# Models Used

| Model | Description |
|------|------------|
| RF | Ensemble tree-based regression |
| SVR | Kernel-based regression |
| PLSR | Linear latent variable regression |
| XGB | Gradient boosting regression |

---

# Workflow Pipeline

1. Load training and testing data  
2. Generate feature combinations  
3. Apply VIF-based feature selection  
4. Standardize features and target variable  
5. Train models using RandomizedSearchCV  
6. Perform cross-validation  
7. Train final model on full dataset  
8. Evaluate on training and test sets  
9. Compute SHAP feature importance  
10. Save outputs to Excel files  
## Repository Structure

```text
project/
│
├── Data/                      # Input datasets (train/test Excel files)
├── outputs/                   # Model outputs (Excel results per experiment)
│
├── src/
│   ├── main.py                # Main workflow pipeline
│   ├── config.py              # Global parameters and settings
│   ├── models.py              # Model factory (RF, SVR, PLSR, XGB)
│   ├── tuning.py              # Hyperparameter tuning (RandomizedSearchCV)
│   ├── evaluation.py          # Cross-validation and evaluation metrics
│   ├── explainability.py      # SHAP-based feature importance
│   ├── utils.py               # Preprocessing + VIF + helper functions
│
├── requirements.txt
└── README.md
