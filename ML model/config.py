import os

SEED = 42 ## for reproducibility 

TARGET = "Biomass"

OUTPUT_PATH = "outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

VIF_THRESHOLD = 5

N_ITER_SEARCH = 100
MODEL_LIST = ["SVR", "RF", "PLSR", "XGB"]

N_SPLITS = 5

VIs = [
    "ARVI", "CCCI", "CSI", "CVI", "Datt99",
    "DVI", "EVI", "GCI", "GNDVI", "MCARIOSAVI",
    "MCARI", "MSAVI", "MSRI", "MTCI", "NDRI",
    "NDVI", "NGRDI", "OSAVI", "RDVI", "RECI",
    "RGI", "RTVI", "SAVI", "SRI",
    "TCARIOSAVI", "TCARI", "TVI", "VARI",
    "EGVI", "LCI"
]

SFs = ["CC", "PH"]

SBs = ["blue", "green", "nir", "red edge", "red"]

TFs = [
    "Contrast",
    "Dissimilarity",
    "Correlation",
    "Energy",
    "Homogeneity",
    "Entropy"
]

# -----------------------------
# Hyperparameter grids
# -----------------------------
RF_PARAM_GRID = {
    'n_estimators': range(50, 700),
    'max_depth': [None] + list(range(1, 10)),
    'min_samples_split': list(range(2, 10)),
    'min_samples_leaf': list(range(1, 5)),
    'max_features': ['sqrt', 'log2', None]
}

XGB_PARAM_GRID = {
    'n_estimators': range(50, 700),
    'max_depth': range(3, 10),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [0, 0.5, 1.0],
    'reg_alpha': [0, 0.5, 1.0]
}

SVR_PARAM_GRID = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
    'epsilon': [0.001, 0.01, 0.1],
    'gamma': ['scale', 'auto']
}

PLSR_PARAM_GRID = {
    'n_components': None
}
