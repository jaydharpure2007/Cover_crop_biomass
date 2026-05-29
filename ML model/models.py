# src/models.py
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor

def get_model(model_name, seed):
    if model_name == "RF":
        return RandomForestRegressor(
            random_state=seed,
            n_jobs=-1
        )

    elif model_name == "SVR":
        return SVR()

    elif model_name == "PLSR":
        return PLSRegression()

    elif model_name == "XGB":
        return XGBRegressor(
            random_state=seed,
            n_jobs=-1,
            verbosity=0
        )

    else:
        raise ValueError("Invalid model name")
