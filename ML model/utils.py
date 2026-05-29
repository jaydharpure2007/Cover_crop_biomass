# src/utils.py

import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
def preprocess_data(data_model):

    X = np.array(data_model.iloc[:, :-1])
    y = np.array(data_model.iloc[:, -1]).reshape(-1, 1)

    return X, y

def calculate_vif(X):
    """Calculate VIF for each feature."""
    X = sm.add_constant(X)

    vif_df = pd.DataFrame()

    vif_df["feature"] = X.columns[1:]

    vif_df["VIF"] = [
        variance_inflation_factor(X.values, i + 1)
        for i in range(len(X.columns) - 1)
    ]

    return vif_df

def vif_feature_selection(data, features, target, vif_threshold=5):
    # Prepare data
    data_vif = data[features + [target]]
    X_vif = data_vif[features].copy()
    # Iteratively remove high-VIF features
    while True:
        vif_df = calculate_vif(X_vif)
        max_vif = vif_df["VIF"].max()
        if max_vif > vif_threshold:
            drop_feature = (
                vif_df.sort_values("VIF", ascending=False)
                .iloc[0]["feature"]
            )
            X_vif.drop(columns=[drop_feature], inplace=True)
        else:
            break

    selected_features = list(X_vif.columns)
    return selected_features, vif_df
