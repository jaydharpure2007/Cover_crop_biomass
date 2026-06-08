"""
Utility functions for ANN-based biomass modeling workflow.

This module includes:
    - Reproducibility control (random seed setting)
    - Data preprocessing utilities
    - PyTorch DataLoader preparation
    - Variance Inflation Factor (VIF) computation
    - Feature selection based on multicollinearity
"""

import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def set_seed(seed=42):
    """
    Set random seed for full reproducibility across libraries.

    Ensures deterministic behavior for:
        - Python random
        - NumPy
        - PyTorch (CPU and GPU)
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_data(data_model):
    """
    Split dataset into features (X) and target (y).

    Assumes last column is the target variable.
    """
    X = np.array(data_model.iloc[:, :-1])
    y = np.array(data_model.iloc[:, -1]).reshape(-1, 1)

    return X, y

def prepare_dataloader(X, y, batch_size, seed=42):
    """
    Create PyTorch DataLoader for training and validation.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix

    y : numpy.ndarray
        Target variable

    batch_size : int
        Batch size for training

    seed : int
        Random seed for shuffling
    """
    g = torch.Generator()
    g.manual_seed(seed)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(X_tensor, y_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )

    return dataloader
    
def calculate_vif(X):
    """
    Compute Variance Inflation Factor (VIF)
    for each feature to detect multicollinearity.
    """
    
    X = sm.add_constant(X)

    vif_df = pd.DataFrame()

    vif_df["feature"] = X.columns[1:]

    vif_df["VIF"] = [
        variance_inflation_factor(X.values, i + 1)
        for i in range(len(X.columns) - 1)
    ]

    return vif_df

def vif_feature_selection(data, features, target, vif_threshold=5):
    """
    Iteratively remove features with high multicollinearity
    using Variance Inflation Factor (VIF).
    """
    
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
