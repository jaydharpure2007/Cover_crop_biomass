"""
SHAP-based model explainability utilities.

This module computes feature importance scores using
SHAP (SHapley Additive Explanations) for trained
PyTorch ANN models.
"""

import shap
import numpy as np
import torch
import pandas as pd

def compute_shap(final_model, X_train, feature_names, device):
    """
    Compute mean absolute SHAP values for model features.

    Parameters
    ----------
    final_model : torch.nn.Module
        Trained ANN model.

    X_train : numpy.ndarray
        Training features used as background and
        evaluation data.

    feature_names : list
        Names of predictor variables.

    device : torch.device
        CPU or GPU device used for inference.

    Returns
    -------
    numpy.ndarray
        Mean absolute SHAP importance values for
        each feature.
    """

    # Set model to evaluation mode
    final_model.eval()

    # Wrapper function required by SHAP
    def model_predict(x):
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(device)
            return final_model(x_tensor).cpu().numpy()

    # Use a subset of training samples as background data
    explainer = shap.KernelExplainer(
        model_predict,
        X_train[:100]
    )

    # Compute SHAP values for all training samples
    shap_values = explainer.shap_values(X_train)

    # Calculate mean absolute SHAP importance
    shap_importance = np.abs(shap_values).mean(axis=0).flatten()

    return pd.DataFrame({
        "Feature": feature_names,
        "Mean SHAP": shap_importance
    })


    