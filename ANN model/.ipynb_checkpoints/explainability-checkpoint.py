# src/explainability.py

import shap
import numpy as np
import torch

def compute_shap(final_model, X_train, feature_names, device):
    final_model.eval()

    def model_predict(x):
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(device)
            return final_model(x_tensor).cpu().numpy()
    explainer = shap.KernelExplainer(
        model_predict,
        X_train[:100]
    )
    shap_values = explainer.shap_values(X_train)
    shap_importance = np.abs(shap_values).mean(axis=0).flatten()
    return shap_importance
