# src/explainability.py

import shap
import numpy as np
import pandas as pd


def compute_shap(model, model_name, X_train, feature_names):

    if model_name in ["RF", "XGB"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

    elif model_name == "SVR":
        explainer = shap.KernelExplainer(
            model.predict,
            X_train[:100]
        )
        shap_values = explainer.shap_values(X_train)

    else:
        explainer = shap.LinearExplainer(
            model,
            X_train
        )
        shap_values = explainer.shap_values(X_train)

    shap_importance = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean SHAP Value": shap_importance
    })

    shap_df = shap_df.sort_values(
        by="Mean SHAP Value",
        ascending=False
    )

    return shap_df