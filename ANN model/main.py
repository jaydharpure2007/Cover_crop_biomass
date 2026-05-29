import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import warnings
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import prepare_dataloader
from config import *
from utils import (
    set_seed,
    preprocess_data,
    prepare_dataloader,
    vif_feature_selection
)

from model import ANNModel

from train import (
    train_model,
    evaluate_model,
    objective
)

from explainability import compute_shap

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    set_seed(SEED)
    print(f"Running on: {DEVICE}")

    start_time = time.time()

    all_lists = [VIs, SFs, SBs, TFs]
    list_names = ["VIs", "SFs", "SBs", "TFs"]

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    Train = pd.read_excel("Data/AGB_train_test.xlsx", sheet_name="train")

    Test = pd.read_excel("Data/AGB_train_test.xlsx", sheet_name="test")

    # --------------------------------------------------
    # Feature combinations
    # --------------------------------------------------
    for r in range(1, 5):
        for idx_combo in combinations(range(4), r):
            selected_lists = [all_lists[i] for i in idx_combo]
            selected_names = [list_names[i] for i in idx_combo]
            combined_features = []
            for lst in selected_lists:
                combined_features.extend(lst)
            folder_name = "_".join(selected_names)
            print(f"\nProcessing: {folder_name}")
            output_dir = os.path.join(OUTPUT_PATH,folder_name)
            os.makedirs(output_dir, exist_ok=True)

            # --------------------------------------------------
            # VIF Feature Selection
            # --------------------------------------------------
            features_vif, vif_df = vif_feature_selection(
                data=Train,
                features=combined_features,
                target=TARGET,
                vif_threshold=VIF_THRESHOLD
            )
            vif_df = vif_df.sort_values(by="VIF", ascending=True)
            print("Remaining features:", features_vif)

            # --------------------------------------------------
            # Training data
            # --------------------------------------------------
            train_data = Train[features_vif + [TARGET]].copy()
            X_train, y_train = preprocess_data(train_data)

            scaler_x = StandardScaler()
            scaler_y = StandardScaler()

            X_train = scaler_x.fit_transform(X_train)
            y_train = scaler_y.fit_transform(y_train)
            X_train = X_train.astype(np.float32)

            # --------------------------------------------------
            # Testing data
            # --------------------------------------------------
            test_data = Test[features_vif + [TARGET]].copy()
            X_test, y_test = preprocess_data(test_data)
            X_test = scaler_x.transform(X_test)
            y_test = scaler_y.transform(y_test)
            X_test = X_test.astype(np.float32)

            # --------------------------------------------------
            # Optuna
            # --------------------------------------------------
            sampler = optuna.samplers.TPESampler(seed=SEED)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(lambda trial: objective(trial, X_train, y_train, DEVICE, scaler_y, seed = SEED), n_trials=N_TRIALS, catch=(ValueError, FloatingPointError))
            # --------------------------------------------------
            # Trial results
            # --------------------------------------------------
            trial_data = []
            for trial in study.trials:
                trial_data.append({
                    "Trial": trial.number,
                    "RMSE": trial.value,
                    "R2": trial.user_attrs.get("R2"),
                    "Best_Epoch": trial.user_attrs.get(
                        "max_best_epoch"
                    ),
                    **trial.params
                })

            trials_df = pd.DataFrame(trial_data)
            trials_df = (
                trials_df
                .dropna(subset=["R2"])
                .sort_values(
                    by="R2",
                    ascending=False
                )
            )
            best_params = trials_df.iloc[0]
            # --------------------------------------------------
            # Final ANN Model
            # --------------------------------------------------
            final_model = ANNModel(
                input_size=X_train.shape[1],
                hidden_size=int(best_params["hidden_size"]),
                dropout_rate=float(best_params["dropout_rate"])
                ).to(DEVICE)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(final_model.parameters(), lr=float(best_params["lr"]))
            batch_size = int(best_params["batch_size"])
            patience = int(best_params["patience"])
            max_epochs = int(best_params["Best_Epoch"])
            # --------------------------------------------------
            # Train-validation split
            # --------------------------------------------------
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
            train_loader = prepare_dataloader(X_tr, y_tr, batch_size=batch_size, seed = SEED)
            val_loader = prepare_dataloader(X_val, y_val, batch_size=batch_size, seed = SEED)
            # --------------------------------------------------
            # Train model
            # --------------------------------------------------
            model, best_epoch = train_model(
                final_model,
                criterion,
                optimizer,
                train_loader,
                val_loader,
                device=DEVICE,
                max_epochs=max_epochs,
                patience=patience
            )
            # --------------------------------------------------
            # Evaluate
            # --------------------------------------------------

            train_rmse, train_r2, y_train_obs, y_train_pred = evaluate_model(
                final_model,
                X_train,
                y_train,
                scaler_y,
                DEVICE
            )
            test_rmse, test_r2, y_test_obs, y_test_pred = evaluate_model(
                final_model,
                X_test,
                y_test,
                scaler_y,
                DEVICE
            )

            print(f"Train R²: {train_r2:.3f}")
            print(f"Test R² : {test_r2:.3f}")

            # --------------------------------------------------
            # SHAP
            # --------------------------------------------------
            shap_importance = compute_shap(
                final_model=final_model,
                X_train=X_train,
                feature_names=features_vif,
                device=DEVICE
            )
            shap_df = pd.DataFrame({
                "Feature": features_vif,
                "Mean SHAP": shap_importance
            })
            shap_df = shap_df.sort_values(
                by="Mean SHAP",
                ascending=False
            )
            # --------------------------------------------------
            # Performance table
            # --------------------------------------------------
            performance_df = pd.DataFrame([
                {"Dataset": "Training", "RMSE": train_rmse, "R²": train_r2},
                {"Dataset": "Cross-Validation", "RMSE": best_params["RMSE"], "R²": best_params["R2"]},
                {"Dataset": "Testing", "RMSE": test_rmse, "R²": test_r2}
            ])
            # --------------------------------------------------
            # Save outputs
            # --------------------------------------------------

            total_time = time.time() - start_time
            processing_time_df = pd.DataFrame([{"Processing Time model (seconds)": total_time}])
            train_result_df = pd.DataFrame({"Observed": y_train_obs, "Predicted": y_train_pred})
            test_result_df = pd.DataFrame({"Observed": y_test_obs, "Predicted": y_test_pred})
            best_hyperparams_df = pd.DataFrame([best_params])
            output_file = os.path.join(output_dir,  "ANN.xlsx")
            with pd.ExcelWriter(output_file) as writer:
                performance_df.to_excel(writer, sheet_name="Performance", index=False)
                train_result_df.to_excel(writer, sheet_name="Train results", index=False)
                test_result_df.to_excel(writer, sheet_name="Test results", index=False)
                best_hyperparams_df.to_excel(writer, sheet_name="Best Hyperparameters", index=False)
                shap_df.to_excel(writer, sheet_name="SHAP", index=False)
                trials_df.to_excel(writer, sheet_name="Trial", index=False)
                processing_time_df.to_excel(writer, sheet_name="Processing Time", index=False)
                vif_df.to_excel(writer, sheet_name="VIF", index=False)

            print("Processing completed with PyTorch + CUDA" if torch.cuda.is_available() else "Processing completed with CPU")
            print(performance_df)
            print(f"Saved: {output_file}")

if __name__ == "__main__":

    main()
