"""
Training utilities for ANN-based biomass prediction.

This module includes:
    - Model training with early stopping
    - Model evaluation (RMSE, R²)
    - Optuna objective function for hyperparameter tuning
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from utils import prepare_dataloader
from model import ANNModel
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    device,
    max_epochs=1000,
    patience=10
):
    """
    Train ANN model using early stopping on validation loss.

    Stops training when validation loss does not improve
    for 'patience' consecutive epochs.
    """
    best_loss = float('inf')
    counter = 0
    best_model_state = None
    best_epoch = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            break
    model.load_state_dict(best_model_state)
    return model, best_epoch

def evaluate_model(model, X, y, scaler_y, device):
    """
    Evaluate trained model on dataset and return metrics in original scale.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_tensor).cpu().numpy()
    preds_inv = scaler_y.inverse_transform(preds)
    y_inv = scaler_y.inverse_transform(y)
    rmse = np.sqrt(mean_squared_error(y_inv, preds_inv))
    r2 = r2_score(y_inv, preds_inv)
    return rmse, r2, y_inv.flatten(), preds_inv.flatten()

def objective(trial, X_train, y_train, device, scaler_y, seed = 42):
    """
    Optuna objective function for ANN hyperparameter tuning.

    Uses K-Fold cross-validation and returns mean RMSE.
    """
    # Hyperparameter search space
    hidden_size = trial.suggest_int("hidden_size", 50, 300)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
    batch_size = trial.suggest_int("batch_size", 8, 64)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("patience", 10, 20)

    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses, r2s, best_epochs = [], [], []

    for train_idx, val_idx in kf.split(X_train):
        
        # Split data
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Load data
        train_loader = prepare_dataloader(X_tr, y_tr, batch_size, seed)
        val_loader = prepare_dataloader(X_val, y_val, batch_size, seed)

        # Initialize model
        model = ANNModel(X_train.shape[1], hidden_size, dropout_rate).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model, best_epoch = train_model(model, criterion, optimizer, train_loader, val_loader, device,  patience)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb.to(device)).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(yb.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Ensure no NaN values during training/evaluation
        if np.isnan(all_preds).any() or np.isnan(all_targets).any():
            raise ValueError("NaN detected in predictions or targets.")

        preds_inv = scaler_y.inverse_transform(all_preds)
        y_val_inv = scaler_y.inverse_transform(all_targets)

        rmse = np.sqrt(mean_squared_error(y_val_inv, preds_inv))
        r2 = r2_score(y_val_inv, preds_inv)
        rmses.append(rmse)
        r2s.append(r2)
        best_epochs.append(best_epoch)
    
    # Store Optuna metadata for analysis
    trial.set_user_attr("R2", np.mean(r2s))
    trial.set_user_attr("max_best_epoch", np.max(best_epochs))
    return np.mean(rmses)