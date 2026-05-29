import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def cross_validate_model(regressor, X_train, y_train, kf, scaler_y):
    """
    Custom cross-validation with inverse scaling.
    Returns averaged RMSE, R2, MAE.
    """

    cv_scores_rmse, cv_scores_r2, cv_scores_mae = [], [], []

    for train_index, val_index in kf.split(X_train):

        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        regressor.fit(X_train_fold, y_train_fold)
        y_pred_fold = regressor.predict(X_val_fold)

        # inverse transform
        y_val_orig = scaler_y.inverse_transform(y_val_fold.reshape(-1, 1))
        y_pred_orig = scaler_y.inverse_transform(y_pred_fold.reshape(-1, 1))

        # metrics
        rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        r2 = r2_score(y_val_orig, y_pred_orig)
        mae = mean_absolute_error(y_val_orig, y_pred_orig)

        cv_scores_rmse.append(rmse)
        cv_scores_r2.append(r2)
        cv_scores_mae.append(mae)

    return {
        "rmse": np.mean(cv_scores_rmse),
        "r2": np.mean(cv_scores_r2),
        "mae": np.mean(cv_scores_mae)
    }

def evaluate_model(regressor, X_train, X_test, y_train, y_test, scaler_y):
    """
    Predicts and evaluates model on train and test sets in original scale.
    Returns a dictionary of metrics and predictions.
    """

    # -------------------------
    # Predictions (scaled space)
    # -------------------------
    train_pred_scaled = regressor.predict(X_train)
    test_pred_scaled = regressor.predict(X_test)

    # -------------------------
    # Inverse transform predictions
    # -------------------------
    train_pred = scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()

    test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()

    # -------------------------
    # True values (original scale)
    # -------------------------
    y_train_true = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()

    y_test_true = y_test.ravel()

    # -------------------------
    # Metrics - Train
    # -------------------------
    train_rmse = np.sqrt(mean_squared_error(y_train_true, train_pred))
    train_r2 = r2_score(y_train_true, train_pred)
    train_mae = mean_absolute_error(y_train_true, train_pred)

    # -------------------------
    # Metrics - Test
    # -------------------------
    test_rmse = np.sqrt(mean_squared_error(y_test_true, test_pred))
    test_r2 = r2_score(y_test_true, test_pred)
    test_mae = mean_absolute_error(y_test_true, test_pred)

    return {
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "train_mae": train_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "y_train_true": y_train_true,
        "y_test_true": y_test_true
    }