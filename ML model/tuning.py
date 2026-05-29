# src/tuning.py

from sklearn.model_selection import RandomizedSearchCV
def tune_model(model, param_grid, X_train, y_train, kf, n_iter, seed):
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=kf,
        random_state=seed,
        n_jobs=-1,
        verbose=2
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_