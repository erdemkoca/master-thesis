import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

def run_random_forest(X_train, y_train, X_test, y_test, rng, iteration, randomState, X_columns=None):
    # Hyperparameter-Tuning
    param_grid = {'n_estimators': [100, 200, 300]}
    grid = GridSearchCV(
        RandomForestClassifier(), #random_state=randomState
        param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_probs    = best_model.predict_proba(X_test)[:, 1]
    importances = best_model.feature_importances_

    # Threshold-Optimierung
    thresholds = np.linspace(0, 1, 100)
    f1_scores  = [f1_score(y_test, (y_probs >= t).astype(int)) for t in thresholds]
    best_idx   = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx]
    best_f1    = f1_scores[best_idx]
    y_pred     = (y_probs >= best_threshold).astype(int)

    # Feature Importances - use threshold-based selection
    importances = best_model.feature_importances_
    importance_threshold = 0.01
    selected_features = [X_columns[i] for i, imp in enumerate(importances) if imp > importance_threshold] \
        if X_columns is not None else []

    return {
        'model_name':         'RandomForest',
        'iteration':          iteration,
        'best_params':        grid.best_params_,
        'best_threshold':     best_threshold,
        'best_f1':            best_f1,
        'y_pred':             y_pred.tolist(),
        'y_prob':             y_probs.tolist(),
        'feature_importances': importances.tolist(),
        'selected_features':  selected_features,
        'method_has_selection': False,
        'n_selected': len(selected_features)
    }
