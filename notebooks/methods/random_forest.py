import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils import standardize_method_output
except ImportError as e:
    print(f"Import error in random_forest.py: {e}")
    # Fallback: define a simple version
    def standardize_method_output(result):
        # Simple conversion to native types
        import numpy as np
        converted = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                converted[k] = v.tolist()
            elif isinstance(v, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                converted[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32, np.float16)):
                converted[k] = float(v)
            else:
                converted[k] = v
        return converted

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

    result = {
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
    
    return standardize_method_output(result)
