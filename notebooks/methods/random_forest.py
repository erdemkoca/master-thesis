import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
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


def run_random_forest(X_train, y_train, X_test, y_test, iteration, randomState, X_columns=None, X_val=None, y_val=None):
    # Feature scaling (like Lasso for fair comparison)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    
    # Fast Hyperparameter-Tuning - minimal grid for speed
    param_grid = {
        'n_estimators': [200],  # Fixed number of trees
        'max_depth': [10, None],  # Only 2 depth options
        'min_samples_leaf': [2, 5],  # Only 2 leaf options
        'min_samples_split': [5],  # Fixed split requirement
        'max_features': ['sqrt'],  # Only one feature strategy
        'max_samples': [0.8]  # Fixed subsampling
    }
    
    grid = GridSearchCV(
        RandomForestClassifier(
            random_state=randomState,
            n_jobs=-1,
            oob_score=True,  # Enable out-of-bag scoring for generalization assessment
            bootstrap=True
        ),
        param_grid,
        scoring='f1',
        cv=3,  # Keep CV folds
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    y_probs = best_model.predict_proba(X_test_scaled)[:, 1]
    importances = best_model.feature_importances_

    # Threshold optimization - STRICTLY use validation set only, never test set
    if X_val is not None and y_val is not None:
        # Use validation set for threshold optimization (CORRECT approach)
        y_probs_val = best_model.predict_proba(X_val_scaled)[:, 1]
        thresholds = np.linspace(0.000, 1.000, 1001)
        f1_scores = [f1_score(y_val, (y_probs_val >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f1_scores))
        best_threshold = thresholds[best_idx]
        # Apply chosen threshold to test set
        y_pred = (y_probs >= best_threshold).astype(int)
        best_f1 = f1_score(y_test, y_pred, zero_division=0)
    else:
        # NO FALLBACK - use fixed threshold to avoid data leakage
        # This ensures fair comparison and prevents overfitting to test set
        best_threshold = 0.5  # Fixed threshold
        y_pred = (y_probs >= best_threshold).astype(int)
        best_f1 = f1_score(y_test, y_pred, zero_division=0)

    # Feature Importances - use threshold-based selection with balanced threshold
    importances = best_model.feature_importances_
    importance_threshold = 0.01  # Balanced threshold to capture important features
    selected_features = [X_columns[i] for i, imp in enumerate(importances) if imp > importance_threshold] \
        if X_columns is not None else []

    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate training performance for overfitting analysis
    y_train_probs = best_model.predict_proba(X_train_scaled)[:, 1]
    y_train_pred = (y_train_probs >= best_threshold).astype(int)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Get OOB score for additional generalization assessment
    oob_score = best_model.oob_score_ if hasattr(best_model, 'oob_score_') else None
    
    result = {
        'model_name': 'random_forest',
        'iteration': iteration,
        'random_seed': randomState,
        'f1': best_f1,
        'accuracy': accuracy,
        'train_f1': train_f1,
        'train_accuracy': train_accuracy,
        'threshold': best_threshold,
        'y_pred': y_pred.tolist(),
        'y_prob': y_probs.tolist(),
        'feature_importances': importances.tolist(),
        'selected_features': selected_features,
        'method_has_selection': False,
        'n_selected': len(selected_features),
        'hyperparams': grid.best_params_,
        'oob_score': oob_score  # Add OOB score for generalization assessment
    }

    return standardize_method_output(result)
