import numpy as np
import pandas as pd
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils import standardize_method_output
except ImportError as e:
    print(f"Import error in random_forest_fixed.py: {e}")
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

def calculate_hash(arr):
    """Calculate MD5 hash of array for debugging"""
    if isinstance(arr, np.ndarray):
        return hashlib.md5(arr.tobytes()).hexdigest()[:8]
    else:
        return hashlib.md5(str(arr).encode()).hexdigest()[:8]

def run_random_forest(X_train, y_train, X_test, y_test, rng, iteration, randomState, X_columns=None):
    """
    FIXED Random Forest with proper randomness and validation-based threshold optimization
    """
    # Split training data into train/val for threshold optimization
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=randomState)
    train_idx, val_idx = next(splitter.split(X_train, y_train))
    
    X_train_split = X_train[train_idx]
    y_train_split = y_train[train_idx]
    X_val_split = X_train[val_idx]
    y_val_split = y_train[val_idx]

    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            random_state=randomState,  # Use iteration-specific seed
            n_jobs=1  # Single job for debugging consistency
        ))
    ])

    # Extended hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__max_depth': [3, 5, 10, None],
        'classifier__max_features': ['sqrt', 'log2', None]
    }

    # Create custom CV split that uses our train/val split
    X_train_val = np.vstack([X_train_split, X_val_split])
    y_train_val = np.hstack([y_train_split, y_val_split])
    cv_splits = [(np.arange(len(X_train_split)), np.arange(len(X_train_split), len(X_train_val)))]

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='f1',
        cv=cv_splits,
        n_jobs=1,
        verbose=0
    )

    grid_search.fit(X_train_val, y_train_val)

    # Get best model and refit on training data only
    best_pipeline = grid_search.best_estimator_
    best_pipeline.fit(X_train_split, y_train_split)

    # Get validation predictions for threshold optimization
    y_val_probs = best_pipeline.predict_proba(X_val_split)[:, 1]

    # Fine-grained threshold optimization on validation set
    thresholds = np.linspace(0.000, 1.000, 1001)  # 0.001 step size
    f1_scores_val = []

    for t in thresholds:
        y_val_pred = (y_val_probs >= t).astype(int)
        f1_val = f1_score(y_val_split, y_val_pred, zero_division=0)
        f1_scores_val.append(f1_val)

    best_threshold_idx = np.argmax(f1_scores_val)
    best_threshold = thresholds[best_threshold_idx]
    best_f1_val = f1_scores_val[best_threshold_idx]

    # Apply fixed threshold to test set
    y_test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_probs >= best_threshold).astype(int)

    # Calculate test metrics
    f1_test = f1_score(y_test, y_test_pred, zero_division=0)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, zero_division=0)
    recall_test = recall_score(y_test, y_test_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    # Feature selection based on importance threshold
    importances = best_pipeline.named_steps['classifier'].feature_importances_
    importance_threshold = np.mean(importances) + np.std(importances)  # Dynamic threshold
    selected_features = [X_columns[i] for i, imp in enumerate(importances) if imp > importance_threshold] \
        if X_columns is not None else []

    # Calculate importance hash for debugging
    importance_hash = calculate_hash(importances)

    result = {
        'model_name': 'RandomForest',
        'iteration': iteration,
        'seed_iter': randomState,
        'best_f1': f1_test,  # Test F1 score
        'best_threshold': best_threshold,
        'f1_val': best_f1_val,  # Validation F1 score used for threshold selection
        'accuracy': accuracy_test,
        'precision': precision_test,
        'recall': recall_test,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'y_pred': y_test_pred.tolist(),
        'y_prob': y_test_probs.tolist(),
        'selected_features': selected_features,
        'method_has_selection': False,  # RF doesn't do explicit feature selection
        'n_selected': len(selected_features),
        'feature_importances': importances.tolist(),
        'importance_hash': importance_hash,
        'best_params': grid_search.best_params_,
        # Debug information
        'train_samples': len(X_train_split),
        'val_samples': len(X_val_split),
        'test_samples': len(X_test)
    }
    
    return standardize_method_output(result)
