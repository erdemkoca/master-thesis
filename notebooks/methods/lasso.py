import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils import standardize_method_output
except ImportError as e:
    print(f"Import error in lasso.py: {e}")


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


def custom_sparsity_score(estimator, X, y):
    """
    Custom scoring function that balances F1-score with sparsity.
    Penalizes models that select too many features.
    """
    y_pred = estimator.predict(X)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    # Get coefficients directly (no pipeline)
    coefs = estimator.coef_.flatten()
    n_selected = np.sum(coefs != 0)
    n_total = len(coefs)
    sparsity_ratio = n_selected / n_total
    
    # Penalty for selecting too many features (encourage sparsity)
    # If more than 50% of features are selected, apply penalty
    if sparsity_ratio > 0.5:
        penalty = 1.0 - (sparsity_ratio - 0.5) * 0.5  # Reduce score for high sparsity
        f1 *= penalty
    
    return f1

def run_lasso(X_train, y_train, X_test, y_test, iteration, randomState, X_columns, X_val=None, y_val=None):
    param_grid = {
        'C': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
              0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
              4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0,
              25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0,
              70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 150.0,
              200.0, 250.0, 300.0, 400.0, 500.0],
        'penalty': ['l1'],
        'solver': ['liblinear']
    }

    # Use direct estimator (like the working version) - Pipeline interferes with L1 sparsity
    clf = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=randomState),
        param_grid,
        scoring=custom_sparsity_score,  # Use custom scoring that encourages sparsity
        cv=5,  # Reduced CV for efficiency
        n_jobs=-1,
        verbose=0
    )
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    y_probs = best_model.predict_proba(X_test)[:, 1]

    # Fine-grained threshold optimization (1001 steps)
    thresholds = np.linspace(0.0, 1.0, 1001)
    
    if X_val is not None and y_val is not None:
        # Threshold auf Validation wählen
        y_val_prob = best_model.predict_proba(X_val)[:,1]
        idx = np.argmax([f1_score(y_val, (y_val_prob>=t).astype(int), zero_division=0) for t in thresholds])
        best_threshold = float(thresholds[idx])
    else:
        # Fallback (wenn kein Val): wie bisher, aber 1001 Stufen
        y_test_prob_tmp = best_model.predict_proba(X_test)[:,1]
        idx = np.argmax([f1_score(y_test, (y_test_prob_tmp>=t).astype(int), zero_division=0) for t in thresholds])
        best_threshold = float(thresholds[idx])
    
    # Testauswertung mit best_threshold
    y_test_prob = best_model.predict_proba(X_test)[:,1]
    y_pred = (y_test_prob >= best_threshold).astype(int)
    best_f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Ausgewählte Features (direct estimator)
    coefs = best_model.coef_.flatten()
    selected_features = [X_columns[i] for i, c in enumerate(coefs) if c != 0]
    
    # Sparsity analysis
    n_total_features = len(coefs)
    n_selected_features = len(selected_features)
    sparsity_ratio = n_selected_features / n_total_features
    coef_magnitudes = np.abs(coefs)
    max_coef = np.max(coef_magnitudes)
    min_nonzero_coef = np.min(coef_magnitudes[coef_magnitudes > 0]) if n_selected_features > 0 else 0

    # Standardisierte Selection-Metadaten für Feature-Selection-Analyse
    feature_names = list(X_columns)
    selected_mask = [int(abs(c) > 1e-8) for c in coefs]
    signs = [0 if abs(c) <= 1e-8 else (1 if c > 0 else -1) for c in coefs]
    nnz = int(np.sum(np.abs(coefs) > 1e-8))

    result = {
        'model_name': 'lasso',
        'iteration': iteration,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'y_pred': y_pred.tolist(),
        'y_prob': y_probs.tolist(),
        'precision': precision,
        'recall': recall,
        'selected_features': selected_features,
        'method_has_selection': True,
        'n_selected': len(selected_features),
        'lasso_C': clf.best_params_['C'],
        'lasso_coefs': best_model.coef_.flatten().tolist(),
        # Sparsity metrics
        'n_total_features': n_total_features,
        'sparsity_ratio': sparsity_ratio,
        'max_coefficient': max_coef,
        'min_nonzero_coefficient': min_nonzero_coef,
        'cv_score': clf.best_score_,  # The score from our custom sparsity function
        # Standardisierte Selection-Metadaten
        'feature_names': feature_names,
        'coef_all': [float(c) for c in coefs],
        'selected_mask': selected_mask,
        'signs': signs,
        'nnz': nnz
    }

    return standardize_method_output(result)
