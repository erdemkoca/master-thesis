import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
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

def run_lasso(X_train, y_train, X_test, y_test, rng, iteration, randomState, X_columns):

    param_grid = {
        'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
          0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
          4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0,
          25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0,
          70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 150.0,
          200.0, 250.0, 300.0, 400.0, 500.0],
        'penalty': ['l1'],
        'solver': ['liblinear']
    }

    clf = GridSearchCV(
        LogisticRegression(max_iter=1000),#, random_state=randomState
        param_grid,
        scoring='f1',
        cv=15,
        n_jobs=-1,
        verbose=0
    )
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    X_test_df  = pd.DataFrame(X_test, columns=X_columns)
    y_probs    = best_model.predict_proba(X_test_df)[:, 1]

    # Schwellenwert-Optimierung
    thresholds = np.linspace(0, 1, 100)
    f1_scores  = [f1_score(y_test, (y_probs >= t).astype(int)) for t in thresholds]
    best_idx   = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx]
    best_f1    = f1_scores[best_idx]
    y_pred     = (y_probs >= best_threshold).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Ausgewählte Features
    coefs = best_model.coef_.flatten()
    selected_features = [X_columns[i] for i, c in enumerate(coefs) if c != 0]

    result = {
        'model_name': 'lasso',
        'iteration':   iteration,
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
        'lasso_coefs': best_model.coef_.flatten().tolist()
    }
    
    return standardize_method_output(result)
