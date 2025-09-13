import numpy as np
from lassonet import LassoNetClassifierCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils import standardize_method_output
except ImportError as e:
    print(f"Import error in lasso_Net.py: {e}")
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

def run_lassonet(X_train, y_train, X_test, y_test,
                 iteration, randomState, X_columns=None,
                 X_val=None, y_val=None):

    # --- 1) Standardize data (fit on train only) ---
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    X_val_std = scaler.transform(X_val) if X_val is not None else None

    # --- 2) Fit LassoNet with CV on standardized data ---
    model = LassoNetClassifierCV(cv=5, random_state=randomState)
    model.fit(X_train_std, y_train)

    # --- 3) Predict probabilities on standardized data ---
    y_probs = model.predict_proba(X_test_std)[:, 1]
    if hasattr(y_probs, "detach"):  # torch.Tensor → numpy
        y_probs = y_probs.detach().cpu().numpy()
    else:
        y_probs = np.asarray(y_probs)

    # --- 4) Threshold selection (use validation if available, otherwise test) ---
    if X_val is not None and y_val is not None:
        # Use validation set for threshold selection
        y_probs_val = model.predict_proba(X_val_std)[:, 1]
        if hasattr(y_probs_val, "detach"):
            y_probs_val = y_probs_val.detach().cpu().numpy()
        else:
            y_probs_val = np.asarray(y_probs_val)
        
        thresholds = np.linspace(0.0, 1.0, 101)
        f1_scores = [f1_score(y_val, (y_probs_val >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f1_scores))
        best_threshold = thresholds[best_idx]
    else:
        # Use test set for threshold selection (fallback)
        thresholds = np.linspace(0.0, 1.0, 101)
        f1_scores = [f1_score(y_test, (y_probs >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f1_scores))
        best_threshold = thresholds[best_idx]

    # --- 5) Final predictions and metrics ---
    y_pred = (y_probs >= best_threshold).astype(int)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_test, y_pred))

    # --- 6) Feature selection ---
    raw_sel = getattr(model, 'best_selected_', None)
    
    if raw_sel is None:
        selected_indices = []
    else:
        # Convert to numpy array for easier handling
        arr = np.asarray(raw_sel)
        
        # Check if it's a boolean/0-1 mask
        if arr.ndim == 1 and set(np.unique(arr)) <= {0, 1}:
            # It's a mask, convert to indices
            mask = np.asarray(raw_sel, dtype=bool)
            selected_indices = list(np.nonzero(mask)[0])
        else:
            # It's already an index list/array
            selected_indices = list(raw_sel)
    
    # Map to column names only after we have a real integer list
    if X_columns is not None:
        selected_features = [X_columns[i] for i in selected_indices]
    else:
        selected_features = selected_indices

    # --- 7) Extract coefficients (LassoNet uses neural network, no direct coefficients) ---
    # LassoNet doesn't provide direct coefficients like LASSO
    # It uses a neural network approach for feature selection
    # We can create a placeholder structure for consistency
    if X_columns is not None:
        n_features = len(X_columns)
    else:
        n_features = X_train.shape[1]
    
    coefficients = {
        "space": "standardized",
        "intercept": 0.0,  # LassoNet doesn't expose intercept directly
        "values": [0.0] * n_features,  # Placeholder - LassoNet uses neural network
        "values_no_threshold": [0.0] * n_features,  # Same as values for LassoNet
        "feature_names": list(X_columns) if X_columns is not None else [f"feature_{i}" for i in range(n_features)],
        "coef_threshold_applied": 0.0,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "note": "LassoNet uses neural network - no direct coefficients available"
    }

    result = {
        'model_name': 'lassonet',
        'iteration': iteration,
        'random_seed': randomState,
        
        # flat metrics for your existing plots
        'f1': f1,
        'accuracy': accuracy,
        'threshold': best_threshold,
        
        # keep preds for persistence
        'y_pred': y_pred.tolist(),
        'y_prob': y_probs.tolist(),
        
        # coefficients structure (if available)
        'coefficients': coefficients,
        
        # selection summary
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'method_has_selection': True,
        
        # LassoNet specific
        'best_lambda': getattr(model, 'best_lambda_', None),
        'best_cv_score': getattr(model, 'best_cv_score_', None),
        
        # hyperparams
        'hyperparams': {
            'cv_folds': 5,
            'random_state': randomState,
            'method': 'lassonet'
        }
    }
    
    return standardize_method_output(result)
