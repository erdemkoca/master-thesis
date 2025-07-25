import numpy as np
from lassonet import LassoNetClassifierCV
from sklearn.metrics import f1_score
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
                 rng, iteration, randomState, X_columns=None):

    # --- 1) Fit LassoNet with CV ---
    model = LassoNetClassifierCV(cv=5)
    model.fit(X_train, y_train)

    # --- 2) Predict probabilities & pick best threshold by F1 ---
    y_probs = model.predict_proba(X_test)[:, 1]
    if hasattr(y_probs, "detach"):  # torch.Tensor → numpy
        y_probs = y_probs.detach().cpu().numpy()
    else:
        y_probs = np.asarray(y_probs)

    thresholds     = np.linspace(0, 1, 100)
    f1_scores      = [f1_score(y_test, (y_probs >= t).astype(int)) for t in thresholds]
    best_idx       = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx]
    best_f1        = f1_scores[best_idx]
    y_pred         = (y_probs >= best_threshold).astype(int)

    # --- 3) Pull out the selected feature indices ---
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

    result = {
        'model_name':          'lassonet',
        'iteration':           iteration,
        'best_threshold':      best_threshold,
        'best_f1':             best_f1,
        'y_pred':              y_pred.tolist(),
        'y_prob':              y_probs.tolist(),
        'selected_features':   selected_features,
        'method_has_selection': True,
        'n_selected':          len(selected_features),
        'best_lambda':         getattr(model, 'best_lambda_', None),
        'best_cv_score':       getattr(model, 'best_cv_score_', None)
    }
    
    return standardize_method_output(result)
