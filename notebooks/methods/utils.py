# methods/utils.py

import numpy as np
import torch

def to_native(obj):
    """
    Convert numpy / torch types to native Python types for JSON serialization.
    """
    # numpy types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    # torch tensors
    if isinstance(obj, torch.Tensor):
        return to_native(obj.detach().cpu().numpy())
    # lists and dicts
    if isinstance(obj, list):
        return [ to_native(v) for v in obj ]
    if isinstance(obj, dict):
        return { k: to_native(v) for k, v in obj.items() }
    return obj

def standardize_method_output(result: dict) -> dict:
    """
    Ensures that every method returns a dict of only JSON‑serializable native types,
    and that common fields (model_name, y_pred, y_prob, etc.) exist.
    """
    # Beispiel: setze Defaults, falls fehlen
    keys = {
        'model_name':         None,
        'iteration':          None,
        'y_pred':             [],
        'y_prob':             [],
        'selected_features':  [],
        'best_f1':            None,
        'best_threshold':     None,
    }
    # Füge alle Default‑Keys hinzu
    for k, default in keys.items():
        result.setdefault(k, default)
    
    # Add missing columns for plotting compatibility
    if 'coef_all' in result and 'selected_features' in result:
        coef_all = result['coef_all']
        selected_features = result['selected_features']
        
        # Create feature_names if missing
        if 'feature_names' not in result:
            # Try to infer from selected_features or create generic names
            if selected_features and isinstance(selected_features[0], str):
                # Extract feature indices from names like 'feature_0', 'feature_1', etc.
                max_idx = 0
                for feat in selected_features:
                    if '_' in feat:
                        try:
                            idx = int(feat.split('_')[-1])
                            max_idx = max(max_idx, idx)
                        except:
                            pass
                # Create feature names for all coefficients
                result['feature_names'] = [f'feature_{i}' for i in range(len(coef_all))]
            else:
                result['feature_names'] = [f'feature_{i}' for i in range(len(coef_all))]
        
        # Create selected_mask (binary mask of selected features)
        if 'selected_mask' not in result:
            feature_names = result['feature_names']
            selected_set = set(selected_features) if selected_features else set()
            result['selected_mask'] = [1 if name in selected_set else 0 for name in feature_names]
        
        # Create signs (signs of coefficients)
        if 'signs' not in result:
            result['signs'] = [0 if abs(c) <= 1e-8 else (1 if c > 0 else -1) for c in coef_all]
        
        # Create nnz (number of non-zero features)
        if 'nnz' not in result:
            result['nnz'] = sum(result['selected_mask'])
    
    # Konvertiere alle values
    return { k: to_native(v) for k, v in result.items() }
