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
    # Konvertiere alle values
    return { k: to_native(v) for k, v in result.items() }
