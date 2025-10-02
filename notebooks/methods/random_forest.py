import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
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
    # Start timing
    start_time = time.perf_counter()
    
    # Scaling wie bei Lasso (fairer Vergleich)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None

    # Simpler RF mit OOB
    rf = RandomForestClassifier(
        n_estimators=200,       # genug Bäume, aber nicht zu viel
        max_features="sqrt",    # Default-Setting für Klassifikation
        oob_score=True,         # Out-of-Bag zur Generalisierung
        random_state=randomState,
        n_jobs=-1,
        bootstrap=True
    )
    rf.fit(X_train_scaled, y_train)

    # Wahrscheinlichkeiten
    y_probs = rf.predict_proba(X_test_scaled)[:, 1]

    # Threshold: Val-Set optimieren oder fix 0.5
    if X_val is not None and y_val is not None:
        y_probs_val = rf.predict_proba(X_val_scaled)[:, 1]
        thresholds = np.linspace(0, 1, 501)
        f1_scores = [f1_score(y_val, (y_probs_val >= t).astype(int), zero_division=0) for t in thresholds]
        best_threshold = thresholds[int(np.argmax(f1_scores))]
    else:
        best_threshold = 0.5

    # Predictions
    y_pred = (y_probs >= best_threshold).astype(int)

    # Metriken
    f1 = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    # Training Performance
    y_train_probs = rf.predict_proba(X_train_scaled)[:, 1]
    y_train_pred = (y_train_probs >= best_threshold).astype(int)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Feature Importances
    importances = rf.feature_importances_
    importance_threshold = 0.01
    selected_features = [X_columns[i] for i, imp in enumerate(importances) if imp > importance_threshold] \
        if X_columns is not None else []

    # End timing
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    result = {
        "model_name": "RF",
        "iteration": iteration,
        "random_seed": randomState,
        "f1": f1,
        "accuracy": acc,
        "train_f1": train_f1,
        "train_accuracy": train_acc,
        "threshold": best_threshold,
        "y_pred": y_pred.tolist(),
        "y_prob": y_probs.tolist(),
        "feature_importances": importances.tolist(),
        "selected_features": selected_features,
        "method_has_selection": False,
        "n_selected": len(selected_features),
        "hyperparams": {
            "n_estimators": rf.n_estimators,
            "max_features": rf.max_features,
            "oob_score": True
        },
        "oob_score": rf.oob_score_,
        
        # Timing information
        "execution_time": execution_time,
        "timing": {
            "total_seconds": execution_time,
            "start_time": start_time,
            "end_time": end_time
        }
    }
    return standardize_method_output(result)
