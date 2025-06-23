### methods/run_lassonet.py
import numpy as np
from lassonet import LassoNetClassifierCV
from sklearn.metrics import f1_score


def run_lassonet(X_train, y_train, X_test, y_test, rng, iteration, randomState, X_columns=None):

    # LassoNet mit CV (bei Bedarf cv-Parameter anpassen)
    model = LassoNetClassifierCV(cv=5) #random_state=randomState,
    model.fit(X_train, y_train)

    # Vorhersage-Wahrscheinlichkeiten
    y_probs = model.predict_proba(X_test)[:, 1]

    # In NumPy-Array umwandeln, falls Torch-Tensor
    if hasattr(y_probs, "detach"):
        y_probs = y_probs.detach().cpu().numpy()
    else:
        y_probs = np.asarray(y_probs)

    # Threshold-Optimierung
    thresholds = np.linspace(0, 1, 100)
    f1_scores  = [f1_score(y_test, (y_probs >= t).astype(int)) for t in thresholds]
    best_idx    = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx]
    best_f1     = f1_scores[best_idx]
    y_pred      = (y_probs >= best_threshold).astype(int)

    # Feature-Auswahl-Indices aus LassoNet
    selected_indices = getattr(model, 'best_selected_', [])
    if X_columns:
        selected_features = [X_columns[i] for i in selected_indices]
    else:
        selected_features = list(selected_indices)

    return {
        'model_name':      'lassonet',
        'iteration':       iteration,
        'best_threshold':  best_threshold,
        'best_f1':         best_f1,
        'y_pred':          y_pred,
        'y_prob':          y_probs,
        'selected_features': selected_features,
        'best_lambda':     getattr(model, 'best_lambda_', None),
        'best_cv_score':   getattr(model, 'best_cv_score_', None)
    }