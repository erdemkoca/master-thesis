import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils import resample
from lassonet import LassoNetClassifierCV

def run_lassonet(X_train, y_train, X_test, y_test, seed, X_columns=None):
    # Subsample majority class
    X_minority = X_train[y_train == 1]
    y_minority = y_train[y_train == 1]
    X_majority = X_train[y_train == 0]
    y_majority = y_train[y_train == 0]

    X_majority_down, y_majority_down = resample(
        X_majority, y_majority,
        replace=False,
        n_samples=len(y_minority),
        random_state=seed
    )

    X_resampled = np.vstack([X_majority_down, X_minority])
    y_resampled = np.concatenate([y_majority_down, y_minority])

    # Train model with CV
    model = LassoNetClassifierCV()
    model.fit(X_resampled, y_resampled)

    # Convert X_test if needed
    if X_columns is not None:
        X_test = pd.DataFrame(X_test, columns=X_columns)

    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]

    # 🔥 Convert to NumPy if it's a tensor
    if hasattr(y_probs, "detach"):
        y_probs = y_probs.detach().cpu().numpy()
    else:
        y_probs = np.asarray(y_probs)

    # Threshold optimization
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_test, (y_probs >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    y_pred = (y_probs >= best_threshold).astype(int)

    # Feature selection
    selected_feature_indices = model.best_selected_
    selected_features = [X_columns[i] for i in selected_feature_indices] if X_columns else selected_feature_indices

    return {
        'model_name': 'lassonet',
        'seed': seed,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'y_pred': y_pred,
        'y_prob': y_probs,
        'selected_features': selected_features,
        'best_lambda': model.best_lambda_,
        'best_cv_score': model.best_cv_score_,
    }
