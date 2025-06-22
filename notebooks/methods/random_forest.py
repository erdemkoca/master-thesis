import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def run_random_forest(X_train, y_train, X_test, y_test, seed, X_columns=None):

    # Hyperparameter tuning
    param_grid = {'n_estimators': [100, 200, 300]}
    model = GridSearchCV(RandomForestClassifier(random_state=seed),
                         param_grid,
                         scoring='f1',
                         cv=5,
                         n_jobs=-1)
    model.fit(X_train, y_train)

    best_model = model.best_estimator_
    y_probs = best_model.predict_proba(X_test)[:, 1]

    # Threshold optimization
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_test, (y_probs >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    y_pred = (y_probs >= best_threshold).astype(int)

    return {
        'model_name': 'RandomForest',
        'seed': seed,
        'best_params': model.best_params_,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'y_pred': y_pred,
        'y_prob': y_probs,
        'feature_importances': best_model.feature_importances_.tolist()
    }
