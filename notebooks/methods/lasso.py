import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.utils import resample

def run_lasso(X_train, y_train, X_test, y_test, seed, X_columns):
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

    # GridSearchCV
    clf = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=seed),
        param_grid,
        scoring='f1',
        cv=15,
        n_jobs=-1,
        verbose=0
    )
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    X_test_df = pd.DataFrame(X_test, columns=X_columns)
    y_probs = best_model.predict_proba(X_test_df)[:, 1]

    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_test, (y_probs >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    y_pred = (y_probs >= best_threshold).astype(int)

    # Selected features
    coefs = best_model.coef_.flatten()
    selected_features = [X_columns[i] for i, c in enumerate(coefs) if c != 0]

    return {
        'model_name': 'lasso',
        'seed': seed,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'y_pred': y_pred,
        'y_prob': y_probs,
        'selected_features': selected_features,
        'lasso_C': clf.best_params_['C'],
        'lasso_coefs': best_model.coef_.flatten().tolist()
    }
