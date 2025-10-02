import numpy as np
import json
import time
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

def run_lasso(
    X_train, y_train, X_test, y_test, iteration, randomState, X_columns,
    X_val=None, y_val=None,
    Cs_logspace=(1e-3, 1e2, 15),
    class_weight=None,
    max_iter=5000, tol=1e-4,
    tau_report=0.0     # post-hoc |beta| threshold for reporting (applied in RAW space)
):
    # Start timing
    start_time = time.perf_counter()
    
    # ---------------------------
    # 1) Standardize (fit on train only)
    # ---------------------------
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    X_va = scaler.transform(X_val) if X_val is not None else None
    mu = scaler.mean_
    s = scaler.scale_
    s_safe = np.where(s == 0.0, 1.0, s)  # guard (shouldn't happen with random Gaussian X)

    # ---------------------------
    # 2) CV L1-Logistic
    # ---------------------------
    Cs = np.logspace(np.log10(Cs_logspace[0]), np.log10(Cs_logspace[1]), Cs_logspace[2])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=randomState)

    clf = LogisticRegressionCV(
        penalty='l1', solver='liblinear',
        Cs=Cs, cv=cv, scoring='neg_log_loss',
        class_weight=class_weight,
        max_iter=max_iter, tol=tol, n_jobs=1, refit=True,
        random_state=randomState
    ).fit(X_tr, y_train)

    # ---------------------------
    # 3) Fixed threshold (deactivated threshold selection to avoid overfitting)
    # ---------------------------
    # Threshold selection often causes overfitting on small validation sets
    # Using fixed threshold 0.5 for consistency and stability
    best_thr = 0.5

    # ---------------------------
    # 4) Test predictions/metrics
    # ---------------------------
    p_te = clf.predict_proba(X_te)[:, 1]
    y_pred = (p_te >= best_thr).astype(int)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))
    
    # Training performance for overfitting analysis
    p_tr = clf.predict_proba(X_tr)[:, 1]
    y_train_pred = (p_tr >= best_thr).astype(int)
    train_f1 = float(f1_score(y_train, y_train_pred, zero_division=0))
    train_acc = float(accuracy_score(y_train, y_train_pred))

    # ---------------------------
    # 5) Coefficients: report standardized coefficients
    # ---------------------------
    intercept_std = float(clf.intercept_[0])
    beta_std = clf.coef_.ravel().astype(float)

    # optional: post-hoc sparsity threshold IN STANDARDIZED UNITS
    beta_for_sel = beta_std.copy()
    if tau_report > 0:
        beta_for_sel[np.abs(beta_for_sel) < tau_report] = 0.0

    # selection (based on standardized coefficients)
    eps = 1e-12
    sel_mask = (np.abs(beta_for_sel) > eps).astype(int).tolist()
    selected = [name for name, c in zip(X_columns, beta_for_sel) if abs(c) > eps]

    # End timing
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    return {
        "model_name": "Lasso",
        "iteration": iteration,
        "random_seed": randomState,

        # flat metrics for your existing plots
        "f1": f1,
        "accuracy": acc,
        "train_f1": train_f1,
        "train_accuracy": train_acc,
        "threshold": best_thr,

        # keep preds for persistence
        "y_prob": p_te.tolist(),
        "y_pred": y_pred.tolist(),

        # STANDARDIZED-space coefficients (primary output)
        "coefficients": json.dumps({
            "space": "standardized",
            "intercept": intercept_std,
            "values": beta_for_sel.tolist(),          # for selection
            "values_no_threshold": beta_std.tolist(), # raw (unthresholded) for plotting
            "feature_names": list(X_columns),
            "coef_threshold_applied": float(tau_report),
            "mean": mu.tolist(),
            "scale": s.tolist()
        }),

        # selection summary
        "n_selected": int(sum(sel_mask)),
        "selection": {
            "mask": sel_mask,
            "features": selected
        },

        "hyperparams": {
            "C": float(clf.C_[0]),
            "penalty": "l1",
            "solver": "liblinear"
        },
        
        # Timing information
        "execution_time": execution_time,
        "timing": {
            "total_seconds": execution_time,
            "start_time": start_time,
            "end_time": end_time
        }
    }
