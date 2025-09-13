import numpy as np
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
    # 3) Threshold from validation (fallback 0.5)
    # ---------------------------
    if X_va is not None and y_val is not None:
        p_va = clf.predict_proba(X_va)[:, 1]
        qs = np.linspace(0.0, 1.0, 101)
        ths = np.unique(np.quantile(p_va, qs))
        f1s = [f1_score(y_val, (p_va >= t).astype(int), zero_division=0) for t in ths]
        best_thr = float(ths[int(np.argmax(f1s))])
    else:
        best_thr = 0.5

    # ---------------------------
    # 4) Test predictions/metrics
    # ---------------------------
    p_te = clf.predict_proba(X_te)[:, 1]
    y_pred = (p_te >= best_thr).astype(int)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))

    # ---------------------------
    # 5) Coefficients: std → raw
    # ---------------------------
    intercept_std = float(clf.intercept_[0])
    beta_std = clf.coef_.ravel().astype(float)

    # raw-space mapping
    beta_raw = beta_std / s_safe
    intercept_raw = intercept_std - np.sum(beta_std * (mu / s_safe))

    # optional reporting threshold in RAW space
    beta_raw_report = beta_raw.copy()
    if tau_report > 0.0:
        beta_raw_report[np.abs(beta_raw_report) < tau_report] = 0.0

    # selection (based on raw-space, tiny eps)
    eps = 1e-12
    sel_mask = (np.abs(beta_raw_report) > eps).astype(int).tolist()
    selected = [name for name, c in zip(X_columns, beta_raw_report) if abs(c) > eps]

    return {
        "model_name": "lasso",
        "iteration": iteration,
        "random_seed": randomState,

        # flat metrics for your existing plots
        "f1": f1,
        "accuracy": acc,
        "threshold": best_thr,

        # keep preds for persistence
        "y_prob": p_te.tolist(),
        "y_pred": y_pred.tolist(),

        # RAW-space coefficients (use these for comparison with ground truth)
        "coefficients": {
            "intercept": float(intercept_raw),
            "values": beta_raw_report.tolist(),           # thresholded view
            "values_no_threshold": beta_raw.tolist(),     # pristine raw β
            "feature_names": list(X_columns),
            "coef_threshold_applied": float(tau_report)
        },

        # Also provide STD-space (in case you want it)
        "coefficients_std": {
            "intercept": float(intercept_std),
            "values": beta_std.tolist(),
            "feature_names": list(X_columns)
        },

        # selection summary
        "n_selected": int(sum(sel_mask)),
        "selection": {
            "mask": sel_mask,
            "features": selected
        },

        "hyperparams": {
            "C": float(clf.C_[0]),
            "penalty": "l1",
            "solver": "liblinear",
            "standardized": True
        }
    }
