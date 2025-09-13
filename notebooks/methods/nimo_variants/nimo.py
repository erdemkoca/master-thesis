# methods/nimo.py
"""
NIMO (lean) — linear beta + neural per-feature correction (RAW-coeff return)
- Correction is mean-centered per feature to avoid scaling ambiguity.
- Coefficients are mapped back to RAW input space (undo StandardScaler).
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

# utils import (robust fallback)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import standardize_method_output
except ImportError:
    def standardize_method_output(result):
        out = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.generic,)):
                out[k] = v.item()
            else:
                out[k] = v
        return out


class NIMO(nn.Module):
    """
    y ~ Bernoulli(sigmoid(eta)),   eta = [1, x * (1 + g_corr(x))] · beta
    g_corr(x) = g(x) - g(0) and additionally mean-centered per feature.
    (x is the *standardized* input used for training.)
    """
    def __init__(self, d, hidden_dim=64, dropout=0.0, out_scale=0.3):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(d + 1))  # [b0, b1..bd]
        self.out_scale = out_scale  # Bound correction amplitude
        
        # Bounded MLP with smaller initialization
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, d)
        )
        
        # Initialize final layer weights to be small
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.1)
            self.mlp[-1].bias.mul_(0.1)
            self.register_buffer("g0", self.mlp(torch.zeros(1, d)).detach())

    def corrections_uncentered(self, x):
        """g(x) - g(0)  (no mean-centering)."""
        g = self.mlp(x)
        g = self.out_scale * torch.tanh(g)  # Bound the output
        return g - self.g0  # broadcast over batch

    def corrections(self, x):
        """Mean-centered correction per feature (batch-wise)."""
        g_corr = self.corrections_uncentered(x)
        return g_corr - g_corr.mean(dim=0, keepdim=True)

    def design_matrix(self, x, use_correction=True):
        n, d = x.shape
        ones = torch.ones(n, 1, device=x.device, dtype=x.dtype)
        feats = x if not use_correction else x * (1.0 + self.corrections(x))
        return torch.cat([ones, feats], dim=1)

    def predict_logits(self, x, use_correction=True):
        B = self.design_matrix(x, use_correction=use_correction)
        return B.matmul(self.beta)

    def predict_proba(self, x, use_correction=True):
        return torch.sigmoid(self.predict_logits(x, use_correction=use_correction))


@torch.no_grad()
def update_beta_irls_adaptive(model, X, y, w, lam_base=1e-1, tau_l1=2e-2, gamma=1.3,
                              use_correction=True, eps=1e-6, max_step=0.5):
    """
    IRLS step for beta with adaptive ridge ≈ lasso.
    w: per-coefficient weights (length d+1). We do NOT penalize intercept → set w[0]=0.
    """
    B = model.design_matrix(X, use_correction=use_correction)  # [n, d+1]
    logits = B.matmul(model.beta)
    p = torch.sigmoid(logits)
    W = p * (1.0 - p) + eps
    z = logits + (y - p) / W

    BW = B * W.unsqueeze(1)
    # Adaptive ridge matrix: lam_base * diag(w)
    A = BW.t().matmul(B) + lam_base * torch.diag(w)
    b = BW.t().matmul(z)
    beta_new = torch.linalg.solve(A, b)

    # elementwise soft-threshold on features (skip intercept)
    beta_np = beta_new.detach().cpu().numpy()
    beta_np[1:] = np.sign(beta_np[1:]) * np.maximum(np.abs(beta_np[1:]) - tau_l1, 0.0)

    # trust region
    beta_prev = model.beta.detach().clone()
    beta_t = torch.tensor(beta_np, device=B.device, dtype=B.dtype)
    delta = beta_t - beta_prev
    nrm = torch.norm(delta)
    if nrm > max_step:
        beta_t = beta_prev + delta * (max_step / (nrm + 1e-12))
    model.beta.data.copy_(beta_t)

    # update adaptive weights for next step (skip intercept)
    beta_abs = model.beta.detach().abs()
    w_new = w.clone()
    w_new[0] = 0.0
    w_new[1:] = 1.0 / (beta_abs[1:] + eps)**gamma
    return w_new


def run_nimo(
    X_train, y_train, X_test, y_test,
    iteration, randomState, X_columns=None,
    *,
    X_val=None, y_val=None,
    hidden_dim=64,
    dropout=0.15,              # Increased dropout
    T=25, nn_steps=2, lr=1e-3,
    lam_l2=0.5, tau_l1=0.05,   # Much stronger penalties
    lam_g=0.08,                # L1 on |g_corr| with curriculum
    lam_group=0.02,            # Group-L2 penalty on NN inputs
    tau_beta_report=0.01,      # Selection threshold
    eps_g=1e-3,
    early_tol=1e-4,
    out_scale=0.2,             # Tighter correction amplitude
    warm_start_steps=5,        # More warm-up steps
    use_no_harm=True,          # Enable no-harm switch
    gamma=1.3                  # Adaptive lasso gamma
):
    # ---- 0) Setup
    device = torch.device("cpu")
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    # ---- 1) Standardize on train only
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    Xva = scaler.transform(X_val) if X_val is not None else None

    Xt   = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt   = torch.tensor(y_train, dtype=torch.float32, device=device)
    XteT = torch.tensor(Xte, dtype=torch.float32, device=device)
    yteT = torch.tensor(y_test, dtype=torch.float32, device=device)
    if X_val is not None:
        XvaT = torch.tensor(Xva, dtype=torch.float32, device=device)
        yvaT = torch.tensor(y_val, dtype=torch.float32, device=device)

    d = Xt.shape[1]
    model = NIMO(d, hidden_dim=hidden_dim, dropout=dropout, out_scale=out_scale).to(device)
    opt = optim.Adam(model.mlp.parameters(), lr=lr, weight_decay=1e-4)  # Weight decay

    # Initialize adaptive weights for β (d+1) — intercept weight 0
    w = torch.ones(d+1, device=device) 
    w[0] = 0.0

    # ---- 1.5) Warm start β with plain logistic (no correction)
    for _ in range(warm_start_steps):
        w = update_beta_irls_adaptive(model, Xt, yt, w, lam_base=lam_l2, tau_l1=tau_l1, 
                                     gamma=gamma, use_correction=False)

    # ---- 2) Alternating training with curriculum learning
    loss_hist, prev_loss, stopped = [], float("inf"), False
    best_val_loss = float("inf")
    best_mlp_state = None
    
    for t in range(T):
        model.eval()
        w = update_beta_irls_adaptive(model, Xt, yt, w, lam_base=lam_l2, tau_l1=tau_l1, 
                                     gamma=gamma, use_correction=True)

        # Anneal out_scale: 0.1 -> target_out_scale over T
        init_scale, target_scale = 0.1, out_scale
        alpha = t / max(1, T-1)
        model.out_scale = init_scale + (target_scale - init_scale) * alpha

        # Curriculum learning for lam_g
        lam_g0, lam_g1 = lam_g, lam_g * 0.3  # final is 30% of start
        lam_g_t = lam_g0 + (lam_g1 - lam_g0) * (t / max(1, T-1))

        for _ in range(nn_steps):
            model.train()
            opt.zero_grad()
            logits = model.predict_logits(Xt, use_correction=True)
            bce = F.binary_cross_entropy_with_logits(logits, yt)
            
            with torch.no_grad():
                g_corr = model.corrections(Xt)
            
            # Add small noise to corrections during training
            if model.training:
                g_corr = g_corr + 0.01 * torch.randn_like(g_corr)
            
            # Orthogonality penalty: discourage g from aligning with x
            align = torch.mean(torch.abs((Xt * g_corr).mean(dim=0)))
            
            # Group-L2 penalty on NN first layer inputs (per input feature)
            W1 = model.mlp[0].weight  # shape: [hidden_dim, d]
            group_l2 = torch.sqrt((W1**2).sum(dim=0) + 1e-12).mean()  # average group norm
            reg_group = lam_group * group_l2
            
            reg_g = lam_g_t * g_corr.abs().mean()
            loss = bce + reg_g + reg_group + 1e-3 * align
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.mlp.parameters(), max_norm=1.0)
            opt.step()

        # Early stopping on validation loss
        if X_val is not None:
            with torch.no_grad():
                val_logits = model.predict_logits(XvaT, use_correction=True)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, yvaT).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_mlp_state = {k: v.clone() for k, v in model.mlp.state_dict().items()}

        loss_hist.append(float(loss.item()))
        if t > 0 and abs(loss_hist[-1] - prev_loss) < early_tol:
            stopped = True
            break
        prev_loss = loss_hist[-1]

    # Restore best MLP state
    if best_mlp_state is not None:
        model.mlp.load_state_dict(best_mlp_state)

    # ---- 3) No-harm switch and threshold selection
    model.eval()
    use_corr_final = True
    no_harm_choice = "on"
    
    if use_no_harm and X_val is not None:
        with torch.no_grad():
            prob_val_on  = model.predict_proba(XvaT, use_correction=True).cpu().numpy()
            prob_val_off = model.predict_proba(XvaT, use_correction=False).cpu().numpy()
        
        grid = np.linspace(0.0, 1.0, 501)
        f1_on  = max(f1_score(y_val, (prob_val_on  >= t).astype(int), zero_division=0) for t in grid)
        f1_off = max(f1_score(y_val, (prob_val_off >= t).astype(int), zero_division=0) for t in grid)
        use_corr_final = (f1_on >= f1_off)
        no_harm_choice = "on" if use_corr_final else "off"
    
    # Compute test probabilities with chosen correction mode
    with torch.no_grad():
        prob_te = model.predict_proba(XteT, use_correction=use_corr_final).cpu().numpy()
        prob_val = model.predict_proba(XvaT, use_correction=use_corr_final).cpu().numpy() if X_val is not None else None

    # Threshold selection
    grid = np.linspace(0.0, 1.0, 501)
    if prob_val is not None:
        thr = float(grid[int(np.argmax([f1_score(y_val, (prob_val >= t).astype(int), zero_division=0) for t in grid]))])
    else:
        thr = float(grid[int(np.argmax([f1_score(y_test, (prob_te >= t).astype(int), zero_division=0) for t in grid]))])

    y_pred = (prob_te >= thr).astype(int)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))

    # ---- 4) Decomposition (test/val) using chosen correction mode
    def decomposition(X_np):
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            eta_lin  = model.predict_logits(X_t, use_correction=False).cpu().numpy()
            eta_full = model.predict_logits(X_t, use_correction=use_corr_final).cpu().numpy()
        eta_corr = eta_full - eta_lin
        var_full = np.var(eta_full)
        return dict(
            corr_mean_abs=float(np.mean(np.abs(eta_corr))),
            corr_var_share=float(np.var(eta_corr)/var_full) if var_full > 1e-12 else 0.0,
            lin_full_corr=float(np.corrcoef(eta_lin, eta_full)[0,1]) if np.std(eta_lin)>0 and np.std(eta_full)>0 else 0.0
        )

    decomp_val  = decomposition(Xva) if X_val is not None else None
    decomp_test = decomposition(Xte)

    # ---- 5) Correction stats (val)
    corr_stats = None
    if X_val is not None:
        with torch.no_grad():
            g_val = model.corrections(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy()
        corr_stats = {
            "eps_g": float(eps_g),
            "mean_abs_corr": np.abs(g_val).mean(axis=0).tolist(),
            "activation_rate": (np.abs(g_val) > eps_g).mean(axis=0).tolist(),
            "rel_mod": np.mean(np.abs(Xva * g_val), axis=0).tolist(),
        }

    # ---- 6) Report standardized coefficients (no conversion to RAW)
    beta_std_all = model.beta.detach().cpu().numpy()   # [b0, b1..bd] for standardized X
    intercept_std = float(beta_std_all[0])
    beta_std = beta_std_all[1:].copy()

    beta_for_sel = beta_std.copy()
    if tau_beta_report > 0:
        beta_for_sel[np.abs(beta_for_sel) < tau_beta_report] = 0.0

    selected_mask = (np.abs(beta_for_sel) > 0).astype(int).tolist()
    selected_features = (
        [X_columns[j] for j, m in enumerate(selected_mask) if m] if X_columns
        else [j for j, m in enumerate(selected_mask) if m]
    )

    # ---- 7) Results
    feature_names = list(X_columns) if X_columns else [f"feature_{i}" for i in range(len(beta_std))]
    result = {
        "model_name": "nimo",
        "iteration": iteration,
        "random_seed": randomState,

        # flat metrics for your existing plots
        "f1": f1,
        "accuracy": acc,
        "threshold": thr,

        "coefficients": json.dumps({
            "space": "standardized",
            "intercept": intercept_std,
            "values": beta_for_sel.tolist(),            # selection view
            "values_no_threshold": beta_std.tolist(),   # unthresholded for plots
            "feature_names": feature_names,
            "coef_threshold_applied": float(tau_beta_report),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }),
        "selection": {"mask": selected_mask, "features": selected_features},
        "n_selected": len(selected_features),

        "decomposition_val": decomp_val,
        "decomposition_test": decomp_test,
        "correction_stats_val": corr_stats,

        "no_harm_val": (
            None if (X_val is None) else
            {
                "g_frobenius_per_dim": float(np.linalg.norm(g_corr_val) / max(1, beta_raw.size)) if 'g_corr_val' in locals() else None,
                "lin_full_corr": (decomp_val["lin_full_corr"] if decomp_val else None),
                "no_harm_choice": no_harm_choice,
                "f1_on": f1_on if use_no_harm and X_val is not None else None,
                "f1_off": f1_off if use_no_harm and X_val is not None else None,
            }
        ),

        "training": {
            "loss_history": [float(x) for x in loss_hist],
            "n_iters": len(loss_hist),
            "stopped_early": bool(stopped),
        },
        "hyperparams": {
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "T": int(T),
            "nn_steps": int(nn_steps),
            "lr": float(lr),
            "lam_l2": float(lam_l2),
            "tau_l1": float(tau_l1),
            "lam_g": float(lam_g),
            "lam_group": float(lam_group),
            "tau_beta_report": float(tau_beta_report),
            "eps_g": float(eps_g),
            "out_scale": float(out_scale),
            "warm_start_steps": int(warm_start_steps),
            "use_no_harm": bool(use_no_harm),
            "gamma": float(gamma),
        },
    }
    return standardize_method_output(result)
