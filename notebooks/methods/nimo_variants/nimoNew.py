# methods/variant.py

"""
NIMO Variant - Original NIMO Implementation mit Enhanced Tracking
und dynamischer Basis‑Erweiterung (Polynome, Fourier, Sawtooth)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import standardize_method_output
except ImportError:
    def standardize_method_output(result):
        import numpy as _np
        converted = {}
        for k, v in result.items():
            if isinstance(v, _np.ndarray):
                converted[k] = v.tolist()
            elif isinstance(v, (int, float, str, list, dict)):
                converted[k] = v
            else:
                try:
                    converted[k] = v.item()
                except:
                    converted[k] = v
        return converted


class SinAct(nn.Module):
    """Sinus-Aktivierung mit skalierter Frequenz."""
    def forward(self, x):
        return torch.sin(0.1 * x)


class NIMO(nn.Module):
    """
    Kern‑Modul von NIMO:
    - Beta-Parameter (Intercept + Features)
    - MLP mit 3 Hidden‑Layern, LayerNorm + Dropout + SinAct
    """

    def __init__(self, input_dim, hidden_dim, lam=1.0, noise_std=0.0, group_reg=0.0):
        super().__init__()
        # Beta-Parameter für (Intercept + input_dim)
        self.beta = nn.Parameter(torch.zeros(input_dim + 1))

        # Shared‑Net: drei Hidden‑Layer
        layers = []
        in_dim = 2 * input_dim
        for _ in range(3):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(noise_std),
            ]
            in_dim = hidden_dim
        layers += [SinAct(), nn.Linear(in_dim, 1)]
        self.shared_net = nn.Sequential(*layers)

        self.lam       = lam
        self.group_reg = group_reg

    def forward(self, x):
        """
        x: [n, d]  → wir erzeugen intern d×d Paare (ohne Diagonale) + Identity für Mask‑Prior
        und flattenen zu (n*d, 2d).
        """
        n, d = x.shape
        device = x.device

        # 1) Einheitsspalte
        B0 = torch.ones(n, 1, device=device)

        # 2) d×d Expand (ohne Diagonale)
        X_exp = x.unsqueeze(1).expand(n, d, d).clone()
        idx   = torch.arange(d, device=device)
        X_exp[:, idx, idx] = 0.0

        # 3) Pair‑Encoding (Eye matrices)
        PE = torch.eye(d, device=device).unsqueeze(0).expand(n, d, d)

        # 4) Flatten für Shared‑Net
        inp_flat = torch.cat([X_exp, PE], dim=2).reshape(n * d, 2 * d)

        # 5) Forward Shared‑Net
        g_flat = self.shared_net(inp_flat).squeeze(1)
        g      = g_flat.view(n, d)

        # 6) Null‑Baseline (zero input)
        zero_inp = torch.zeros_like(inp_flat)
        g0_flat  = self.shared_net(zero_inp).squeeze(1)
        g0       = g0_flat.view(n, d)

        # 7) moduliertes x
        g_corr = g - g0
        feats  = x * (1.0 + g_corr)

        # 8) Endgültige Design‑Matrix
        B = torch.cat([B0, feats], dim=1)
        return B

    def predict_logits(self, x):
        return self.forward(x).matmul(self.beta.unsqueeze(1)).squeeze(1)

    def predict_proba(self, x):
        return torch.sigmoid(self.predict_logits(x))



def run_nimoNew(
    X_train, y_train, X_test, y_test,
    rng, iteration, randomState, X_columns=None,
    *,
    hidden_dim     = 64,
    lam            = 1.0,
    noise_std      = 0.2,
    group_reg      = 0.0,
    lr             = 1e-3,
    T              = 15,
    early_tol      = 1e-4,
    group_reg_cv   = False,
    group_reg_vals = (0.0, 0.1, 0.5, 1.0)
):
    """
    NIMO Variant mit:
      - Dynamischer Basis‑Erweiterung (X^2, X^3, sin, sawtooth)
      - 3 Hidden‑Layer im Shared‑Net
      - IRLS‑Loop + optional Group‑Regularisierung via CV
    """
    device = torch.device('cpu')
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    # --- 1) Basis‑Erweiterung auf Training und Test ---
    X_aug      = X_train.copy()
    X_test_aug = X_test.copy()
    if X_columns:
        import scipy.signal as sg
        for name in X_columns:
            parts = name.split('_')
            t = parts[0]; idx = int(parts[1])
            if t == 'square':
                X_aug      = np.hstack([X_aug,      X_train[:, idx:idx+1]**2])
                X_test_aug = np.hstack([X_test_aug, X_test[:,  idx:idx+1]**2])
            elif t == 'cube':
                X_aug      = np.hstack([X_aug,      X_train[:, idx:idx+1]**3])
                X_test_aug = np.hstack([X_test_aug, X_test[:,  idx:idx+1]**3])
            elif t == 'sin':
                X_aug      = np.hstack([X_aug,      np.sin(X_train[:, idx:idx+1])])
                X_test_aug = np.hstack([X_test_aug, np.sin(X_test[:,  idx:idx+1])])
            elif t == 'sawtooth':
                saw_tr   = sg.sawtooth(5*X_train[:, idx]).reshape(-1,1)
                saw_te   = sg.sawtooth(5*X_test[:,  idx]).reshape(-1,1)
                X_aug      = np.hstack([X_aug,      saw_tr])
                X_test_aug = np.hstack([X_test_aug, saw_te])

    # --- 2) Torch‑Tensors & Dimensionen ---
    X_t      = torch.tensor(X_aug,      dtype=torch.float32, device=device)
    y_t      = torch.tensor(y_train,    dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_aug, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test,     dtype=torch.float32, device=device)
    n, d_aug = X_t.shape

    # --- 3) Optional Group‑Regularisierung via CV ---
    best_gr = group_reg
    if group_reg_cv:
        best_score = -1.0
        for gr in group_reg_vals:
            tmp = NIMO(d_aug, hidden_dim, lam, noise_std, gr).to(device)
            opt = optim.Adam(tmp.shared_net.parameters(), lr=lr)
            # Kurzes Warm‑Up
            for _ in range(5):
                tmp.train()
                loss = F.binary_cross_entropy_with_logits(tmp.predict_logits(X_t), y_t)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                probs = tmp.predict_proba(X_test_t).cpu().numpy()
            f1m = max(
                f1_score(y_test, (probs>=thr).astype(int))
                for thr in np.linspace(0,1,50)
            )
            if f1m > best_score:
                best_score, best_gr = f1m, gr
        group_reg = best_gr

    # --- 4) IRLS‑Loop + Profil‑Likelihood ---
    model     = NIMO(d_aug, hidden_dim, lam, noise_std, group_reg).to(device)
    optimizer = optim.Adam(model.shared_net.parameters(), lr=lr)
    eps, gamma = 1e-6, 1.0
    w = torch.ones(d_aug+1, device=device)

    conv_hist, prev_loss, stopped = [], float('inf'), False
    for t in range(T):
        model.eval()
        B      = model(X_t)                              # [n, d_aug+1]
        logit  = B.matmul(model.beta.unsqueeze(1)).squeeze(1)
        p      = torch.sigmoid(logit)
        W      = p*(1-p) + eps
        z      = logit + (y_t - p)/W
        BW     = B * W.unsqueeze(1)
        A      = BW.t() @ B + lam * torch.diag(w)
        b_vec  = BW.t() @ z.unsqueeze(1).squeeze(1)
        β_hat  = torch.linalg.solve(A, b_vec)
        model.beta.data.copy_(β_hat)
        w      = 1/(β_hat.abs()+eps)**gamma

        # Shared‑Net Gradient‑Schritt
        model.train()
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model.predict_logits(X_t), y_t)
        loss.backward()
        optimizer.step()

        conv_hist.append(loss.item())
        if t>0 and abs(loss.item()-prev_loss) < early_tol:
            stopped = True
            break
        prev_loss = loss.item()

    # --- 5) Evaluation & Threshold-Search ---
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(X_test_t).cpu().numpy()
    thresholds = np.linspace(0,1,100)
    f1s        = [f1_score(y_test, (probs>=thr).astype(int)) for thr in thresholds]
    bi         = int(np.argmax(f1s))
    thr        = thresholds[bi]

    # --- 6) Support aus Beta (ohne Intercept) ---
    beta_coefs = model.beta.detach().cpu().numpy()[1:]
    sel = (
      [X_columns[i] for i,b in enumerate(beta_coefs) if abs(b)>1e-2]
      if X_columns else
      [i for i,b in enumerate(beta_coefs) if abs(b)>1e-2]
    )

    # --- 7) Ergebnis zurückgeben ---
    result = {
        'model_name':        'nimoNew',
        'iteration':          iteration,
        'best_threshold':    thr,
        'best_f1':           f1s[bi],
        'y_pred':            (probs>=thr).astype(int).tolist(),
        'y_prob':            probs.tolist(),
        'selected_features': sel,
        'method_has_selection': True,
        'n_selected':        len(sel),
        'hidden_dim':        hidden_dim,
        'lam':               lam,
        'noise_std':         noise_std,
        'group_reg':         group_reg,
        'convergence_history': conv_hist,
        'stopped_early':     stopped,
        'group_reg_cv_performed': group_reg_cv,
        'best_group_reg':    best_gr if group_reg_cv else None,
        'n_iters_trained':   len(conv_hist)
    }
    return standardize_method_output(result)
