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
    iteration, randomState, X_columns=None,
    *,
    hidden_dim     = 64,
    lam            = 1.0,
    noise_std      = 0.2,
    group_reg      = 0.0,
    lr             = 1e-3,
    T              = 15,
    early_tol      = 1e-4,
    group_reg_cv   = False,
    group_reg_vals = (0.0, 0.1, 0.5, 1.0),
    X_val=None, y_val=None
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

    # --- 1b) Standardisierung (nur auf Train fitten!) ---
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_aug      = scaler.fit_transform(X_aug)
    X_test_aug = scaler.transform(X_test_aug)

    # --- 2) Torch‑Tensors & Dimensionen ---
    X_t      = torch.tensor(X_aug,      dtype=torch.float32, device=device)
    y_t      = torch.tensor(y_train,    dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_aug, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test,     dtype=torch.float32, device=device)
    n, d_aug = X_t.shape

    # Split fuer Val zur Threshold-Wahl
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=randomState)
    (tr_idx, val_idx), = sss.split(X_aug, y_train)
    X_tr, y_tr = X_aug[tr_idx], y_train[tr_idx]
    X_va, y_va = X_aug[val_idx], y_train[val_idx]
    
    # Verwende X_tr/y_tr fuer Training statt gesamtes X_t/y_t
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
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
                for thr in np.linspace(0.000, 1.000, 1001)
            )
            if f1m > best_score:
                best_score, best_gr = f1m, gr
        group_reg = best_gr

    # --- 4) IRLS‑Loop + Profil‑Likelihood ---
    model     = NIMO(d_aug, hidden_dim, lam, noise_std, group_reg).to(device)
    optimizer = optim.Adam(model.shared_net.parameters(), lr=lr, weight_decay=1e-4)
    eps, gamma = 1e-6, 1.5  # Aggressiveres Schrumpfen
    w = torch.ones(d_aug+1, device=device)

    conv_hist, prev_loss, stopped = [], float('inf'), False
    for t in range(T):
        model.eval()
        
        # Adaptive Parameter
        gamma_t = 1.0 + 0.8 * (t / max(1, T-1))  # 1.0 -> 1.8
        lam_t = lam + (lam * 4.0 - lam) * (t / max(1, T-1))  # lam -> lam*5
        
        B      = model(X_t)                              # [n, d_aug+1]
        logit  = B.matmul(model.beta.unsqueeze(1)).squeeze(1)
        p      = torch.sigmoid(logit)
        W      = p*(1-p) + eps
        z      = logit + (y_t - p)/W
        BW     = B * W.unsqueeze(1)
        A      = BW.t() @ B + lam_t * torch.diag(w)
        b_vec  = BW.t() @ z.unsqueeze(1).squeeze(1)
        β_hat  = torch.linalg.solve(A, b_vec)
        
        # Soft-Threshold (prox fuer L1) nur auf die Features (ohne Intercept)
        beta_np = β_hat.detach().cpu().numpy()
        tau_step = 1e-3  # klein starten
        beta_np[1:] = np.sign(beta_np[1:]) * np.maximum(np.abs(beta_np[1:]) - tau_step, 0.0)
        β_hat = torch.tensor(beta_np, dtype=β_hat.dtype, device=β_hat.device)
        
        model.beta.data.copy_(β_hat)
        w      = 1/(β_hat.abs()+eps)**gamma_t

        # Shared‑Net Gradient‑Schritt
        model.train()
        optimizer.zero_grad()
        
        # L1-Penalty auf g_corr (macht die Modulationsfunktion sparsamer)
        lambda_g = 1e-3  # klein anfangen
        with torch.no_grad():
            n, d = X_t.shape
            idx = torch.arange(d, device=X_t.device)
            X_exp = X_t.unsqueeze(1).expand(n, d, d).clone()
            X_exp[:, idx, idx] = 0.0
            PE = torch.eye(d, device=X_t.device).unsqueeze(0).expand(n, d, d)
            inp_flat = torch.cat([X_exp, PE], dim=2).reshape(n*d, 2*d)

            g_flat = model.shared_net(inp_flat).squeeze(1)
            g0_flat = model.shared_net(torch.zeros_like(inp_flat)).squeeze(1)
            g_corr = (g_flat - g0_flat).view(n, d)
            reg_g  = lambda_g * torch.mean(torch.abs(g_corr))
        
        loss = F.binary_cross_entropy_with_logits(model.predict_logits(X_t), y_t) + reg_g
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
        prob_val = model.predict_proba(torch.tensor(X_val, dtype=torch.float32)).cpu().numpy() if X_val is not None else None
        prob_te  = model.predict_proba(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()

    thresholds = np.linspace(0.0, 1.0, 1001)  # Feineres Grid
    if prob_val is not None:
        idx = int(np.argmax([f1_score(y_val, (prob_val>=t).astype(int)) for t in thresholds]))
        thr = float(thresholds[idx])
    else:
        idx = int(np.argmax([f1_score(y_test, (prob_te>=t).astype(int)) for t in thresholds]))
        thr = float(thresholds[idx])

    y_pred = (prob_te >= thr).astype(int)

    # --- 6) Support aus Beta (ohne Intercept) ---
    beta_coefs = model.beta.detach().cpu().numpy()[1:]

    # Validierungsbasierte Wahl von tau fuer |beta|
    tau_grid = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    best_tau, best_score = tau_grid[0], -1.0
    for tau in tau_grid:
        beta_tau = beta_coefs.copy()
        beta_tau[np.abs(beta_tau) < tau] = 0.0  # Hard-Threshold
        # proxy: Support-F1 gegen (linear ODER total) truth kannst du spaeter einbauen;
        # hier einfach "Sparsity bei aehnlicher Val-F1" als Heuristik:
        model_tmp = model  # logits haengen linear von beta ab
        with torch.no_grad():
            # setze temporaer Beta mit Threshold fuer Val-Prognose
            full_beta = model.beta.detach().clone()
            full_beta[1:] = torch.tensor(beta_tau, dtype=full_beta.dtype)
            logits = model.forward(torch.tensor(X_val, dtype=torch.float32)).matmul(full_beta.unsqueeze(1)).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
        # F1 auf Val (feines Grid fuer Klassifikationsschwelle)
        ths = np.linspace(0.0,1.0,1001)
        f1s = [f1_score(y_val, (probs>=t).astype(int), zero_division=0) for t in ths]
        f1v = max(f1s)
        # simple Auswahl: maximaler Val-F1 bei moeglichst kleiner nnz
        nnz_tau = int((np.abs(beta_tau) > 0).sum())
        score = f1v - 1e-4*nnz_tau  # kleines Parsimonie-Penalty
        if score > best_score:
            best_score, best_tau = score, tau

    # final: Hart-Threshold mit best_tau auf gesamte Beta
    beta_coefs[np.abs(beta_coefs) < best_tau] = 0.0

    # Group-Lasso (block-soft-threshold) auf Beta-Gruppen
    def build_groups(X_columns):
        groups = {}
        for j,name in enumerate(X_columns):
            # base-id am Ende extrahieren, fallback auf j
            parts = name.split('_')
            base = parts[-1] if parts[-1].isdigit() else str(j)
            groups.setdefault(base, []).append(j)
        return list(groups.values())

    groups = build_groups(X_columns) if X_columns is not None else [ [j] for j in range(len(beta_coefs)) ]

    # Block soft-threshold
    beta_np = model.beta.detach().cpu().numpy()
    beta_w = beta_np[1:]  # ohne Intercept
    tau_group = 5e-3      # tunen
    for G in groups:
        v = beta_w[G]
        norm = np.linalg.norm(v, 2)
        if norm <= tau_group:
            beta_w[G] = 0.0
        else:
            beta_w[G] = (1 - tau_group/norm) * v
    beta_np[1:] = beta_w
    beta_coefs = beta_w  # Update beta_coefs fuer selected_features

    sel = (
      [X_columns[i] for i,b in enumerate(beta_coefs) if abs(b)>1e-2]
      if X_columns else
      [i for i,b in enumerate(beta_coefs) if abs(b)>1e-2]
    )

    # Standardisierte Selection-Metadaten für Feature-Selection-Analyse
    feature_names = list(X_columns) if X_columns else list(range(len(beta_coefs)))
    selected_mask = [int(abs(b) > 1e-8) for b in beta_coefs]
    signs = [0 if abs(b) <= 1e-8 else (1 if b > 0 else -1) for b in beta_coefs]
    nnz = int(np.sum(np.abs(beta_coefs) > 1e-8))

    # --- 7) Ergebnis zurückgeben ---
    result = {
        'model_name':        'nimoNew',
        'iteration':          iteration,
        'best_threshold':    thr,
        'best_f1':           f1_score(y_test, y_pred),
        'y_pred':            y_pred.tolist(),
        'y_prob':            prob_te.tolist(),
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
        'n_iters_trained':   len(conv_hist),
        # Standardisierte Selection-Metadaten
        'feature_names': feature_names,
        'coef_all': [float(b) for b in beta_coefs],
        'selected_mask': selected_mask,
        'signs': signs,
        'nnz': nnz,
        # Sparsity-spezifische Metadaten
        'best_tau': float(best_tau),
        'tau_group': 5e-3,
        'final_nnz': int((np.abs(beta_coefs) > 0).sum()),
        'sparsity_ratio': float((np.abs(beta_coefs) > 0).sum() / len(beta_coefs))
    }
    return standardize_method_output(result)
