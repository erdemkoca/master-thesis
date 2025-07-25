"""
NIMO Variant - Original NIMO Implementation with Enhanced Tracking
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import standardize_method_output
except ImportError as e:
    print(f"Import error in variant.py: {e}")
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

class SinAct(nn.Module):
    def forward(self, x):
        return torch.sin(0.1 * x)

class NIMO(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, pe_dim, lam=1.0, noise_std=0.0, group_reg=0.0
    ):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(input_dim + 1))
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim + pe_dim, hidden_dim),
            nn.Tanh(),
            SinAct(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(noise_std),
            nn.Linear(hidden_dim, 1)
        )
        self.pe_dim    = pe_dim
        self.lam       = lam
        self.group_reg = group_reg

    def forward(self, x):
        n, d = x.shape
        device = x.device
        B0 = torch.ones(n, 1, device=device)

        X_exp = x.unsqueeze(1).expand(n, d, d).clone()
        idx   = torch.arange(d, device=device)
        X_exp[:, idx, idx] = 0.0
        PE = torch.eye(d, device=device).unsqueeze(0).expand(n, d, d)

        inp_flat = torch.cat([X_exp, PE], dim=2).reshape(n*d, 2*d)
        g_flat = self.shared_net(inp_flat).squeeze(1)
        g = g_flat.view(n, d)

        zero_inp = torch.cat([torch.zeros_like(X_exp), PE], dim=2).reshape(n*d, 2*d)
        g0_flat = self.shared_net(zero_inp).squeeze(1)
        g0 = g0_flat.view(n, d)

        g_corr = g - g0
        feats = x * (1.0 + g_corr)
        B = torch.cat([B0, feats], dim=1)
        return B

    def predict_logits(self, x):
        return self.forward(x).matmul(self.beta.unsqueeze(1)).squeeze(1)

    def predict_proba(self, x):
        return torch.sigmoid(self.predict_logits(x))


def run_nimo_variant(
    X_train, y_train, X_test, y_test,
    rng, iteration, randomState, X_columns=None,
    *, hidden_dim=32, pe_dim=None, lam=1.0, noise_std=0.3,
      group_reg=0.0, lr=1e-3, T=10,
      early_stopping_tol=1e-4,
      group_reg_cv=False,
      group_reg_values=(0.0,0.1,0.5,1.0)
):
    """
    NIMO Variant - Original NIMO Implementation with IRLS‐Loop + Profil‐Likelihood.
    Enhanced with convergence tracking, early-stopping, and group regularization CV.
    """
    device = torch.device('cpu')
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    # Prepare tensors
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)
    n, d = X_t.shape

    if pe_dim is None:
        pe_dim = d

    # Group regularization CV
    best_group_reg = group_reg
    if group_reg_cv:
        best_score = -1.0
        for gr in group_reg_values:
            model_cv = NIMO(d, hidden_dim, pe_dim, lam, noise_std, gr).to(device)
            opt_cv = optim.Adam(model_cv.shared_net.parameters(), lr=lr)
            for t in range(min(5, T)):
                model_cv.eval()
                B_u = model_cv(X_t)
                logits = B_u.matmul(model_cv.beta.unsqueeze(1)).squeeze(1)
                p = torch.sigmoid(logits)
                W = p*(1-p) + 1e-6
                z = logits + (y_t - p)/W
                BW = B_u * W.unsqueeze(1)
                A = BW.t() @ B_u + lam * torch.eye(d+1, device=device)
                b = BW.t() @ z.unsqueeze(1).squeeze(1)
                beta_hat = torch.linalg.solve(A, b)
                model_cv.beta.data.copy_(beta_hat)
                model_cv.train(); opt_cv.zero_grad()
                loss_cv = F.binary_cross_entropy_with_logits(model_cv.predict_logits(X_t), y_t)
                loss_cv.backward(); opt_cv.step()
                with torch.no_grad():
                    W1 = model_cv.shared_net[0].weight
                    for j in range(d):
                        col = W1[:, d+j]; norm = col.norm(2)
                        if norm > gr: W1[:, d+j] *= (1 - gr/norm)
                        else: W1[:, d+j].zero_()
            model_cv.eval()
            with torch.no_grad():
                probs_cv = model_cv.predict_proba(X_t).cpu().numpy()
            f1_cv = max(
                f1_score(y_train, (probs_cv >= thr).astype(int))
                for thr in np.linspace(0,1,50)
            )
            if f1_cv > best_score:
                best_score, best_group_reg = f1_cv, gr
        group_reg = best_group_reg

    # Main model
    model = NIMO(d, hidden_dim, pe_dim, lam, noise_std, group_reg).to(device)
    optimizer = optim.Adam(model.shared_net.parameters(), lr=lr)
    eps, gamma = 1e-6, 1.0
    w = torch.ones(d+1, device=device)

    # Convergence tracking
    convergence_history = []
    stopped_early = False
    prev_loss = float('inf')

    for t in range(T):
        model.eval()
        B_u = model(X_t)
        logits = B_u.matmul(model.beta.unsqueeze(1)).squeeze(1)
        p = torch.sigmoid(logits)
        W = p*(1-p) + eps
        z = logits + (y_t - p)/W
        BW = B_u * W.unsqueeze(1)
        A = BW.t() @ B_u + lam * torch.diag(w)
        b = BW.t() @ z.unsqueeze(1).squeeze(1)
        beta_hat = torch.linalg.solve(A, b)
        model.beta.data.copy_(beta_hat)
        w = 1/(beta_hat.abs()+eps)**gamma
        model.train(); optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model.predict_logits(X_t), y_t)
        loss.backward(); optimizer.step()
        with torch.no_grad():
            W1 = model.shared_net[0].weight
            for j in range(d):
                col = W1[:, d+j]; norm = col.norm(2)
                if norm > group_reg: W1[:, d+j] *= (1-group_reg/norm)
                else: W1[:, d+j].zero_()
        current_loss = loss.item()
        convergence_history.append(current_loss)
        if t>0 and abs(current_loss-prev_loss) < early_stopping_tol:
            stopped_early = True
            break
        prev_loss = current_loss

    # Evaluate
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(X_test_t).cpu().numpy()
    thresholds = np.linspace(0,1,100)
    f1s = [f1_score(y_test, (probs>=thr).astype(int)) for thr in thresholds]
    best_idx = int(np.argmax(f1s))
    best_thr = thresholds[best_idx]

    beta_coeffs = model.beta.detach().cpu().numpy()[1:]
    selected_features = [
        X_columns[i] for i,b in enumerate(beta_coeffs) if abs(b)>1e-2
    ] if X_columns else [
        i for i,b in enumerate(beta_coeffs) if abs(b)>1e-2
    ]

    result = {
        'model_name': 'nimo_variant',
        'iteration': iteration,
        'best_threshold': best_thr,
        'best_f1': f1s[best_idx],
        'y_pred': (probs>=best_thr).astype(int).tolist(),
        'y_prob': probs.tolist(),
        'selected_features': selected_features,
        'method_has_selection': True,
        'n_selected': len(selected_features),
        'hidden_dim': hidden_dim,
        'pe_dim': pe_dim,
        'convergence_history': convergence_history,
        'stopped_early': stopped_early,
        'group_reg_cv_performed': group_reg_cv,
        'best_group_reg': best_group_reg if group_reg_cv else None,
        'n_iters_trained': len(convergence_history)
    }
    
    return standardize_method_output(result)
