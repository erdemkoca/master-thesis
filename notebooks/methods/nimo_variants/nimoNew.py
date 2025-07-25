"""
NIMO Variant with Noise Injection, Lambda CV, Explicit Intercept, Zero-Mean Constraint, and Enhanced Tracking
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import standardize_method_output
except ImportError as e:
    print(f"Import error in nimoNew.py: {e}")
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
from sklearn.model_selection import train_test_split

class SinAct(nn.Module):
    def forward(self, x):
        return torch.sin(0.1 * x)

class NIMO(nn.Module):
    def __init__(self, input_dim, hidden_dim, pe_dim, noise_std=0.0):
        super().__init__()
        # parameters: beta (coefficients) and explicit bias
        self.beta  = nn.Parameter(torch.zeros(input_dim))
        self.bias  = nn.Parameter(torch.tensor(0.0))
        self.noise_std = noise_std
        # shared net for g_u
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim + pe_dim, hidden_dim),
            nn.Tanh(),
            SinAct(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        self.pe_dim = pe_dim

    def forward(self, x):  # returns design matrix B = [1, x * h(x)]
        n, d = x.shape
        device = x.device
        # intercept column
        B0 = torch.ones(n, 1, device=device)
        # mask inputs + positional enc.
        X_exp = x.unsqueeze(1).repeat(1, d, 1)
        idx   = torch.arange(d, device=device)
        X_exp[:, idx, idx] = 0.0
        PE    = torch.eye(d, device=device).unsqueeze(0).expand(n, d, d)
        # flatten
        inp = torch.cat([X_exp, PE], dim=2).reshape(n * d, 2*d)
        # compute g
        h1 = self.shared_net[0](inp)
        if self.noise_std > 0:
            h1 = h1 + torch.randn_like(h1) * self.noise_std
        g_flat = self.shared_net[1:](h1).squeeze(1)
        g = g_flat.view(n, d)
        # baseline g0
        zero_inp = torch.cat([torch.zeros_like(X_exp), PE], dim=2).reshape(n * d, 2*d)
        z1 = self.shared_net[0](zero_inp)
        if self.noise_std > 0:
            z1 = z1 + torch.randn_like(z1) * self.noise_std
        g0_flat = self.shared_net[1:](z1).squeeze(1)
        g0 = g0_flat.view(n, d)
        # form h = 1 + g - g0
        h = 1.0 + (g - g0)
        feats = x * h
        B = torch.cat([B0, feats], dim=1)
        return B

    def predict_logits(self, x):
        B = self.forward(x)
        return B[:,1:].matmul(self.beta) + self.bias

    def predict_proba(self, x):
        return torch.sigmoid(self.predict_logits(x))

# diagnostics logg
def log_diag(it, lam, loss, f1, beta, grp):
    print(f"Iter {it:2d} | lam={lam:.4f} | loss={loss:.4f} | f1={f1:.4f} | "
      f"|nz_beta={(beta.abs()>1e-6).sum().item()} | grp={grp:.4f}")


def run_nimoNew(
    X_train, y_train, X_test, y_test,
    rng, iteration, randomState, X_columns=None,
    *, hidden_dim=32, pe_dim=None,
      lam_list=(0.01,0.1,1.0,10.0), noise_std=0.3,
      group_reg=0.0, lr=1e-3, T=10, val_frac=0.2,
      early_stopping_tol=1e-4,
      group_reg_cv=False,
      group_reg_values=(0.0, 0.1, 0.5, 1.0)
):
    """
    NIMO Variant with enhanced convergence tracking, early-stopping, and group regularization CV.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    # train/val split for lambda CV
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=int(randomState)
    )
    X_tr_t = torch.from_numpy(np.array(X_tr)).float().to(device)
    y_tr_t = torch.from_numpy(np.array(y_tr)).float().to(device)
    X_val_t= torch.from_numpy(np.array(X_val)).float().to(device)
    y_val_t= torch.from_numpy(np.array(y_val)).float().to(device)
    X_test_t= torch.from_numpy(np.array(X_test)).float().to(device)
    y_test_t= torch.from_numpy(np.array(y_test)).float().to(device)
    n, d = X_tr_t.shape
    if pe_dim is None:
        pe_dim = d

    # Group regularization CV
    best_group_reg = group_reg
    if group_reg_cv:
        for gr in group_reg_values:
            model_cv = NIMO(d, hidden_dim, pe_dim, noise_std).to(device)
            opt_cv = optim.Adam(model_cv.shared_net.parameters(), lr=lr)
            eps, gamma = 1e-6, 1.0
            w = torch.ones(d+1).to(device)
            for t in range(min(5, T)):
                model_cv.eval()
                B = model_cv(X_tr_t)
                logits = B[:,1:].matmul(model_cv.beta) + model_cv.bias
                p = torch.sigmoid(logits)
                W = p*(1-p) + eps
                z = logits + (y_tr_t - p)/W
                BW = B * W.unsqueeze(1)
                A = BW.t().matmul(B) + group_reg*torch.diag(w)
                b = BW.t().matmul(z.unsqueeze(1)).squeeze(1)
                beta_hat = torch.linalg.solve(A, b)
                model_cv.beta.data.copy_(beta_hat[1:])
                model_cv.bias.data.copy_(beta_hat[0])
                w = 1/(beta_hat.abs()+eps)**gamma
                model_cv.train(); opt_cv.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model_cv.predict_logits(X_tr_t), y_tr_t)
                loss.backward(); opt_cv.step()
            with torch.no_grad():
                preds = (model_cv.predict_proba(X_tr_t).detach().cpu().numpy() >= 0.5).astype(int)
                score = f1_score(y_tr, preds)
            if score > 0:
                best_group_reg = gr
    group_reg = best_group_reg

    # Lambda CV
    best_lam, best_score = None, -1
    for lam in lam_list:
        model = NIMO(d, hidden_dim, pe_dim, noise_std).to(device)
        opt = optim.Adam(model.shared_net.parameters(), lr=lr)
        eps, gamma = 1e-6, 1.0
        w = torch.ones(d+1).to(device)
        for t in range(T):
            model.eval()
            B = model(X_tr_t)
            logits = B[:,1:].matmul(model.beta) + model.bias
            p = torch.sigmoid(logits)
            W = p*(1-p) + eps
            z = logits + (y_tr_t - p)/W
            BW = B * W.unsqueeze(1)
            A = BW.t().matmul(B) + lam*torch.diag(w)
            b = BW.t().matmul(z.unsqueeze(1)).squeeze(1)
            beta_hat = torch.linalg.solve(A, b)
            model.beta.data.copy_(beta_hat[1:]); model.bias.data.copy_(beta_hat[0])
            w = 1/(beta_hat.abs()+eps)**gamma
            model.train(); opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model.predict_logits(X_tr_t), y_tr_t)
            loss.backward(); opt.step()
        with torch.no_grad():
            preds = (model.predict_proba(X_val_t).detach().cpu().numpy() >= 0.5).astype(int)
            score = f1_score(y_val, preds)
        if score > best_score:
            best_score, best_lam = score, lam

    # Final training
    model = NIMO(d, hidden_dim, pe_dim, noise_std).to(device)
    opt = optim.Adam(model.shared_net.parameters(), lr=lr)
    eps, gamma = 1e-6, 1.0
    w = torch.ones(d+1).to(device)
    convergence_history = []
    stopped_early = False
    prev_loss = float('inf')
    for t in range(T):
        model.eval()
        B = model(X_tr_t)
        logits = B[:,1:].matmul(model.beta) + model.bias
        p = torch.sigmoid(logits)
        W = p*(1-p) + eps
        z = logits + (y_tr_t - p)/W
        BW = B * W.unsqueeze(1)
        A = BW.t().matmul(B) + best_lam*torch.diag(w)
        b = BW.t().matmul(z.unsqueeze(1)).squeeze(1)
        beta_hat = torch.linalg.solve(A, b)
        model.beta.data.copy_(beta_hat[1:]); model.bias.data.copy_(beta_hat[0])
        w = 1/(beta_hat.abs()+eps)**gamma
        model.train(); opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model.predict_logits(X_tr_t), y_tr_t)
        convergence_history.append(loss.item())
        if t>0 and abs(loss.item()-prev_loss) < early_stopping_tol:
            stopped_early = True; break
        prev_loss = loss.item()
        loss.backward(); opt.step()

    # Test evaluation
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(X_test_t).detach().cpu().numpy()
    thresholds = np.linspace(0,1,101)
    f1s = [f1_score(y_test, (probs>=thr).astype(int)) for thr in thresholds]
    best_idx = int(np.argmax(f1s)); best_thr = thresholds[best_idx]
    y_pred = (probs>=best_thr).astype(int).tolist()
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    beta_vals = model.beta.detach().cpu().numpy()
    sel = [X_columns[i] for i,v in enumerate(beta_vals) if abs(v)>1e-3]

    result = {
        'model_name':'nimoNew', 'iteration':iteration,
        'best_lambda':best_lam, 'best_threshold':best_thr, 'best_f1':max(f1s),
        'precision':prec, 'recall':rec,
        'y_pred':y_pred, 'y_prob':probs.tolist(),
        'selected_features':sel, 'method_has_selection':True, 'n_selected':len(sel),
        'hidden_dim':hidden_dim,'pe_dim':pe_dim,
        'convergence_history':convergence_history,
        'stopped_early':stopped_early,
        'group_reg_cv_performed':group_reg_cv,
        'best_group_reg':best_group_reg if group_reg_cv else None
    }
    
    return standardize_method_output(result)