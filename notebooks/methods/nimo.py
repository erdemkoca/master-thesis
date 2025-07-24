import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import torch.nn.functional as F

class SinAct(nn.Module):
    def forward(self, x):
        return torch.sin(0.1 * x)

class NIMO(nn.Module):
    def __init__(self, input_dim, hidden_dim, pe_dim, lam=1.0, noise_std=0.0, group_reg=0.0):
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

        # parallel Masking + One-Hot encoding
        X_exp = x.unsqueeze(1).expand(n, d, d).clone()
        idx   = torch.arange(d, device=device)
        X_exp[:, idx, idx] = 0.0
        PE = torch.eye(d, device=device).unsqueeze(0).expand(n, d, d)

        inp_flat  = torch.cat([X_exp, PE], dim=2).reshape(n * d, 2 * d)
        g_flat    = self.shared_net(inp_flat).squeeze(1)
        g         = g_flat.view(n, d)

        zero_inp  = torch.cat([torch.zeros_like(X_exp), PE], dim=2).reshape(n * d, 2 * d)
        g0_flat   = self.shared_net(zero_inp).squeeze(1)
        g0        = g0_flat.view(n, d)

        g_corr    = g - g0
        feats     = x * (1.0 + g_corr)    # (n, d)
        B         = torch.cat([B0, feats], dim=1)
        return B

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x).matmul(self.beta.unsqueeze(1)).squeeze(1))

    def predict_logits(self, x):
        return self.forward(x).matmul(self.beta.unsqueeze(1)).squeeze(1)


def run_nimo(
    X_train, y_train, X_test, y_test,
    rng, iteration, randomState, X_columns=None,
    *,  # alle folgenden args nur per keyword
    hidden_dim=32,
    pe_dim=None,
    lam=1.0,
    noise_std=0.3,
    group_reg=0.0,
    lr=1e-3,
    T=10
):
    """
    Trainings‐Routine für NIMO mit IRLS‐Loop + Profil‐Likelihood.
    hyperparams: hidden_dim, pe_dim, lam, noise_std, group_reg, lr, T
    """
    device = torch.device('cpu')
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    X_t     = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t     = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t= torch.tensor(X_test,  dtype=torch.float32, device=device)
    n, d    = X_t.shape

    # falls pe_dim nicht gesetzt, auf Anzahl Features setzen
    if pe_dim is None:
        pe_dim = d

    model = NIMO(
        input_dim = d,
        hidden_dim= hidden_dim,
        pe_dim    = pe_dim,
        lam       = lam,
        noise_std = noise_std,
        group_reg = group_reg
    ).to(device)
    optimizer = optim.Adam(model.shared_net.parameters(), lr=lr)

    eps   = 1e-6
    gamma = 1.0
    w     = torch.ones(d+1, device=device)

    for t in range(T):
        # 1) IRLS-Schritt (wie gehabt)
        model.eval()
        B_u    = model(X_t)                              # (n, d+1)
        logits = B_u.matmul(model.beta.unsqueeze(1)).squeeze(1)
        p      = torch.sigmoid(logits)
        W      = p * (1 - p) + eps
        z      = logits + (y_t - p) / W

        BW     = B_u * W.unsqueeze(1)
        A      = BW.t().matmul(B_u) + lam * torch.diag(w)
        b      = BW.t().matmul(z.unsqueeze(1)).squeeze(1)
        beta_hat = torch.linalg.solve(A, b)
        model.beta.data.copy_(beta_hat)

        # update adaptive-Ridge weights
        w = 1.0 / (beta_hat.abs() + eps)**gamma

        # 2) Gradientenschritt für shared_net
        model.train()
        optimizer.zero_grad()
        logits2 = model.predict_logits(X_t)
        bce     = F.binary_cross_entropy_with_logits(logits2, y_t)
        # **ohne** group_penalty hier!
        bce.backward()
        optimizer.step()

        # 3) Proximal-Update für Group-Lasso auf das erste Linear-Layer
        with torch.no_grad():
            W1 = model.shared_net[0].weight  # shape (hidden_dim, d + pe_dim)
            for j in range(d):
                col = W1[:, d + j]           # Spalte j (One-Hot part)
                norm = col.norm(2)
                if norm > group_reg:
                    W1[:, d + j] = (1 - group_reg / norm) * col
                else:
                    W1[:, d + j].zero_()


    # Evaluation
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(X_test_t).cpu().numpy()
    thresholds = np.linspace(0, 1, 100)
    f1_scores  = [f1_score(y_test, (probs >= t).astype(int)) for t in thresholds]
    best_idx   = int(np.argmax(f1_scores))

    # Feature selection based on beta coefficients
    beta_coeffs = model.beta.data.cpu().numpy()[1:]  # Skip intercept
    beta_threshold = 0.01
    if X_columns is not None:
        selected_features = [X_columns[i] for i, beta in enumerate(beta_coeffs) if abs(beta) > beta_threshold]
    else:
        selected_features = [i for i, beta in enumerate(beta_coeffs) if abs(beta) > beta_threshold]

    return {
        'model_name':      'nimo',
        'iteration':       iteration,
        'best_threshold':  thresholds[best_idx],
        'best_f1':         f1_scores[best_idx],
        'y_pred':          (probs >= thresholds[best_idx]).astype(int).tolist(),
        'y_prob':          probs.tolist(),
        'selected_features': selected_features,
        'method_has_selection': True,
        'n_selected': len(selected_features),
        'hidden_dim':      hidden_dim,
        'pe_dim':          pe_dim
    }
