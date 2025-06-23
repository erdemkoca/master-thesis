import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import torch.nn.functional as F

class NIMO(nn.Module):
    """
    Nonlinear Interpretable Model (NIMO) mit Profil-Likelihood für logistische Regression.
    F(x): design matrix B_u, intern genutzt für Analytisches Update und logistische Vorhersage.
    Shared network g_u mit positional encoding.
    """
    def __init__(self, input_dim, hidden_dim, pe_dim, lam=1.0, noise_std=0.0, group_reg=0.0):
        super(NIMO, self).__init__()
        # Linear coefficients beta (bias + weights)
        self.beta = nn.Parameter(torch.zeros(input_dim + 1))
        # Shared network g_u
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim + pe_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(noise_std),
            nn.Linear(hidden_dim, 1)
        )
        self.pe_dim = pe_dim
        self.lam = lam
        self.group_reg = group_reg

    def forward(self, x):
        # Erzeuge Design-Matrix B_u: [1, x1*(1+g1),..., xd*(1+gd)]
        n, d = x.shape
        device = x.device
        B = [torch.ones(n, 1, device=device)]
        for j in range(d):
            x_minus_j = x.clone()
            x_minus_j[:, j] = 0.0
            pe = torch.zeros(n, d, device=device)
            pe[:, j] = 1.0
            inp = torch.cat([x_minus_j, pe[:, :self.pe_dim]], dim=1)
            g_j = self.shared_net(inp).squeeze(1)
            col = x[:, j] * (1.0 + g_j)
            B.append(col.unsqueeze(1))
        return torch.cat(B, dim=1)  # (n, d+1)

    def predict_proba(self, x):
        B_u = self.forward(x)
        logits = B_u.matmul(self.beta.unsqueeze(1)).squeeze(1)
        return torch.sigmoid(logits)

    def predict_logits(self, x):
        B_u = self.forward(x)
        return B_u.matmul(self.beta.unsqueeze(1)).squeeze(1)

# Einzige run_nimo-Funktion

def run_nimo(X_train, y_train, X_test, y_test,
             rng, iteration, randomState, X_columns=None):
    """
    Trainingsroutine für NIMO mit logistischem IRLS + Profil-Likelihood.
    rng: zentraler Generator (für Consistency)
    randomState: int-Seed für Torch/NumPy Prozesse
    """
    device = torch.device('cpu')
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    n, d = X_t.shape
    hidden_dim = 64
    pe_dim = d
    lam = 1.0
    noise_std = 0.0
    group_reg = 0.01
    lr = 1e-3
    T = 10  # Profil-Likelihood Iterationen

    model = NIMO(d, hidden_dim, pe_dim, lam=lam,
                 noise_std=noise_std, group_reg=group_reg).to(device)
    optimizer = optim.Adam(model.shared_net.parameters(), lr=lr)

    for t in range(T):
        # IRLS-Schritt für logistische Regression
        model.eval()
        B_u = model(X_t)
        logits = B_u.matmul(model.beta.unsqueeze(1)).squeeze(1)
        p = torch.sigmoid(logits)
        W = p * (1 - p)
        z = logits + (y_t - p) / (W + 1e-6)
        # Weighted least squares mit Ridge
        BW = B_u * W.unsqueeze(1)
        A = BW.t().matmul(B_u) + lam * torch.eye(d + 1, device=device)
        b = BW.t().matmul(z.unsqueeze(1)).squeeze(1)
        beta_hat = torch.linalg.solve(A, b)
        model.beta.data.copy_(beta_hat)

        # Gradienten-Update für shared_net
        model.train()
        optimizer.zero_grad()
        logits2 = model.predict_logits(X_t)
        loss = F.binary_cross_entropy_with_logits(logits2, y_t)
        # Gruppen-L2 Penalty auf erste FC-Schicht
        group_loss = sum(
            param.norm(2)
            for name, param in model.shared_net.named_parameters()
            if 'weight' in name and param.dim() > 1
        )
        loss = loss + group_reg * group_loss
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(X_test_t).cpu().numpy()
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_test, (probs >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    y_pred = (probs >= best_threshold).astype(int)

    selected_features = X_columns if X_columns else list(range(d))
    return {
        'model_name': 'nimo',
        'iteration': iteration,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'y_pred': y_pred.tolist(),
        'y_prob': probs.tolist(),
        'selected_features': selected_features,
        'hidden_dim': hidden_dim,
        'pe_dim': pe_dim
    }