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
except ImportError as e:
    print(f"Import error in variant.py: {e}")
    def standardize_method_output(result):
        # Convert numpy types to native Python types for JSON serialization
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
    """
    Custom sinusoidal activation for low-frequency features.
    Non-linearity auf jeden einzelnen Netz-Input
    SinAct(x) = sin(0.1 * x)
    """
    def forward(self, x):
        return torch.sin(0.1 * x)


class NIMO(nn.Module):
    """
    Core NIMO module: parameter vector beta (linear GLM) + shared neural net
    that computes feature-wise modulation weights via IRLS design matrix.
    Uses hybrid IRLS for beta updates and neural net for modulation.
    """
    def __init__(
        self, input_dim, hidden_dim, pe_dim, lam=1.0, noise_std=0.0, group_reg=0.0
    ):
        super().__init__()
        # Beta parameters: [intercept, d feature coefficients]
        self.beta = nn.Parameter(torch.zeros(input_dim + 1))

        # Shared modulation network: MLP mapping pairwise interactions + identity
        # positional encoding to scalar g_ij for every feature-pair
        #forward passing
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim + pe_dim, hidden_dim),  # input: x_j and one-hot id
            nn.Tanh(),                                   # nonlinearity
            SinAct(),                                    # capture low-frequency patterns
            nn.LayerNorm(hidden_dim),                    # stabilize training
            nn.Dropout(noise_std),                       # regularization
            nn.Linear(hidden_dim, 1)                     # output scalar g_ij
        )
        # Regularization params
        self.pe_dim = pe_dim      # dimension of positional encoding (identity)
        self.lam = lam            # IRLS ridge penalty
        self.group_reg = group_reg  # group-lasso on net weights

    def forward(self, x):
        """
        Forward pass builds IRLS design matrix B of size [n, d+1]:
          B[:,0] = 1 (intercept)
          B[:,j+1] = x_j * (1 + g_j)
        where g_j = g(x_i, identity j) - g(0, identity j)
        """
        n, d = x.shape
        device = x.device
        # Intercept column, Bias term in GLM
        B0 = torch.ones(n, 1, device=device)

        # Build pairwise features X_exp and positional encodings PE, all possible pairs
        # X_exp[i,j,k] = x[i,k] if j!=k else 0
        X_exp = x.unsqueeze(1).expand(n, d, d).clone() # all pair-combinations, form [n,d] to [n,d,d]
        idx = torch.arange(d, device=device)
        X_exp[:, idx, idx] = 0.0  # zero diagonal, delete all x_{n,j,j}
        # One-hot positional encoding for feature j
        PE = torch.eye(d, device=device).unsqueeze(0).expand(n, d, d)

        # Flatten to feed shared net: shape (n*d, 2*d)
        inp_flat = torch.cat([X_exp, PE], dim=2).reshape(n * d, 2 * d)
        # Shared net forward to obtain g_flat: shape (n*d)
        g_flat = self.shared_net(inp_flat).squeeze(1)
        g = g_flat.view(n, d)

        # Zero-input baseline g0 to remove bias (if x=0), just interactions with other x_k
        zero_inp = torch.cat([torch.zeros_like(X_exp), PE], dim=2).reshape(n * d, 2 * d)
        g0_flat = self.shared_net(zero_inp).squeeze(1)
        g0 = g0_flat.view(n, d)

        # Correction term
        g_corr = g - g0
        # Modulated features: x * (1 + g_corr)
        feats = x * (1.0 + g_corr)
        # Combine intercept and features
        B = torch.cat([B0, feats], dim=1)  # [n, d+1]
        return B

    def predict_logits(self, x):
        # Linear predictor: B(x) @ beta
        return self.forward(x).matmul(self.beta.unsqueeze(1)).squeeze(1)

    def predict_proba(self, x):
        # Sigmoid link
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
    NIMO Variant with:
      - IRLS loop for beta (GLM closed-form updates)
      - Neural net modulates features g_j
      - Optional group-regularization CV on network weights
      - Early stopping on loss convergence
    """
    device = torch.device('cpu')
    # Set random seeds for reproducibility
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    # Convert data to torch tensors
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device) # [nxd]
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device) # [n]
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)
    n, d = X_t.shape

    # Default positional encoding dim
    if pe_dim is None:
        pe_dim = d

    # --- Optional - Testing Stage - Group Regularization CV ---
    best_group_reg = group_reg
    if group_reg_cv:
        best_score = -1.0
        # Try different group_reg values
        for gr in group_reg_values:
            model_cv = NIMO(d, hidden_dim, pe_dim, lam, noise_std, gr).to(device)
            opt_cv = optim.Adam(model_cv.shared_net.parameters(), lr=lr)
            # Brief training for CV
            for _ in range(min(5, T)):
                # IRLS beta update (no backprop)
                model_cv.eval()
                B_u = model_cv(X_t)
                logits = B_u.matmul(model_cv.beta.unsqueeze(1)).squeeze(1)
                p = torch.sigmoid(logits)
                W = p * (1 - p) + 1e-6
                z = logits + (y_t - p) / W
                BW = B_u * W.unsqueeze(1)
                A = BW.t() @ B_u + lam * torch.eye(d+1, device=device)
                b = BW.t() @ z.unsqueeze(1).squeeze(1)
                # Closed-form beta
                beta_hat = torch.linalg.solve(A, b)
                model_cv.beta.data.copy_(beta_hat)
                # Backprop to update shared_net
                model_cv.train(); opt_cv.zero_grad()
                loss_cv = F.binary_cross_entropy_with_logits(model_cv.predict_logits(X_t), y_t)
                loss_cv.backward(); opt_cv.step()
                # Apply group-lasso shrinkage on positional enc weights
                with torch.no_grad():
                    W1 = model_cv.shared_net[0].weight
                    for j in range(d):
                        col = W1[:, d+j]; norm = col.norm(2)
                        if norm > gr:
                            W1[:, d+j] *= (1 - gr/norm)
                        else:
                            W1[:, d+j].zero_()
            # Evaluate CV F1
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

    # --- Main IRLS + Training Loop ---
    model = NIMO(d, hidden_dim, pe_dim, lam, noise_std, group_reg).to(device)
    optimizer = optim.Adam(model.shared_net.parameters(), lr=lr)
    eps, gamma = 1e-6, 1.0  # IRLS dämpfung and weight exponent fpr adaptive ridge
    w = torch.ones(d+1, device=device)  # initial weights for ridge penalty

    convergence_history = []
    stopped_early = False
    prev_loss = float('inf')

    for t in range(T):
        # --- IRLS Beta Update (closed-form) ---
        model.eval() # [n, d+1] Design‑Matrix B aus aktuellem shared_net + β
        B_u = model(X_t)
        logits = B_u.matmul(model.beta.unsqueeze(1)).squeeze(1)
        p = torch.sigmoid(logits) # Vorhersage p_i
        W = p * (1 - p) + eps  # IRLS‑Gewichte W_i, aus der 2.Ableitung(Hesse)
        z = logits + (y_t - p) / W  #Taylor approximation
        BW = B_u * W.unsqueeze(1) # weighted Designmatrix
        A = BW.t() @ B_u + lam * torch.diag(w)  #  (Bᵀ W B + λ diag(w))  ← GLM‑Normalgleichung
        b = BW.t() @ z.unsqueeze(1).squeeze(1)
        beta_hat = torch.linalg.solve(A, b) #closed solution for beta
        model.beta.data.copy_(beta_hat)
        # Update IRLS weights for adaptive ridge
        w = 1/(beta_hat.abs() + eps)**gamma #new ridge weights, grosse koeffs -> stärker regularisiert

        # --- Neural Net Update ---
        model.train(); optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model.predict_logits(X_t), y_t)
        loss.backward(); optimizer.step()
        # Apply group-lasso shrinkage
        with torch.no_grad():
            W1 = model.shared_net[0].weight
            for j in range(d):
                col = W1[:, d+j]; norm = col.norm(2)
                if norm > group_reg:
                    W1[:, d+j] *= (1 - group_reg/norm)
                else:
                    W1[:, d+j].zero_()

        # Track convergence and early stop
        current_loss = loss.item()
        convergence_history.append(current_loss)
        if t > 0 and abs(current_loss - prev_loss) < early_stopping_tol:
            stopped_early = True
            break
        prev_loss = current_loss

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(X_test_t).cpu().numpy()
    thresholds = np.linspace(0,1,100)
    f1s = [f1_score(y_test, (probs >= thr).astype(int)) for thr in thresholds]
    best_idx = int(np.argmax(f1s))
    best_thr = thresholds[best_idx]

    # --- Feature Selection via Beta Coeffs ---
    beta_coeffs = model.beta.detach().cpu().numpy()[1:]
    if X_columns:
        selected_features = [X_columns[i] for i,b in enumerate(beta_coeffs) if abs(b) > 1e-2]
    else:
        selected_features = [i for i,b in enumerate(beta_coeffs) if abs(b) > 1e-2]

    # --- Pack result ---
    result = {
        'model_name':'nimo_variant',
        'iteration':iteration,
        'best_threshold':best_thr,
        'best_f1':f1s[best_idx],
        'y_pred':(probs>=best_thr).astype(int).tolist(),
        'y_prob':probs.tolist(),
        'selected_features':selected_features,
        'method_has_selection':True,
        'n_selected':len(selected_features),
        'hidden_dim':hidden_dim,
        'pe_dim':pe_dim,
        'convergence_history':convergence_history,
        'stopped_early':stopped_early,
        'group_reg_cv_performed':group_reg_cv,
        'best_group_reg':best_group_reg if group_reg_cv else None,
        'n_iters_trained':len(convergence_history)
    }
    return standardize_method_output(result)
