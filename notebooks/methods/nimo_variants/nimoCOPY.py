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
from torch.utils.data import DataLoader, TensorDataset


def to_bin(x, n_bits):
    """Convert integer to binary representation with n_bits, centered around 0."""
    return np.array([int(b) for b in format(x, f'0{n_bits}b')]) - 0.5

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
    Enhanced with Baseline features: binary encoding, CO-matrix, c-scaling.
    """
    def __init__(self, d, hidden_dim=128, dropout=0.3, out_scale=0.3, activation='relu',
                 use_baseline_features=True):
        super().__init__()
        self.d = d
        self.out_scale = out_scale
        self.use_baseline_features = use_baseline_features

        # Beta parameters (with c-scaling like Baseline)
        self.beta = nn.Parameter(torch.zeros(d + 1))  # [b0, b1..bd] Parameters
        self.c = nn.Parameter(torch.ones(d) * 0.1)    # Feature scaling parameters
        self.alpha2 = nn.Parameter(torch.tensor(2.0)) # Output scaling

        # Binary encoding setup (like Baseline)
        if use_baseline_features:
            self.n_bits = int(np.floor(np.log2(d))) + 1
            BinMap = np.vstack([to_bin(i, self.n_bits) for i in range(1, d + 1)])
            self.register_buffer("binmap", torch.tensor(BinMap, dtype=torch.float32))

            # CO-matrix for interactions (like Baseline)
            CO = np.ones((d, d))
            np.fill_diagonal(CO, 0)
            CO = np.hstack([CO, BinMap])
            self.register_buffer("CO", torch.tensor(CO, dtype=torch.float32))

            # Enhanced MLP with binary encoding (like Baseline)
            # FIXED: Input dimension is d + d*n_bits (not d + n_bits)
            self.fc1 = nn.Linear(d + d * self.n_bits, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim + d * self.n_bits, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim + hidden_dim + d * self.n_bits, d)
            self.dropout2 = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

            # Group layers for easier state dict handling
            self.mlp_layers = nn.ModuleList([self.fc1, self.fc2, self.fc3])
        else:
            # Original MLP architecture
            if activation == 'relu':
                act_fn = nn.ReLU
            elif activation == 'silu':
                act_fn = nn.SiLU
            else:  # default to tanh
                act_fn = nn.Tanh

            self.mlp = nn.Sequential(
                nn.Linear(d, hidden_dim), act_fn(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), act_fn(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2), act_fn(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, d)
            )

            # Initialize final layer weights to be small
            with torch.no_grad():
                self.mlp[-1].weight.mul_(0.1)
                self.mlp[-1].bias.mul_(0.1)
                self.register_buffer("g0", self.mlp(torch.zeros(1, d)).detach())

        # For baseline features, initialize fc3 weights to be small
        if self.use_baseline_features:
            with torch.no_grad():
                self.fc3.weight.mul_(0.1)
                self.fc3.bias.mul_(0.1)
                # No g0 buffer needed for baseline features

    def get_mlp_parameters(self):
        """Get MLP parameters for optimizer setup."""
        if self.use_baseline_features:
            return list(self.mlp_layers.parameters())
        else:
            return list(self.mlp.parameters())

    def get_first_layer_weights(self):
        """Get first layer weights for Group-L2 regularization."""
        if self.use_baseline_features:
            return self.fc1.weight
        else:
            return self.mlp[0].weight

    def get_mlp_state_dict(self):
        """Get MLP state dict for saving/loading."""
        if self.use_baseline_features:
            return self.mlp_layers.state_dict()
        else:
            return self.mlp.state_dict()

    def load_mlp_state_dict(self, state_dict):
        """Load MLP state dict."""
        if self.use_baseline_features:
            self.mlp_layers.load_state_dict(state_dict)
        else:
            self.mlp.load_state_dict(state_dict)

    def verify_dimensions(self, x):
        """Verify that all dimensions are correct (for debugging)."""
        if self.use_baseline_features:
            n = x.size(0)
            d = x.size(1)

            # Binary encoding
            bin_codes = self.binmap.unsqueeze(0).expand(n, -1, -1)  # [n, d, n_bits]
            bin_codes = bin_codes.reshape(n, -1)  # [n, d*n_bits]
            x_aug = torch.cat([x, bin_codes], dim=1)  # [n, d + d*n_bits]

            print(f"Input x: {x.shape}")
            print(f"Binary codes: {bin_codes.shape}")
            print(f"Augmented input: {x_aug.shape}")
            print(f"fc1 expects: {self.fc1.in_features}, got: {x_aug.shape[1]}")

            # Test forward pass
            z1 = self.fc1(x_aug)
            print(f"fc1 output: {z1.shape}")

            z1_cat = torch.cat([z1, x_aug[:, self.d:(self.d + self.d * self.n_bits)]], dim=1)
            print(f"z1 concatenated: {z1_cat.shape}")
            print(f"fc2 expects: {self.fc2.in_features}, got: {z1_cat.shape[1]}")

            z2 = torch.sin(2 * np.pi * self.fc2(z1_cat))
            print(f"fc2 output: {z2.shape}")

            z_final = torch.cat([z2, z1_cat], dim=1)
            print(f"Final concatenated: {z_final.shape}")
            print(f"fc3 expects: {self.fc3.in_features}, got: {z_final.shape[1]}")

            return z_final

    def forward_MLP_baseline(self, x):
        """Baseline-style MLP with binary encoding and special activations."""
        n = x.size(0)
        # Add binary encoding - FIXED: correct shape [n, d*n_bits]
        bin_codes = self.binmap.unsqueeze(0).expand(n, -1, -1)  # [n, d, n_bits]
        bin_codes = bin_codes.reshape(n, -1)  # [n, d*n_bits]
        x_aug = torch.cat([x, bin_codes], dim=1)

        # First layer with tanh + noise (like Baseline)
        z1 = self.fc1(x_aug)
        z1 = torch.tanh(0.3 * (z1 + 0.2 * torch.randn_like(z1)))  # Noise injection
        # FIXED: Concatenate with binary codes (d*n_bits dimensions)
        z1 = torch.cat([z1, x_aug[:, self.d:(self.d + self.d * self.n_bits)]], dim=1)

        # Second layer with sin activation (like Baseline)
        z2 = torch.sin(2 * np.pi * self.fc2(z1))
        z2 = self.dropout2(z2)

        # Concatenate and final layer
        z = torch.cat([z2, z1], dim=1)
        return self.fc3(z)

    def corrections_uncentered(self, x):
        """g(x) - g(0)  (no mean-centering)."""
        if self.use_baseline_features:
            # Use Baseline-style MLP
            g = self.forward_MLP_baseline(x)
            g = 2 * torch.tanh(g) + 1  # Like Baseline
            g = g * 0.5 * (1.0 + torch.tanh(self.alpha2))  # Scaling

            # Apply CO-matrix masking for interactions (like Baseline)
            if hasattr(self, 'CO'):
                # CO matrix has shape [d, d + n_bits], we only need the first d columns for masking
                co_mask = self.CO[:, :self.d]  # [d, d] for masking
                # Apply per-feature masking: g[i,j] *= co_mask[j,j] (diagonal elements)
                g = g * co_mask.diagonal().unsqueeze(0)  # [n, d] * [1, d]

            return g
        else:
            # Original MLP
            g = self.mlp(x)
            g = self.out_scale * torch.tanh(g)  # Bound the output
            return g - self.g0  # broadcast over batch

    def corrections(self, x):
        """Mean-centered correction per feature (batch-wise)."""
        if self.use_baseline_features:
            # For Baseline features, we don't mean-center (like original Baseline)
            return self.corrections_uncentered(x)
        else:
            # Original mean-centering
            g_corr = self.corrections_uncentered(x)
            return g_corr - g_corr.mean(dim=0, keepdim=True)

    def design_matrix(self, x, use_correction=True):
        n, d = x.shape
        ones = torch.ones(n, 1, device=x.device, dtype=x.dtype)
        feats = x if not use_correction else x * (1.0 + self.corrections(x)) # each x_j is scale with (1+g_j(x))
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
    Enhanced with c-scaling like Baseline.
    w: per-coefficient weights (length d+1). We do NOT penalize intercept → set w[0]=0.
    """
    B = model.design_matrix(X, use_correction=use_correction)  # [n, d+1]
    logits = B.matmul(model.beta)
    p = torch.sigmoid(logits)
    W = p * (1.0 - p) + eps
    z = logits + (y - p) / W

    if model.use_baseline_features:
        # Use c-scaling like Baseline
        c = torch.abs(model.c)  # Ensure positive scaling
        dc = torch.diag(c)
        X_tilde = B[:, 1:] @ dc  # Skip intercept, apply c-scaling to features
        X_tilde = torch.cat([B[:, :1], X_tilde], dim=1)  # Add intercept back

        # IRLS with c-scaling
        A = X_tilde.t().matmul(X_tilde * W.unsqueeze(1)) + lam_base * torch.diag(w)
        b = X_tilde.t().matmul(W * z)
        gamma_sol = torch.linalg.solve(A, b)

        # Apply c-scaling to get final beta
        beta_new = torch.cat([gamma_sol[:1], c * gamma_sol[1:]])
    else:
        # Original IRLS without c-scaling
        BW = B * W.unsqueeze(1)
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
    hidden_dim=128,            # Increased capacity
    dropout=0.3,               # Higher dropout for regularization
    activation='relu',         # Modern activation function
    use_baseline_features=True, # Enable Baseline features (binary encoding, CO-matrix, c-scaling)
    num_epochs=1500,           # More epochs for deep learning style
    irls_every=1,              # IRLS every batch (joint optimization)
    beta_update_freq='batch',  # 'batch', 'epoch', or 'every_n_batches'
    beta_max_step=0.2,         # Smaller step size for batch-wise β updates
    lr=2e-4,                   # Slightly higher learning rate for better convergence
    optimizer='adamw',         # AdamW for better regularization
    lr_scheduler=True,         # Enable learning rate scheduling
    patience=30,               # Patience for early stopping
    min_delta=1e-6,            # Minimum change for early stopping
    lam_l2=0.1, tau_l1=0.01,   # More moderate penalties for better learning
    lam_g=0.08,                # L1 on |g_corr| with curriculum(graduall increase out_scale)
    lam_group=0.02,            # Group-L2 penalty on NN inputs
    tau_beta_report=0.01,      # Selection threshold
    eps_g=1e-3,
    out_scale=0.2,             # Tighter correction amplitude
    warm_start_steps=5,        # More warm-up steps
    use_no_harm=True,          # Enable no-harm switch
    gamma=1.3,                 # Adaptive lasso gamma
    batch_size=64,             # Mini-batch size for training
    noise_scale=0.02,          # Increased noise injection
    verbose=True               # Enable training logs
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
    model = NIMO(d, hidden_dim=hidden_dim, dropout=dropout, out_scale=out_scale,
                 activation=activation, use_baseline_features=use_baseline_features).to(device)

    # Choose optimizer (adapt to architecture)
    params = model.get_mlp_parameters()
    if use_baseline_features:
        # For Baseline features, also optimize c + alpha2
        params.extend([model.c, model.alpha2])

    if optimizer == 'adamw':
        opt = optim.AdamW(params, lr=lr, weight_decay=5e-4)  # More moderate weight decay
    else:
        opt = optim.Adam(params, lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = None
    if lr_scheduler and X_val is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=10, verbose=verbose, min_lr=1e-6
        )

    # Dataset & DataLoader für Mini-Batches
    train_dataset = TensorDataset(Xt, yt)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize adaptive weights for β (d+1) — intercept weight 0
    w = torch.ones(d+1, device=device)
    w[0] = 0.0

    # ---- 1.5) Warm start β with plain logistic (no correction)
    for _ in range(warm_start_steps):
        w = update_beta_irls_adaptive(model, Xt, yt, w, lam_base=lam_l2, tau_l1=tau_l1,
                                     gamma=gamma, use_correction=False)

    # ---- 2) Deep Learning Style Training with Patience-based Early Stopping
    best_val_loss = float("inf")
    best_mlp_state = None
    loss_hist, val_loss_hist = [], []
    train_f1_hist, val_f1_hist = [], []
    stopped = False
    patience_counter = 0

    if verbose:
        print(f"Starting NIMO Joint Optimization for {num_epochs} epochs...")
        print(f"Model: {hidden_dim} hidden units, {dropout} dropout, {activation} activation")
        print(f"Optimizer: {optimizer.upper()}, LR: {lr}")
        print(f"β Update: {beta_update_freq} (max_step={beta_max_step})")
        print("-" * 80)

    for epoch in range(num_epochs):
        # ---- (1) Curriculum annealing ----
        alpha = epoch / max(1, num_epochs-1)
        model.out_scale = 0.1 + (out_scale - 0.1) * alpha
        lam_g_t = lam_g + (lam_g*0.3 - lam_g) * alpha

        # ---- (2) Joint NN + β Training über alle Mini-Batches ----
        model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0

        for xb, yb in train_loader:
            # ---- (2a) NN Training Step ----
            opt.zero_grad()
            logits = model.predict_logits(xb, use_correction=True)
            bce = F.binary_cross_entropy_with_logits(logits, yb)

            with torch.no_grad():
                g_corr = model.corrections(xb)

            # Additional noise injection for regularization (only for non-baseline features)
            if model.training and not model.use_baseline_features:
                g_corr = g_corr + noise_scale * torch.randn_like(g_corr)

            # Orthogonalitäts- & Regularisierungsterms
            align = torch.mean(torch.abs((xb * g_corr).mean(dim=0)))
            W1 = model.get_first_layer_weights()
            group_l2 = torch.sqrt((W1**2).sum(dim=0) + 1e-12).mean()
            reg_group = lam_group * group_l2
            reg_g = lam_g_t * g_corr.abs().mean()

            loss = bce + reg_g + reg_group + 1e-3 * align
            loss.backward()

            # Gradient clipping - FIXED: use params instead of model.mlp.parameters()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

            # ---- (2b) β Update Step (Joint Optimization) ----
            if beta_update_freq == 'batch':
                # Update β after every batch (like AdaptiveRidge)
                with torch.no_grad():
                    w = update_beta_irls_adaptive(
                        model, xb, yb, w,
                        lam_base=lam_l2, tau_l1=tau_l1,
                        gamma=gamma, use_correction=True,
                        eps=1e-6, max_step=beta_max_step
                    )
            elif beta_update_freq == 'every_n_batches' and batch_count % irls_every == 0:
                # Update β every N batches
                with torch.no_grad():
                    w = update_beta_irls_adaptive(
                        model, xb, yb, w,
                        lam_base=lam_l2, tau_l1=tau_l1,
                        gamma=gamma, use_correction=True,
                        eps=1e-6, max_step=beta_max_step
                    )

            epoch_loss += loss.item()
            batch_count += 1

            # Track training accuracy
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                train_correct += (pred == yb).sum().item()
                train_total += yb.size(0)

        # ---- (2c) Epoch-wise β Update (if not batch-wise) ----
        if beta_update_freq == 'epoch':
            model.eval()
            with torch.no_grad():
                w = update_beta_irls_adaptive(
                    model, Xt, yt, w,
                    lam_base=lam_l2, tau_l1=tau_l1,
                    gamma=gamma, use_correction=True,
                    eps=1e-6, max_step=0.5  # Larger step for full dataset
                )

        # ---- (4) Validation and Metrics ----
        epoch_loss /= len(train_loader)
        train_acc = train_correct / train_total
        loss_hist.append(epoch_loss)

        val_loss = None
        val_acc = None
        val_f1 = None

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model.predict_logits(XvaT, use_correction=True)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, yvaT).item()
                val_pred = (torch.sigmoid(val_logits) > 0.5).float()
                val_acc = (val_pred == yvaT).float().mean().item()

                # Calculate F1 score
                val_pred_np = val_pred.cpu().numpy()
                val_true_np = yvaT.cpu().numpy()
                val_f1 = f1_score(val_true_np, val_pred_np, zero_division=0)

                val_loss_hist.append(val_loss)
                val_f1_hist.append(val_f1)

                # Update best model
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_mlp_state = model.get_mlp_state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Learning rate scheduling
                if scheduler is not None:
                    scheduler.step(val_loss)

        # Calculate training F1
        with torch.no_grad():
            train_logits = model.predict_logits(Xt, use_correction=True)
            train_pred = (torch.sigmoid(train_logits) > 0.5).float()
            train_pred_np = train_pred.cpu().numpy()
            train_true_np = yt.cpu().numpy()
            train_f1 = f1_score(train_true_np, train_pred_np, zero_division=0)
            train_f1_hist.append(train_f1)

        # ---- (5) Logging ----
        if verbose and (epoch % 50 == 0 or epoch < 10):
            lr_current = opt.param_groups[0]['lr']
            if X_val is not None:
                print(f"Epoch {epoch:4d} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | "
                      f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | LR: {lr_current:.2e}")
            else:
                print(f"Epoch {epoch:4d} | Train Loss: {epoch_loss:.6f} | Train F1: {train_f1:.4f} | LR: {lr_current:.2e}")

        # ---- (6) Early stopping check ----
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
            stopped = True
            break

    # Restore best MLP state
    if best_mlp_state is not None:
        model.load_mlp_state_dict(best_mlp_state)

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

    # Calculate training performance for overfitting analysis
    with torch.no_grad():
        prob_train = model.predict_proba(Xt, use_correction=use_corr_final).cpu().numpy()
    train_pred = (prob_train >= thr).astype(int)
    train_f1 = float(f1_score(y_train, train_pred, zero_division=0))
    train_acc = float(accuracy_score(y_train, train_pred))

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

    # ---- 6) Extract neural network weights and biases for analysis
    with torch.no_grad():
        # Extract weights and biases from all layers
        nn_weights = []
        nn_biases = []

        # Choose the correct layers based on architecture
        layers = model.mlp if not model.use_baseline_features else model.mlp_layers

        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn_weights.append(layer.weight.detach().cpu().numpy().tolist())
                if layer.bias is not None:
                    nn_biases.append(layer.bias.detach().cpu().numpy().tolist())
                else:
                    nn_biases.append([0.0] * layer.out_features)  # No bias case

        # Also extract the first layer weights specifically for sparsity analysis
        if model.use_baseline_features:
            first_layer_weights = model.fc1.weight.detach().cpu().numpy()  # [hidden_dim, input_dim]
            first_layer_biases = model.fc1.bias.detach().cpu().numpy() if model.fc1.bias is not None else np.zeros(model.fc1.out_features)
        else:
            first_layer_weights = model.mlp[0].weight.detach().cpu().numpy()  # [hidden_dim, input_dim]
            first_layer_biases = model.mlp[0].bias.detach().cpu().numpy() if model.mlp[0].bias is not None else np.zeros(model.mlp[0].out_features)

    # ---- 7) Report standardized coefficients (no conversion to RAW)
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

    # ---- 8) Results
    feature_names = list(X_columns) if X_columns else [f"feature_{i}" for i in range(len(beta_std))]
    result = {
        "model_name": "nimo",
        "iteration": iteration,
        "random_seed": randomState,

        # flat metrics for your existing plots
        "f1": f1,
        "accuracy": acc,
        "train_f1": train_f1,
        "train_accuracy": train_acc,
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
            # Add neural network weights and biases for analysis
            "nn_weights": nn_weights,
            "nn_biases": nn_biases,
            "first_layer_weights": first_layer_weights.tolist(),
            "first_layer_biases": first_layer_biases.tolist(),
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
            "val_loss_history": [float(x) for x in val_loss_hist] if val_loss_hist else None,
            "train_f1_history": [float(x) for x in train_f1_hist],
            "val_f1_history": [float(x) for x in val_f1_hist] if val_f1_hist else None,
            "n_iters": len(loss_hist),
            "stopped_early": bool(stopped),
            "final_epoch": len(loss_hist) - 1,
        },
        "hyperparams": {
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "activation": str(activation),
            "use_baseline_features": bool(use_baseline_features),
            "num_epochs": int(num_epochs),
            "irls_every": int(irls_every),
            "beta_update_freq": str(beta_update_freq),
            "beta_max_step": float(beta_max_step),
            "lr": float(lr),
            "optimizer": str(optimizer),
            "lr_scheduler": bool(lr_scheduler),
            "patience": int(patience),
            "min_delta": float(min_delta),
            "noise_scale": float(noise_scale),
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
            "batch_size": int(batch_size),
            "verbose": bool(verbose),
        },
    }
    return standardize_method_output(result)
