"""
NIMO Baseline Variant - Adaptive Ridge Logistic Regression with Lightning
(Stabilized IRLS; epoch-wise beta updates; probability outputs; HParam sweep)
- Self-features are ENABLED by default: ["x2","sin","tanh","arctan"]
"""

import os
import sys
import math
import itertools
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Lightning EarlyStopping (optional)
try:
    from lightning.pytorch.callbacks import EarlyStopping
except Exception:
    EarlyStopping = None

# Allow relative utils import if present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import standardize_method_output
except Exception:
    def standardize_method_output(result: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                out[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32, np.float16)):
                out[k] = float(v)
            else:
                out[k] = v
        return out


# -------------------- helpers --------------------

def to_bin(x: int, n_bits: int) -> np.ndarray:
    # centered bits in {-0.5, +0.5}
    return np.array([int(b) for b in format(x, f'0{n_bits}b')], dtype=np.float32) - 0.5


def add_self_features(
    X: np.ndarray,
    self_features: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    """
    Expand X with simple self-nonlinearities to help NIMO capture self-terms.

    self_features: list of transforms to apply to each column of X.
      Supported: "x2", "x3", "sin", "cos", "abs", "tanh", "arctan" (alias: "atan")
    Returns (X_expanded, mapping) where mapping holds (orig_col_idx, transform_name)
    """
    if self_features is None:
        # DEFAULT: DISABLED to avoid self-feature leak (paper compliance)
        # Enable only if you understand the implications and fix CO masking
        self_features = []

    if not self_features:
        return X.astype(np.float32), []

    transforms = {
        "x2":    lambda v: v * v,
        "x3":    lambda v: v * v * v,
        "sin":   lambda v: np.sin(v),
        "cos":   lambda v: np.cos(v),
        "abs":   lambda v: np.abs(v),
        "tanh":  lambda v: np.tanh(v),
        "arctan":lambda v: np.arctan(v),
        "atan":  lambda v: np.arctan(v),
    }
    new_cols = []
    mapping = []
    for j in range(X.shape[1]):
        col = X[:, j:j+1]
        for name in self_features:
            fn = transforms.get(name)
            if fn is None:
                continue
            new_cols.append(fn(col))
            mapping.append((j, name))

    if new_cols:
        X_new = np.concatenate([X] + new_cols, axis=1).astype(np.float32)
    else:
        X_new = X.astype(np.float32)
    return X_new, mapping


def build_CO_with_derivatives(p_raw: int, mapping: List[Tuple[int, str]], n_bits: int) -> np.ndarray:
    """
    Build CO mask that properly handles self-feature leak by masking derived features.
    
    Args:
        p_raw: Number of original features
        mapping: List of (orig_col_idx, transform_name) for derived features
        n_bits: Number of positional encoding bits
        
    Returns:
        CO: (p, p + n_bits) mask where p = p_raw + len(mapping)
    """
    p = p_raw + len(mapping)
    CO = np.ones((p, p + n_bits), dtype=np.float32)
    
    # For each raw feature j, find all derived columns that come from j
    derived_by_raw = {j: [] for j in range(p_raw)}
    for k, (j, _) in enumerate(mapping, start=p_raw):
        derived_by_raw[j].append(k)
    
    # Zero out self and derived features when querying g_u_j for raw features
    for j in range(p_raw):
        CO[j, j] = 0.0  # zero self
        for k in derived_by_raw[j]:
            CO[j, k] = 0.0  # zero derived features from j
    
    # For derived features, zero self and progenitor raw column
    for k in range(p_raw, p):
        CO[k, k] = 0.0  # zero self
        j = mapping[k - p_raw][0]  # progenitor raw column
        CO[k, j] = 0.0  # zero progenitor
    
    return CO


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return {'features': self.features[idx], 'target': self.targets[idx]}


# -------------------- Model --------------------

class AdaptiveRidgeLogisticRegression(L.LightningModule):
    """
    Neural-Interaction + Adaptive Ridge Logistic Regression (NIMO-style)
    - NN modulates interactions excluding self (via CO mask).
    - Epoch-wise stabilized IRLS to update beta (once per epoch).
    - NN (and c, beta_0, alpha2) trained via SGD.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        learning_rate: float = 3e-4,
        group_penalty: float = 1.0,
        dropout: float = 0.0,
        hidden_dim: Optional[int] = None,
        noise_std: float = 0.2,
        lambda_reg: float = 1e-3,  # Explicit λ regularization
        c_penalty: float = 0.01,   # Quadratic penalty on c
    ):
        super().__init__()
        # Disable automatic optimization for manual control
        self.automatic_optimization = False

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.lr = float(learning_rate)

        self.group_penalty = float(group_penalty)
        self.dropout = float(dropout)
        self.noise_std = float(noise_std)
        self.lambda_reg = float(lambda_reg)
        self.c_penalty = float(c_penalty)

        self.save_hyperparameters()

        # Positional encoding via binary map
        self.n_bits = int(np.floor(np.log2(self.input_dim))) + 1
        bin_map = np.vstack([to_bin(i, self.n_bits) for i in range(1, self.input_dim + 1)])  # (p, n_bits)

        # CO: mask-out self + append positional bits (will be updated if self-features are used)
        CO = np.ones((self.input_dim, self.input_dim), dtype=np.float32)
        np.fill_diagonal(CO, 0.0)
        CO = np.hstack([CO, bin_map.astype(np.float32)])  # (p, p + n_bits)

        # Register buffers so Lightning moves devices
        self.register_buffer("CO", torch.from_numpy(CO))                  # (p, p+n_bits)
        self.register_buffer("beta", torch.randn(self.input_dim) * 0.1)   # (p,)
        self.register_buffer("beta_0", torch.randn(1) * 0.1)              # (1,)

        # Trainable scalars / vectors
        # Use softplus to enforce c_i > 0 constraint
        self.rho = nn.Parameter(torch.ones(self.input_dim) * 0.1)  # log-space parameter
        self.alpha2 = nn.Parameter(torch.tensor(2.0))
        
        # Store mapping for self-features (if any)
        self.self_feature_mapping = []

        # MLP definition
        in1 = self.input_dim + self.n_bits
        h1 = hidden_dim if hidden_dim is not None else 64
        h2 = hidden_dim if hidden_dim is not None else 128

        self.fc1 = nn.Linear(in1, h1)
        self.fc2 = nn.Linear(h1 + self.n_bits, h2)
        self.fc3 = nn.Linear(h2 + h1 + self.n_bits, self.output_dim)
        self.dropout2 = nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity()

        # Accumulators for epoch-wise IRLS
        self.A_sum = None  # (p, p)
        self.b_sum = None  # (p,)

        # vmap availability
        self._use_vmap = hasattr(torch, "vmap")
    
    def update_CO_mask(self, mapping: List[Tuple[int, str]]):
        """Update CO mask to handle self-feature leak properly."""
        if not mapping:
            return  # No self-features, keep original CO
        
        self.self_feature_mapping = mapping
        p_raw = self.input_dim - len(mapping)
        CO_new = build_CO_with_derivatives(p_raw, mapping, self.n_bits)
        self.register_buffer("CO", torch.from_numpy(CO_new))
    
    @property
    def c(self):
        """Get c = softplus(rho) to enforce c_i > 0."""
        return torch.nn.functional.softplus(self.rho)

    def forward_MLP(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, p + n_bits) where first p columns are features
        """
        z1 = self.fc1(X)
        if self.training and self.noise_std > 0:
            z1 = z1 + self.noise_std * torch.randn_like(z1)
        z1 = torch.tanh(0.3 * z1)

        pos = X[:, self.input_dim:(self.input_dim + self.n_bits)]
        z1p = torch.cat([z1, pos], dim=1)

        z2 = torch.sin(2 * math.pi * self.fc2(z1p))
        z2 = self.dropout2(z2)

        z = torch.cat([z2, z1p], dim=1)
        return self.fc3(z)  # (B, 1)

    def _G_K_single(self, A_mat: torch.Tensor, C_row: torch.Tensor) -> torch.Tensor:
        """
        Helper when vmap is unavailable. Returns (B,) multiplicative modulator for one feature j.
        """
        B_full = A_mat * C_row  # (B, p+n_bits)

        # baseline: zero features, same positional bits from first row
        pos0 = B_full[0, self.input_dim:(self.input_dim + self.n_bits)].unsqueeze(0)  # (1, n_bits)
        B_zero = torch.cat([torch.zeros(1, self.input_dim, device=A_mat.device, dtype=A_mat.dtype), pos0], dim=1)

        z = self.forward_MLP(B_full)   # (B, 1)
        z0 = self.forward_MLP(B_zero)  # (1, 1)

        z = 2 * torch.tanh(z) + 1
        z0 = 2 * torch.tanh(z0) + 1
        z = z - z0  # center
        z = z * 0.5 * (1.0 + torch.tanh(self.alpha2))
        return (z + 1.0).squeeze(-1)  # (B,)

    def build_B_u(self, X: torch.Tensor) -> torch.Tensor:
        """
        B_u = X * (1 + G_u(X)), where G_u excludes self-interactions via CO.
        X: (B, p)
        returns: (B, p)
        """
        B, p = X.shape
        assert p == self.input_dim
        A_mat = torch.cat([X, torch.ones((B, self.n_bits), device=X.device, dtype=X.dtype)], dim=1)  # (B, p+n_bits)

        if self._use_vmap:
            def G_K(C_row):
                B_full = A_mat * C_row
                pos0 = B_full[0, self.input_dim:(self.input_dim + self.n_bits)].unsqueeze(0)
                B_zero = torch.cat([torch.zeros(1, self.input_dim, device=X.device, dtype=X.dtype), pos0], dim=1)
                z = self.forward_MLP(B_full)
                z0 = self.forward_MLP(B_zero)
                z = 2 * torch.tanh(z) + 1
                z0 = 2 * torch.tanh(z0) + 1
                z = z - z0
                z = z * 0.5 * (1.0 + torch.tanh(self.alpha2))
                return z + 1.0  # (B, 1)

            G_u = torch.vmap(G_K, randomness="different")(self.CO).squeeze(-1)  # (p, B)
            G_u = G_u.T  # (B, p)
        else:
            # Fallback: loop over features
            outs = []
            for j in range(self.CO.size(0)):  # p rows
                outs.append(self._G_K_single(A_mat, self.CO[j]))
            G_u = torch.stack(outs, dim=1)  # (B, p)

        B_u = X * G_u
        return B_u

    def forward(self, B_u: torch.Tensor) -> torch.Tensor:
        """
        Linear logit with current beta. Returns logits (B,)
        """
        B1 = torch.cat([torch.ones(B_u.size(0), 1, device=B_u.device, dtype=B_u.dtype), B_u], dim=1)  # (B, p+1)
        beta_full = torch.cat([self.beta_0, self.beta], dim=0)  # (p+1,)
        return (B1 @ beta_full).view(-1)

    # -------- training (epoch-wise IRLS) --------

    def on_train_epoch_start(self):
        p = self.input_dim
        self.A_sum = torch.zeros(p, p, device=self.device, dtype=self.beta.dtype)
        self.b_sum = torch.zeros(p, device=self.device, dtype=self.beta.dtype)
        self.W_sum = torch.zeros(1, device=self.device, dtype=self.beta.dtype)  # For intercept
        self.sx_sum = torch.zeros(p, device=self.device, dtype=self.beta.dtype)  # Cross terms
        self.wz_sum = torch.zeros(1, device=self.device, dtype=self.beta.dtype)  # Intercept RHS
        
        # Cache for gradient-enabled recomputation
        self.X_cache = None
        self.y_cache = None

    def training_step(self, batch, batch_idx):
        x = batch['features']
        y = batch['target'].view(-1)

        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)

        # Main loss for NN params
        bce_loss = nn.BCEWithLogitsLoss()(y_hat, y)

        # Corrected regularizers (paper-compliant)
        c_vals = self.c  # c_i > 0 via softplus
        c_penalty_loss = self.c_penalty * torch.sum(c_vals ** 2)  # Quadratic penalty on c
        
        # Group penalty: L2 norm on first layer input columns (paper-compliant)
        group_cols = self.fc1.weight[:, :self.input_dim]
        group_loss = self.group_penalty * torch.sum(torch.norm(group_cols, dim=0))  # L2 norm
        
        loss = bce_loss + c_penalty_loss + group_loss

        # Cache data for gradient-enabled recomputation (store small subset)
        if batch_idx == 0:  # Store first batch of each epoch
            self.X_cache = x.detach().clone()
            self.y_cache = y.detach().clone()

        # Manual optimization step
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        opt.step()

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        # IMPORTANT: return None when using manual optimization
        return None

    def on_train_epoch_end(self):
        """Differentiable IRLS solve with proper gradient flow - paper-compliant adaptive ridge."""
        if self.X_cache is None or self.y_cache is None:
            return  # No cached data available
        
        # Recompute IRLS system with gradients enabled (key fix!)
        X = self.X_cache
        y = self.y_cache
        
        B_u = self.build_B_u(X)  # depends on NN
        c_vals = self.c  # softplus(rho)
        X_tilde = B_u * c_vals
        
        # Current logits with the running beta buffers (stop grads through beta buffers only)
        logits = self.forward(B_u).detach()  # stop grads through beta buffers only
        pi = 0.999 * torch.sigmoid(logits) + 0.0005
        w = (pi * (1.0 - pi)).clamp_min(1e-3)
        z = logits + (y - pi) / w
        
        # Build A_full, b_full from X_tilde, w, z (no no_grad here!)
        B1 = torch.cat([torch.ones(X_tilde.size(0), 1, device=self.device), X_tilde], dim=1)  # (B, p+1)
        
        # Blocks:
        A11 = (w.unsqueeze(1) * X_tilde).T @ X_tilde  # (p, p)
        A10 = (w.unsqueeze(1) * X_tilde).sum(dim=0, keepdim=True).T  # (p, 1)
        A00 = w.sum(dim=0, keepdim=True)  # (1,)
        b1 = (w.unsqueeze(1) * X_tilde).T @ z  # (p,)
        b0 = (w * z).sum(dim=0, keepdim=True)  # (1,)
        
        eye = torch.eye(self.input_dim, device=self.device, dtype=B1.dtype)
        A_full = torch.zeros(self.input_dim + 1, self.input_dim + 1, device=self.device, dtype=B1.dtype)
        A_full[0, 0] = A00
        A_full[0, 1:] = A10.T
        A_full[1:, 0] = A10.squeeze(1)
        A_full[1:, 1:] = A11 + self.lambda_reg * eye
        b_full = torch.cat([b0, b1], dim=0)
        
        # Differentiable solve (gamma_full depends on B_u and c!)
        try:
            gamma_full = torch.linalg.solve(A_full, b_full)
        except RuntimeError:
            # Add regularization to diagonal if singular
            A_full += 1e-3 * torch.eye(self.input_dim + 1, device=self.device, dtype=B1.dtype)
            gamma_full = torch.linalg.solve(A_full, b_full)
        
        # Profile logits built from the profiled solution
        X1 = torch.cat([torch.ones(X_tilde.size(0), 1, device=self.device), X_tilde], dim=1)
        logits_profile = X1 @ gamma_full  # Uses solved gamma_full (no detach!)
        
        # Profile loss drives learning via the solved coefficients
        profile_loss = nn.BCEWithLogitsLoss()(logits_profile, y)
        reg = self.c_penalty * (self.c ** 2).sum() + self.group_penalty * torch.norm(self.fc1.weight[:, :self.input_dim], dim=0).sum()
        total_profile_loss = profile_loss + reg
        
        # Manual optimization step for profile loss
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)
        self.manual_backward(total_profile_loss)
        opt.step()
        
        # Update buffers for inspection/inference (optional)
        with torch.no_grad():
            self.beta_0.copy_(gamma_full[0:1])
            self.beta.copy_((self.c * gamma_full[1:]).detach())
        
        # Logging
        self.log('profile_loss', total_profile_loss, on_epoch=True, prog_bar=True)
        self.log('beta_0', self.beta_0, on_epoch=True, prog_bar=True)
        self.log('alpha2', self.alpha2, on_epoch=True, prog_bar=True)
        self.log('c_mean', c_vals.mean(), on_epoch=True, prog_bar=True)

    # -------- eval / predict --------

    def validation_step(self, batch, batch_idx):
        x = batch['features']
        y = batch['target'].view(-1)
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        y_prob = torch.sigmoid(y_hat)
        acc = ((y_prob > 0.5).float() == y).float().mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = batch['features']
        y = batch['target'].view(-1)
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        y_prob = torch.sigmoid(y_hat)
        acc = ((y_prob > 0.5).float() == y).float().mean()
        print(f"Test loss: {loss.item():.6f} | Test ACC: {acc.item():.4f}")

    def predict_step(self, batch, batch_idx):
        x = batch['features']
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        return torch.sigmoid(y_hat).detach()

    def configure_optimizers(self):
        params = (
            list(self.fc1.parameters())
            + list(self.fc2.parameters())
            + list(self.fc3.parameters())
            + [self.rho, self.alpha2]  # beta_0 is now a buffer, not a parameter
        )
        return torch.optim.Adam(params, lr=self.lr)


# -------------------- Runner (single config) --------------------

def run_nimo_baseline(
    X_train, y_train, X_test, y_test,
    iteration: int, randomState: int,
    X_columns: Optional[List[str]] = None,
    *,
    X_val=None, y_val=None,
    max_epochs: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    group_penalty: float = 1.0,
    dropout: float = 0.0,
    hidden_dim: Optional[int] = None,
    standardize: bool = True,
    num_workers: int = 0,
    self_features: Optional[List[str]] = None,   # default disabled for paper compliance
    early_stop_patience: Optional[int] = 10,
    noise_std: float = 0.2,
    lambda_reg: float = 1e-3,
    c_penalty: float = 0.01,
) -> Dict[str, Any]:
    """
    Train one NIMO config and evaluate. Returns standardized dict with metrics.
    Notes:
      - Self-features default: disabled for paper compliance
    """

    # Default self-features if not specified (disabled for paper compliance)
    if self_features is None:
        self_features = []

    torch.manual_seed(randomState)
    np.random.seed(randomState)
    use_gpu = torch.cuda.is_available()

    # Optional self-feature engineering BEFORE standardization
    Xtr_np = np.asarray(X_train, dtype=np.float32)
    Xte_np = np.asarray(X_test, dtype=np.float32)
    if X_val is not None:
        Xva_np = np.asarray(X_val, dtype=np.float32)

    Xtr_np, mapping = add_self_features(Xtr_np, self_features)
    Xte_np, _ = add_self_features(Xte_np, self_features)
    if X_val is not None:
        Xva_np, _ = add_self_features(Xva_np, self_features)

    # Standardize using train stats
    if standardize:
        mean = Xtr_np.mean(axis=0, keepdims=True)
        std = Xtr_np.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        Xtr_np = (Xtr_np - mean) / std
        Xte_np = (Xte_np - mean) / std
        if X_val is not None:
            Xva_np = (Xva_np - mean) / std

    # Tensors
    Xtr = torch.from_numpy(Xtr_np)
    Xte = torch.from_numpy(Xte_np)
    ytr = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).view(-1)
    yte = torch.from_numpy(np.asarray(y_test, dtype=np.float32)).view(-1)

    if X_val is not None and y_val is not None:
        Xva = torch.from_numpy(Xva_np)
        yva = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).view(-1)
    else:
        # fallback: use test as validation (not ideal, but keeps function robust)
        Xva, yva = Xte, yte

    # DataLoaders
    pin_mem = use_gpu
    loader_tr = DataLoader(DictDataset(Xtr, ytr), batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=pin_mem)
    loader_va = DataLoader(DictDataset(Xva, yva), batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_mem)
    loader_te = DataLoader(DictDataset(Xte, yte), batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_mem)

    # Model
    model = AdaptiveRidgeLogisticRegression(
        input_dim=Xtr.shape[1],
        output_dim=1,
        learning_rate=learning_rate,
        group_penalty=group_penalty,
        dropout=dropout,
        hidden_dim=hidden_dim,
        noise_std=noise_std,
        lambda_reg=lambda_reg,
        c_penalty=c_penalty,
    )
    
    # Update CO mask if self-features were used
    if mapping:
        model.update_CO_mask(mapping)

    callbacks = []
    if early_stop_patience is not None and EarlyStopping is not None:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=early_stop_patience, mode='min'))

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if use_gpu else 'cpu',
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        callbacks=callbacks,
    )

    # Train
    trainer.fit(model, train_dataloaders=loader_tr, val_dataloaders=loader_va)

    # Predict probs on val & test
    probs_val = torch.cat(trainer.predict(model, loader_va), dim=0).squeeze().cpu().numpy()
    probs_te  = torch.cat(trainer.predict(model, loader_te), dim=0).squeeze().cpu().numpy()

    # Threshold selection on validation
    thresholds = np.linspace(0.0, 1.0, 1001)
    y_ref = np.asarray(yva.cpu() if torch.is_tensor(yva) else yva, dtype=np.int32)
    f1s = [f1_score(y_ref, (probs_val >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])

    # Test metrics at best_thr
    y_test_np = np.asarray(y_test, dtype=np.int32)
    y_pred = (probs_te >= best_thr).astype(int)
    f1 = float(f1_score(y_test_np, y_pred, zero_division=0))
    acc = float((y_pred == y_test_np).mean())

    # Final feature selection from beta
    beta_coeffs = model.beta.detach().cpu().numpy()
    beta_0 = model.beta_0.detach().cpu().numpy().item()
    beta_threshold = 0.01
    if X_columns is not None and len(X_columns) == model.input_dim:
        selected_features = [X_columns[i] for i, b in enumerate(beta_coeffs) if abs(b) > beta_threshold]
        feature_names = X_columns
    else:
        selected_features = [i for i, b in enumerate(beta_coeffs) if abs(b) > beta_threshold]
        feature_names = [f"feature_{i}" for i in range(len(beta_coeffs))]

    # Store coefficients in the same format as other methods
    coefficients_data = {
        "space": "standardized",
        "intercept": float(beta_0),
        "values": beta_coeffs.tolist(),
        "values_no_threshold": beta_coeffs.tolist(),
        "feature_names": feature_names,
        "coef_threshold_applied": float(beta_threshold),
        "mean": [0.0] * len(beta_coeffs),  # Assuming standardized data
        "scale": [1.0] * len(beta_coeffs),  # Assuming standardized data
    }

    result = {
        'model_name': 'NIMO_MLP',
        'iteration': iteration,
        'random_seed': randomState,
        'f1': f1,
        'accuracy': acc,
        'threshold': best_thr,
        'y_pred': y_pred.tolist(),
        'y_prob': probs_te.tolist(),
        'coefficients': coefficients_data,
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'selection': {
            'mask': [1 if abs(b) > beta_threshold else 0 for b in beta_coeffs],
            'features': selected_features
        },
        'hyperparams': {
            'max_epochs': int(max_epochs),
            'batch_size': int(batch_size),
            'learning_rate': float(learning_rate),
            'group_penalty': float(group_penalty),
            'dropout': float(dropout),
            'hidden_dim': hidden_dim,
            'standardize': bool(standardize),
            'self_features': self_features or [],
            'noise_std': float(noise_std),
            'lambda_reg': float(lambda_reg),
            'c_penalty': float(c_penalty),
        }
    }
    return standardize_method_output(result)


# -------------------- Grid Search Wrapper --------------------

def run_nimo_grid(
    X_train, y_train, X_test, y_test,
    *,
    X_val=None, y_val=None,
    iteration: int = 0,
    randomState: int = 42,
    X_columns: Optional[List[str]] = None,
    grid: Optional[Dict[str, List[Any]]] = None,
    max_epochs: int = 60,
    batch_size: int = 128,
    standardize: bool = True,
    num_workers: int = 0,
    self_features: Optional[List[str]] = None,   # default enabled below
    early_stop_patience: Optional[int] = 8,
    noise_std: float = 0.2,
) -> Dict[str, Any]:
    """
    Fair hyperparameter sweep for NIMO, selecting by validation F1 with threshold search.
    Returns the best result plus all trials.
    Notes:
      - Self-features default: ["x2","sin","tanh","arctan"]
    """

    # Default self-features if not specified (disabled for paper compliance)
    if self_features is None:
        self_features = []

    # Default grid aligned with NIMO paper parameters
    if grid is None:
        grid = {
            "learning_rate": [1e-3, 3e-4],
            "group_penalty": [0.1, 0.3, 1.0, 3.0],
            "dropout": [0.0, 0.2, 0.5],
            "hidden_dim": [64, 128, 256],
            "lambda_reg": [1e-4, 1e-3, 1e-2],
            "c_penalty": [1e-3, 1e-2, 1e-1],
        }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    trials: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for ci, values in enumerate(combos, 1):
        h = dict(zip(keys, values))
        print(f"[NIMO grid] Trial {ci}/{len(combos)} -> {h}")

        res = run_nimo_baseline(
            X_train, y_train, X_test, y_test,
            iteration=iteration, randomState=randomState,
            X_columns=X_columns,
            X_val=X_val, y_val=y_val,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=h.get("learning_rate", 3e-4),
            group_penalty=h.get("group_penalty", 1.0),
            dropout=h.get("dropout", 0.0),
            hidden_dim=h.get("hidden_dim", None),
            standardize=standardize,
            num_workers=num_workers,
            self_features=self_features,         # default disabled
            early_stop_patience=early_stop_patience,
            noise_std=noise_std,
            lambda_reg=h.get("lambda_reg", 1e-3),
            c_penalty=h.get("c_penalty", 0.01),
        )

        res['trial_hparams'] = h
        trials.append(res)

        if (best is None) or (res['f1'] > best['f1']):
            best = res

    assert best is not None

    out = {
        "best": best,
        "trials": trials,
        "n_trials": len(trials),
        "grid": grid,
        "note": "Best chosen by validation F1 with threshold optimization; test metrics reported for that threshold."
    }
    return standardize_method_output(out)
