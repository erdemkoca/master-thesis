"""Transformer-enhanced NIMO variant.

Hybrid model: sparse adaptive logistic regression backbone (IRLS updates)
combined with a transformer-based per-feature correction module.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Ensure local utils can be imported when executed from notebooks/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import standardize_method_output
except ImportError:  # pragma: no cover - fallback for ad-hoc execution
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


def _binary_code(index: int, n_bits: int) -> np.ndarray:
    """Return centered binary representation used as positional context."""
    return np.array([int(b) for b in format(index, f"0{n_bits}b")], dtype=np.float32) - 0.5


class TransformerCorrection(nn.Module):
    """Transformer encoder that yields per-feature corrections and a residual logit."""

    def __init__(
        self,
        d: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_binary_context: bool = True,
    ) -> None:
        super().__init__()
        self.d = d
        self.embed_dim = embed_dim
        self.use_binary_context = use_binary_context

        self.value_proj = nn.Linear(1, embed_dim)
        self.feature_embed = nn.Parameter(torch.randn(d, embed_dim) * 0.02)

        if use_binary_context:
            n_bits = int(np.floor(np.log2(d))) + 1
            codes = np.stack([_binary_code(i + 1, n_bits) for i in range(d)], axis=0)
            self.register_buffer("binary_codes", torch.tensor(codes, dtype=torch.float32))
            self.binary_proj = nn.Linear(n_bits, embed_dim, bias=False)
        else:
            self.register_buffer("binary_codes", None)
            self.binary_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.corr_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (corrections, residual) before scaling."""
        tokens = self.value_proj(x.unsqueeze(-1))
        tokens = tokens + self.feature_embed.unsqueeze(0)

        if self.use_binary_context and self.binary_proj is not None:
            codes = self.binary_proj(self.binary_codes).unsqueeze(0)
            tokens = tokens + codes

        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        encoded = self.encoder(tokens)
        cls_encoded = encoded[:, 0]
        feature_encoded = encoded[:, 1:]

        corr = torch.tanh(self.corr_head(feature_encoded).squeeze(-1))
        residual = torch.tanh(self.residual_head(cls_encoded)).squeeze(-1)
        return corr, residual


class NIMOTransformer(nn.Module):
    """Hybrid model with sparse β and transformer-based corrections."""

    def __init__(
        self,
        d: int,
        *,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_scale: float = 0.4,
        residual_scale: float = 0.3,
        use_binary_context: bool = True,
    ) -> None:
        super().__init__()
        self.d = d
        self.beta = nn.Parameter(torch.zeros(d + 1))  # [b0, b_1..b_d]
        self.out_scale = out_scale
        self.residual_scale = residual_scale
        self.correction_net = TransformerCorrection(
            d,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_binary_context=use_binary_context,
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(d, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def _raw_modulation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        corr_raw, residual_raw = self.correction_net(x)
        corr = self.out_scale * corr_raw
        residual = self.residual_scale * residual_raw
        return corr, residual

    def modulation(self, x: torch.Tensor, *, detach: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        corr, residual = self._raw_modulation(x)
        if detach:
            corr = corr.detach()
            residual = residual.detach()
        corr = corr - corr.mean(dim=0, keepdim=True)
        return corr, residual

    def corrections(self, x: torch.Tensor, *, detach: bool = False) -> torch.Tensor:
        corr, _ = self.modulation(x, detach=detach)
        return corr

    def _build_features(self, x: torch.Tensor, use_correction: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = x
        if use_correction:
            corr, residual = self.modulation(x)
            residual = residual + torch.sum(x * corr, dim=1)
            residual = residual + self.residual_mlp(x).squeeze(-1)
        else:
            residual = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        return feats, residual

    def predict_logits(self, x: torch.Tensor, use_correction: bool = True) -> torch.Tensor:
        feats, residual = self._build_features(x, use_correction)
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        B = torch.cat([ones, feats], dim=1)
        return B.matmul(self.beta) + residual

    def predict_proba(self, x: torch.Tensor, use_correction: bool = True) -> torch.Tensor:
        return torch.sigmoid(self.predict_logits(x, use_correction=use_correction))


@torch.no_grad()
def update_beta_irls(
    model: NIMOTransformer,
    X: torch.Tensor,
    y: torch.Tensor,
    lam_l2: float = 1e-3,
    tau_l1: float = 1e-3,
    tau_l1_max: float = 5e-2,
    use_correction: bool = True,
    eps: float = 1e-6,
    trust_region: float = 0.5,
    iteration: int = 0,
    max_iterations: int = 25,
    use_adaptive_l1: bool = True,
    use_hard_thresholding: bool = True,
    hard_threshold: float = 1e-6,
) -> None:
    """Single IRLS step with elastic net style penalties, adaptive L1, and hard thresholding."""
    beta_prev = model.beta.detach().clone()

    feats, residual = model._build_features(X, use_correction)
    ones = torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype)
    B = torch.cat([ones, feats], dim=1)
    logits = B.matmul(model.beta) + residual
    p = torch.sigmoid(logits)
    W = p * (1.0 - p) + eps
    z = logits + (y - p) / W

    target = z - residual

    BW = B * W.unsqueeze(1)
    A = BW.t().matmul(B) + lam_l2 * torch.eye(B.shape[1], device=B.device, dtype=B.dtype)
    bvec = BW.t().matmul(target)
    beta_new = torch.linalg.solve(A, bvec)

    # Adaptive L1 thresholding: increase penalty over time
    if use_adaptive_l1 and max_iterations > 0:
        progress = min(iteration / max_iterations, 1.0)
        tau_l1_curr = tau_l1 + (tau_l1_max - tau_l1) * progress
    else:
        tau_l1_curr = tau_l1

    beta_np = beta_new.detach().cpu().numpy()
    beta_np[1:] = np.sign(beta_np[1:]) * np.maximum(np.abs(beta_np[1:]) - tau_l1_curr, 0.0)

    # Hard thresholding: set very small coefficients to exactly zero
    if use_hard_thresholding:
        beta_np[1:] = np.where(np.abs(beta_np[1:]) < hard_threshold, 0.0, beta_np[1:])

    beta_tensor = torch.from_numpy(beta_np).to(B.device, dtype=B.dtype)
    delta = beta_tensor - beta_prev
    delta_norm = torch.norm(delta)
    if delta_norm > trust_region:
        beta_tensor = beta_prev + delta * (trust_region / (delta_norm + 1e-12))

    model.beta.data.copy_(beta_tensor)


@dataclass
class TrainingConfig:
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    out_scale: float = 0.4
    residual_scale: float = 0.3
    lam_l2: float = 5e-2
    tau_l1: float = 2e-2  # Increased from 5e-3 for stronger sparsity
    tau_l1_max: float = 5e-2  # Maximum L1 penalty for adaptive thresholding
    lam_g: float = 2e-2
    lam_align: float = 1e-3
    lam_residual: float = 5e-4
    lam_sparse_corr: float = 1e-3  # Additional sparsity regularization for corrections
    lr: float = 1e-3
    weight_decay: float = 1e-4
    T: int = 25
    nn_steps: int = 1
    warm_start_steps: int = 3
    use_binary_context: bool = True
    use_no_harm: bool = True
    eps_g: float = 1e-3
    tau_beta_report: float = 0.0
    trust_region: float = 1.0  # Increased to allow more aggressive updates
    use_adaptive_l1: bool = True  # Enable adaptive L1 thresholding
    use_hard_thresholding: bool = True  # Enable hard thresholding for exact zeros
    hard_threshold: float = 1e-6  # Threshold below which coefficients are set to exactly zero


def _default_config_grid(d: int) -> List[Tuple[str, TrainingConfig]]:
    base = TrainingConfig()

    medium = replace(
        base,
        embed_dim=96,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        out_scale=0.45,
        residual_scale=0.4,
        lam_l2=2e-2,
        lam_g=1e-2,
        lam_align=5e-4,
        lam_residual=5e-4,
        lr=5e-4,
        weight_decay=5e-5,
        T=40,
        nn_steps=2,
        warm_start_steps=5,
        trust_region=1.5,
    )

    aggressive_heads = 8 if d <= 12 else 8
    aggressive_embed = 128 if d <= 20 else 160
    aggressive = replace(
        base,
        embed_dim=aggressive_embed,
        num_layers=4,
        num_heads=aggressive_heads,
        dropout=0.15,
        out_scale=0.55,
        residual_scale=0.5,
        lam_l2=1e-2,
        lam_g=5e-3,
        lam_align=5e-4,
        lam_residual=1e-3,
        lr=3e-4,
        weight_decay=5e-5,
        T=60,
        nn_steps=3,
        warm_start_steps=6,
        trust_region=2.0,
    )

    residual_heads = 6 if d >= 16 else 4
    residual_embed = 96 if residual_heads == 4 else 120
    residual = replace(
        base,
        embed_dim=residual_embed,
        num_layers=3,
        num_heads=residual_heads,
        dropout=0.2,
        out_scale=0.5,
        residual_scale=0.6,
        lam_l2=1.5e-2,
        lam_g=7e-3,
        lam_align=5e-4,
        lam_residual=1e-3,
        lr=4e-4,
        weight_decay=5e-5,
        T=50,
        nn_steps=3,
        warm_start_steps=5,
        trust_region=1.8,
    )

    # Add sparsity-focused configurations
    sparse_base = replace(
        base,
        tau_l1=3e-2,
        tau_l1_max=8e-2,
        lam_sparse_corr=2e-3,
        trust_region=1.5,
        use_adaptive_l1=True,
        use_hard_thresholding=True,
        hard_threshold=1e-5,
    )
    
    sparse_aggressive = replace(
        aggressive,
        tau_l1=4e-2,
        tau_l1_max=1e-1,
        lam_sparse_corr=3e-3,
        trust_region=2.0,
        use_adaptive_l1=True,
        use_hard_thresholding=True,
        hard_threshold=1e-6,
    )

    configs: List[Tuple[str, TrainingConfig]] = [
        ("base", base),
        ("medium", medium),
        ("aggressive", aggressive),
        ("sparse_base", sparse_base),
        ("sparse_aggressive", sparse_aggressive),
    ]

    if d <= 4:
        lowdim = replace(
            base,
            embed_dim=72,
            num_layers=3,
            num_heads=3,
            dropout=0.05,
            out_scale=0.7,
            residual_scale=0.65,
            lam_l2=1e-2,
            lam_g=2e-3,
            lam_align=3e-4,
            lam_residual=5e-4,
            lr=4e-4,
            weight_decay=5e-5,
            T=60,
            nn_steps=3,
            warm_start_steps=5,
            trust_region=1.5,
        )
        configs.append(("lowdim_nonlin", lowdim))

    if d >= 8:
        configs.append(("residual", residual))

    return configs


def _train_single(
    cfg: TrainingConfig,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    iteration: int,
    randomState: int,
    X_columns,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
):
    device = torch.device("cpu")
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    Xva = scaler.transform(X_val) if X_val is not None else None

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    XteT = torch.tensor(Xte, dtype=torch.float32, device=device)
    yteT = torch.tensor(y_test, dtype=torch.float32, device=device)
    if X_val is not None:
        XvaT = torch.tensor(Xva, dtype=torch.float32, device=device)
        yvaT = torch.tensor(y_val, dtype=torch.float32, device=device)
    else:
        XvaT = yvaT = None

    d = Xt.shape[1]
    model = NIMOTransformer(
        d,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        out_scale=cfg.out_scale,
        residual_scale=cfg.residual_scale,
        use_binary_context=cfg.use_binary_context,
    ).to(device)

    opt = torch.optim.Adam(
        list(model.correction_net.parameters()) + list(model.residual_mlp.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    for _ in range(cfg.warm_start_steps):
        update_beta_irls(
            model,
            Xt,
            yt,
            lam_l2=cfg.lam_l2,
            tau_l1=cfg.tau_l1,
            tau_l1_max=cfg.tau_l1_max,
            use_correction=False,
            trust_region=cfg.trust_region,
            iteration=0,
            max_iterations=cfg.T,
            use_adaptive_l1=cfg.use_adaptive_l1,
            use_hard_thresholding=cfg.use_hard_thresholding,
            hard_threshold=cfg.hard_threshold,
        )

    loss_history = []
    best_val_loss = float("inf")
    best_state: Optional[Tuple[dict, dict, torch.Tensor]] = None
    stopped_early = False

    for t in range(cfg.T):
        model.eval()
        update_beta_irls(
            model,
            Xt,
            yt,
            lam_l2=cfg.lam_l2,
            tau_l1=cfg.tau_l1,
            tau_l1_max=cfg.tau_l1_max,
            use_correction=True,
            trust_region=cfg.trust_region,
            iteration=t,
            max_iterations=cfg.T,
            use_adaptive_l1=cfg.use_adaptive_l1,
            use_hard_thresholding=cfg.use_hard_thresholding,
            hard_threshold=cfg.hard_threshold,
        )

        lam_g_curr = cfg.lam_g * (0.3 + 0.7 * (1.0 - t / max(1, cfg.T - 1)))

        for _ in range(cfg.nn_steps):
            model.train()
            opt.zero_grad()
            logits = model.predict_logits(Xt, use_correction=True)
            bce = F.binary_cross_entropy_with_logits(logits, yt)

            with torch.no_grad():
                g_corr = model.corrections(Xt)
            reg_g = lam_g_curr * g_corr.abs().mean()
            align = cfg.lam_align * torch.mean(torch.abs((Xt * g_corr).mean(dim=0)))
            residual_out = model.residual_mlp(Xt).squeeze(-1)
            reg_residual = cfg.lam_residual * residual_out.abs().mean()
            
            # Additional sparsity regularization for corrections
            reg_sparse_corr = cfg.lam_sparse_corr * torch.sum(torch.abs(g_corr))

            loss = bce + reg_g + align + reg_residual + reg_sparse_corr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.correction_net.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.residual_mlp.parameters(), max_norm=1.0)
            opt.step()

        loss_history.append(float(loss.item()))
        if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < 1e-4:
            stopped_early = True
            break

        if XvaT is not None:
            with torch.no_grad():
                val_logits = model.predict_logits(XvaT, use_correction=True)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, yvaT).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = (
                    model.correction_net.state_dict(),
                    model.residual_mlp.state_dict(),
                    model.beta.detach().clone(),
                )

    if best_state is not None:
        correction_state, residual_mlp_state, beta_state = best_state
        model.correction_net.load_state_dict(correction_state)
        model.residual_mlp.load_state_dict(residual_mlp_state)
        model.beta.data.copy_(beta_state)

    model.eval()
    use_correction_final = True
    no_harm_choice = "on"

    if cfg.use_no_harm and XvaT is not None:
        with torch.no_grad():
            prob_val_on = model.predict_proba(XvaT, use_correction=True).cpu().numpy()
            prob_val_off = model.predict_proba(XvaT, use_correction=False).cpu().numpy()
        grid = np.linspace(0.0, 1.0, 501)
        f1_on = max(f1_score(y_val, (prob_val_on >= thr).astype(int), zero_division=0) for thr in grid)
        f1_off = max(f1_score(y_val, (prob_val_off >= thr).astype(int), zero_division=0) for thr in grid)
        if f1_on < f1_off:
            use_correction_final = False
            no_harm_choice = "off"

    with torch.no_grad():
        prob_test = model.predict_proba(XteT, use_correction=use_correction_final).cpu().numpy()
        prob_val = (
            model.predict_proba(XvaT, use_correction=use_correction_final).cpu().numpy()
            if XvaT is not None
            else None
        )

    grid = np.linspace(0.0, 1.0, 501)
    if prob_val is not None:
        f1_grid = [f1_score(y_val, (prob_val >= thr).astype(int), zero_division=0) for thr in grid]
    else:
        f1_grid = [f1_score(y_test, (prob_test >= thr).astype(int), zero_division=0) for thr in grid]
    best_idx = int(np.argmax(f1_grid))
    threshold = float(grid[best_idx])

    y_pred = (prob_test >= threshold).astype(int)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))

    if prob_val is not None:
        val_pred = (prob_val >= threshold).astype(int)
        val_metrics = {
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_val, val_pred)),
        }
    else:
        val_metrics = {"f1": None, "accuracy": None}

    def _decomposition(X_np: np.ndarray):
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            eta_lin = model.predict_logits(X_t, use_correction=False).cpu().numpy()
            eta_full = model.predict_logits(X_t, use_correction=use_correction_final).cpu().numpy()
        eta_corr = eta_full - eta_lin
        var_full = np.var(eta_full)
        return {
            "corr_mean_abs": float(np.mean(np.abs(eta_corr))),
            "corr_var_share": float(np.var(eta_corr) / var_full) if var_full > 1e-12 else 0.0,
            "lin_full_corr": float(np.corrcoef(eta_lin, eta_full)[0, 1]) if np.std(eta_lin) > 0 and np.std(eta_full) > 0 else 0.0,
        }

    decomp_val = _decomposition(Xva) if X_val is not None else None
    decomp_test = _decomposition(Xte)

    corr_stats = None
    if X_val is not None:
        with torch.no_grad():
            g_corr_val = model.corrections(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy()
            residual_val = model.residual_mlp(
                torch.tensor(Xva, dtype=torch.float32, device=device)
            ).squeeze(-1).cpu().numpy()
        corr_stats = {
            "eps_g": float(cfg.eps_g),
            "mean_abs_corr": np.abs(g_corr_val).mean(axis=0).tolist(),
            "activation_rate": (np.abs(g_corr_val) > cfg.eps_g).mean(axis=0).tolist(),
            "rel_mod": np.mean(np.abs(Xva * g_corr_val), axis=0).tolist(),
            "residual_mean_abs": float(np.mean(np.abs(residual_val))),
        }
    else:
        g_corr_val = None

    beta_std = model.beta.detach().cpu().numpy()[1:]
    b0_std = float(model.beta.detach().cpu().numpy()[0])

    scale = scaler.scale_
    mean = scaler.mean_

    beta_raw = beta_std / scale
    b0_raw = b0_std - float(np.dot(beta_raw, mean))

    beta_for_sel = beta_raw.copy()
    if cfg.tau_beta_report > 0:
        beta_for_sel[np.abs(beta_for_sel) < cfg.tau_beta_report] = 0.0

    selected_mask = (np.abs(beta_for_sel) > 0).astype(int).tolist()
    selected_features = (
        [X_columns[i] for i, m in enumerate(selected_mask) if m]
        if X_columns is not None
        else [i for i, m in enumerate(selected_mask) if m]
    )

    with torch.no_grad():
        g_corr_train = model.corrections(
            torch.tensor(Xtr, dtype=torch.float32, device=device), detach=True
        ).cpu().numpy()
    beta_eff_raw = beta_raw * (1.0 + g_corr_train.mean(axis=0))

    feature_names = list(X_columns) if X_columns is not None else [f"feature_{i}" for i in range(len(beta_raw))]

    result = {
        "model_name": "NIMO",
        "iteration": iteration,
        "random_seed": randomState,
        "f1": f1,
        "accuracy": acc,
        "threshold": threshold,
        "y_pred": y_pred.tolist(),
        "y_prob": prob_test.tolist(),
        "metrics": {"f1": f1, "accuracy": acc},
        "selection": {"mask": selected_mask, "features": selected_features},
        "selected_features": selected_features,
        "n_selected": len(selected_features),
        "coefficients": {
            "intercept": b0_raw,
            "values": beta_raw.tolist(),
            "values_effective": beta_eff_raw.tolist(),
            "feature_names": feature_names,
            "coef_threshold_applied": float(cfg.tau_beta_report),
            "scale": scale.tolist(),
            "mean": mean.tolist(),
        },
        "correction_stats_val": corr_stats,
        "decomposition_val": decomp_val,
        "decomposition_test": decomp_test,
        "no_harm_val": None if X_val is None else {
            "no_harm_choice": no_harm_choice,
            "f1_on": f1_on if cfg.use_no_harm else None,
            "f1_off": f1_off if cfg.use_no_harm else None,
        },
        "training": {
            "loss_history": loss_history,
            "n_iters": len(loss_history),
            "stopped_early": bool(stopped_early),
        },
        "hyperparams": {
            **cfg.__dict__,
            "use_correction_final": bool(use_correction_final),
        },
        "val_metrics": val_metrics,
    }

    if g_corr_val is not None:
        result["no_harm_val"].update(
            {
                "g_mean_abs": float(np.mean(np.abs(g_corr_val))),
                "lin_full_corr": decomp_val["lin_full_corr"] if decomp_val else None,
            }
        )

    return standardize_method_output(result), val_metrics


def run_nimo(
    X_train, y_train, X_test, y_test,
    iteration, randomState, X_columns=None,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[TrainingConfig] = None,
    config_search: Optional[List[Tuple[str, TrainingConfig]]] = None,
    return_all: bool = False,
):
    """Run transformer-based NIMO with optional hyperparameter search."""

    def _with_label(res, val_metrics, label):
        if "val_metrics" not in res:
            res["val_metrics"] = val_metrics
        res["config_label"] = label
        res.setdefault("config_candidates", [])
        res["hyperparams"]["config_label"] = label
        return res

    if config is not None:
        res, val_metrics = _train_single(
            config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            iteration=iteration,
            randomState=randomState,
            X_columns=X_columns,
            X_val=X_val,
            y_val=y_val,
        )
        candidate_summary = [{
            "label": "provided",
            "val_f1": val_metrics.get("f1") if val_metrics else None,
            "val_accuracy": val_metrics.get("accuracy") if val_metrics else None,
            "test_f1": res.get("f1"),
        }]
        res = _with_label(res, val_metrics, "provided")
        res["config_candidates"] = candidate_summary
        return (res, candidate_summary) if return_all else res

    d = X_train.shape[1]
    candidates = config_search or _default_config_grid(d)

    summaries = []
    best_res = best_metrics = None
    best_label = ""
    best_score = float('-inf')

    for label, cfg in candidates:
        try:
            res, val_metrics = _train_single(
                cfg,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                iteration=iteration,
                randomState=randomState,
                X_columns=X_columns,
                X_val=X_val,
                y_val=y_val,
            )
            err_msg = None
        except Exception as exc:
            summaries.append({
                "label": label,
                "error": str(exc),
                "val_f1": None,
                "val_accuracy": None,
                "test_f1": None,
            })
            continue

        val_f1 = None
        if val_metrics and val_metrics.get("f1") is not None:
            val_f1 = val_metrics["f1"]
        score = val_f1 if val_f1 is not None else res.get("f1", 0.0)

        summaries.append({
            "label": label,
            "val_f1": val_f1,
            "val_accuracy": val_metrics.get("accuracy") if val_metrics else None,
            "test_f1": res.get("f1"),
            "error": err_msg,
        })

        if score > best_score:
            best_res = res
            best_metrics = val_metrics
            best_label = label
            best_score = score

    if best_res is None:
        raise RuntimeError("All NIMO transformer config candidates failed")

    best_res = _with_label(best_res, best_metrics, best_label)
    best_res["config_candidates"] = summaries

    return (best_res, summaries) if return_all else best_res