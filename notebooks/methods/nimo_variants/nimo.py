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
    """Transformer encoder that yields per-feature corrections using masked attention."""

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

        # Per-feature correction head
        self.corr_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, d, E) token sequence (no CLS)."""
        tokens = self.value_proj(x.unsqueeze(-1)) + self.feature_embed.unsqueeze(0)
        if self.use_binary_context and self.binary_proj is not None:
            tokens = tokens + self.binary_proj(self.binary_codes).unsqueeze(0)
        return tokens  # (B, d, E)

    def corrections_masked(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute g_j using a per-j attention mask that blocks any attention to key j.
        Ensures g_j = g_j(x_{-j}) at ALL layers.
        """
        B, d = x.size(0), self.d
        tokens = self._tokens(x)                      # (B, d, E)
        g = x.new_zeros(B, d)

        # Use additive mask with -inf on blocked positions (PyTorch supports float masks)
        # Shape must be (L, L) where L=d (since batch_first=True).
        for j in range(d):
            attn_mask = torch.zeros(d, d, device=x.device)
            attn_mask[:, j] = float("-inf")          # block all queries attending to key j
            # Optional: block self-attention too → attn_mask[torch.arange(d), torch.arange(d)] = float("-inf")

            enc = self.encoder(tokens, mask=attn_mask)   # (B, d, E)
            # Others-only pooling (mean over features except j)
            sum_all = enc.sum(dim=1, keepdim=True)       # (B, 1, E)
            others_sum = sum_all - enc[:, j:j+1, :]      # (B, 1, E)
            others_mean = others_sum / max(1, d - 1)     # (B, 1, E)
            g[:, j] = torch.tanh(self.corr_head(others_mean.squeeze(1)).squeeze(-1))
        return g  # (B, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-feature corrections g (B, d) using masked encoder."""
        return self.corrections_masked(x)


class NIMOTransformer(nn.Module):
    """Strict NIMO model with sparse β and transformer-based corrections.
    
    Implements the canonical NIMO formulation:
    η(x) = β₀ + Σⱼ βⱼxⱼ + Σⱼ βⱼxⱼgⱼ(x₋ⱼ)
    
    Key features:
    - Masked attention: gⱼ depends only on x₋ⱼ (strict no-self)
    - β-weighted interaction: corrections are multiplied by βⱼ
    - gⱼ(0) = 0 regularizer: drives corrections to zero at baseline
    - No residual terms: pure NIMO formulation
    """

    def __init__(
        self,
        d: int,
        *,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_scale: float = 0.4,
        use_binary_context: bool = True,
    ) -> None:
        super().__init__()
        self.d = d
        self.beta = nn.Parameter(torch.zeros(d + 1))  # [b0, b1..bd]
        self.out_scale = out_scale
        self.correction_net = TransformerCorrection(
            d,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_binary_context=use_binary_context,
        )

    def corrections(self, x: torch.Tensor, *, detach: bool = False) -> torch.Tensor:
        g_raw = self.correction_net(x)                # (B, d)
        g = self.out_scale * g_raw
        # Batch-center to encourage E[g]=0
        g = g - g.mean(dim=0, keepdim=True)
        if detach:
            g = g.detach()
        return g

    def predict_logits(self, x: torch.Tensor, use_correction: bool = True) -> torch.Tensor:
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        B = torch.cat([ones, x], dim=1)               # (B, d+1)
        eta_lin = B.matmul(self.beta)                 # β0 + Σ βj xj

        if not use_correction:
            return eta_lin

        g = self.corrections(x)                       # (B, d)
        beta_det = self.beta[1:].detach().unsqueeze(0) # (1, d)
        eta_corr = (x * g * beta_det).sum(dim=1)      # Σ βj xj g_j(x_{-j})
        return eta_lin + eta_corr

    def predict_proba(self, x: torch.Tensor, use_correction: bool = True) -> torch.Tensor:
        return torch.sigmoid(self.predict_logits(x, use_correction=use_correction))


@torch.no_grad()
def update_beta_irls(
    model: NIMOTransformer,
    X: torch.Tensor,
    y: torch.Tensor,
    lam_l2: float = 1e-3,
    tau_l1: float = 1e-3,
    use_correction: bool = True,
    eps: float = 1e-6,
    trust_region: float = 0.5,
) -> None:
    """Single IRLS step with elastic net style penalties and trust region."""
    beta_prev = model.beta.detach().clone()
    ones = torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype)
    B = torch.cat([ones, X], dim=1)

    logits_full = model.predict_logits(X, use_correction=use_correction)
    p = torch.sigmoid(logits_full)
    W = p * (1 - p) + eps
    z = logits_full + (y - p) / W

    # Subtract the nonlinear correction part from the target so the linear solve fits β on the "linearized" target.
    if use_correction:
        logits_lin = model.predict_logits(X, use_correction=False)
        nonlinear = logits_full - logits_lin          # = Σ βj xj g_j
        target = z - nonlinear
    else:
        target = z

    BW = B * W.unsqueeze(1)
    A = BW.t().matmul(B) + lam_l2 * torch.eye(B.shape[1], device=B.device, dtype=B.dtype)
    bvec = BW.t().matmul(target)
    beta_new = torch.linalg.solve(A, bvec)

    # soft-threshold β (lasso)
    beta_np = beta_new.detach().cpu().numpy()
    beta_np[1:] = np.sign(beta_np[1:]) * np.maximum(np.abs(beta_np[1:]) - tau_l1, 0.0)
    beta_tensor = torch.from_numpy(beta_np).to(B.device, dtype=B.dtype)

    # trust region
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
    lam_l2: float = 5e-2
    tau_l1: float = 5e-3
    lam_g: float = 2e-2
    lam_align: float = 1e-3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    T: int = 25
    nn_steps: int = 1
    warm_start_steps: int = 3
    use_binary_context: bool = True
    use_no_harm: bool = True
    eps_g: float = 1e-3
    tau_beta_report: float = 0.0
    trust_region: float = 0.5
    lam_g0: float = 1e-3  # Weight for g_j(0) = 0 regularizer


def _default_config_grid(d: int) -> List[Tuple[str, TrainingConfig]]:
    base = TrainingConfig()

    medium = replace(
        base,
        embed_dim=96,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        out_scale=0.45,
        lam_l2=2e-2,
        lam_g=1e-2,
        lam_align=5e-4,
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
        lam_l2=1e-2,
        lam_g=5e-3,
        lam_align=5e-4,
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
        lam_l2=1.5e-2,
        lam_g=7e-3,
        lam_align=5e-4,
        lr=4e-4,
        weight_decay=5e-5,
        T=50,
        nn_steps=3,
        warm_start_steps=5,
        trust_region=1.8,
    )

    configs: List[Tuple[str, TrainingConfig]] = [
        ("base", base),
        ("medium", medium),
        ("aggressive", aggressive),
    ]

    if d <= 4:
        lowdim = replace(
            base,
            embed_dim=72,
            num_layers=3,
            num_heads=3,
            dropout=0.05,
            out_scale=0.7,
            lam_l2=1e-2,
            lam_g=2e-3,
            lam_align=3e-4,
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
        use_binary_context=cfg.use_binary_context,
    ).to(device)

    opt = torch.optim.Adam(
        model.correction_net.parameters(),
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
            use_correction=False,
            trust_region=cfg.trust_region,
        )

    loss_history = []
    best_val_loss = float("inf")
    best_state: Optional[Tuple[dict, torch.Tensor]] = None
    stopped_early = False

    for t in range(cfg.T):
        model.eval()
        update_beta_irls(
            model,
            Xt,
            yt,
            lam_l2=cfg.lam_l2,
            tau_l1=cfg.tau_l1,
            use_correction=True,
            trust_region=cfg.trust_region,
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

            # g_j(0) regularizer: evaluate on zeros (standardized)
            zeros_like = torch.zeros_like(Xt)
            g_at_zero = model.correction_net.corrections_masked(zeros_like)
            reg_g0 = cfg.lam_g0 * g_at_zero.abs().mean()

            loss = bce + reg_g + align + reg_g0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.correction_net.parameters(), max_norm=1.0)
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
                    model.beta.detach().clone(),
                )

    if best_state is not None:
        correction_state, beta_state = best_state
        model.correction_net.load_state_dict(correction_state)
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
        corr_stats = {
            "eps_g": float(cfg.eps_g),
            "mean_abs_corr": np.abs(g_corr_val).mean(axis=0).tolist(),
            "activation_rate": (np.abs(g_corr_val) > cfg.eps_g).mean(axis=0).tolist(),
            "rel_mod": np.mean(np.abs(Xva * g_corr_val), axis=0).tolist(),
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
    # Effective coefficients: β_j * (1 + mean(g_j)) where g_j is β-weighted interaction
    beta_eff_raw = beta_raw * (1.0 + g_corr_train.mean(axis=0))

    feature_names = list(X_columns) if X_columns is not None else [f"feature_{i}" for i in range(len(beta_raw))]

    result = {
        "model_name": "nimo_transformer",
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