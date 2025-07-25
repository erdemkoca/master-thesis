"""
NIMO Baseline Variant - Adaptive Ridge Logistic Regression with Lightning
"""

import torch
import torch.nn as nn
import lightning as L
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import standardize_method_output
except ImportError as e:
    print(f"Import error in baseline.py: {e}")
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


def to_bin(x, n_bits):
    return np.array([int(b) for b in format(x, f'0{n_bits}b')]) - 0.5  # -0.5 to make it -0.5 and 0.5


class AdaptiveRidgeLogisticRegression(L.LightningModule):
    def __init__(self, input_dim, output_dim, learning_rate=3e-4, \
                 lasso_penalty=0.01, group_penalty=1.0, lasso_norm=0.5, group_norm=0.25, \
                 dropout=0, hidden_dim=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        self.lambda_ = 1.0
        self.lasso_penalty = lasso_penalty
        self.group_penalty = group_penalty
        self.lasso_norm = lasso_norm
        self.group_norm = group_norm
        self.dropout = dropout

        self.save_hyperparameters()

        # create binary map, for positional encoding
        self.n_bits = int(np.floor(np.log2(input_dim))) + 1
        BinMap = np.vstack([to_bin(i, self.n_bits) for i in range(1, input_dim + 1)])   # p x n_bits

        # figure out whether we have a GPU or not
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # move yourself (so that any .to(self.device_) below is a no-op once patched)
        super().to(self.device_)

        # create CO matrix and send to the right device
        CO = np.ones((input_dim, input_dim))
        np.fill_diagonal(CO, 0)
        CO = np.hstack([CO, BinMap])
        self.CO = torch.tensor(CO, dtype=torch.float32, device=self.device_)

        # MLP definition
        if hidden_dim is not None:
            self.fc1 = nn.Linear(self.input_dim + self.n_bits, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim + self.n_bits, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim + hidden_dim + self.n_bits, self.output_dim)
        else:
            self.fc1 = nn.Linear(self.input_dim + self.n_bits, 64)
            self.fc2 = nn.Linear(64 + self.n_bits, 128)
            self.fc3 = nn.Linear(128 + 64 + self.n_bits, self.output_dim)

        # self.dropout1 = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.register_buffer("beta", torch.randn(self.input_dim, device=self.device_))
        self.beta_0 = nn.Parameter(torch.randn(1, device=self.device_)*0.1)
        self.c = nn.Parameter(torch.ones(self.input_dim, device=self.device_) * 0.1)
        self.alpha2 = nn.Parameter(torch.tensor(2.0, device=self.device_))

    def forward_MLP(self, X):
        z1 = self.fc1(X)
        # z1 = self.dropout1(z1)
        z1 = torch.tanh(0.3 * (z1 + 0.2 * torch.randn_like(z1)))    # Noise injection
        z1 = torch.cat([z1, X[:, self.input_dim:(self.input_dim + self.n_bits)]], dim=1)
        z2 = torch.sin(2 * np.pi * self.fc2(z1))
        z2 = self.dropout2(z2)
        z = torch.cat([z2, z1], dim=1)
        return self.fc3(z)

    def build_B_u(self, X):
        # B_u = X + X * G_u = X * (1 + G_u)
        # X: Bxp, A_mat: Bx(p+n_bits)
        A_mat = torch.cat([X, torch.ones((X.size(0), self.n_bits), device=X.device)], dim=1)

        def G_K(C):
            B = A_mat * C
            B_zero = torch.cat([torch.zeros(1, self.input_dim, device=B.device), B[0, self.input_dim:(self.input_dim + self.n_bits)].unsqueeze(0)], dim=1)
            z = self.forward_MLP(B)
            z_zero = self.forward_MLP(B_zero)

            z = 2*(torch.tanh(z)) + 1
            z_zero = 2*(torch.tanh(z_zero)) + 1
            z = z - z_zero

            # z = torch.tanh(z)
            z = z * 0.5 * (1.0 + torch.tanh(self.alpha2))

            return z + 1

        G_u = torch.vmap(G_K, randomness="different")(self.CO).squeeze()
        G_u = G_u.T
        B_u = X * G_u

        return B_u

    def forward(self, B_u):
        B_u = torch.cat([torch.ones(B_u.size(0), 1, device=B_u.device), B_u], dim=1)
        beta = torch.cat([self.beta_0, self.beta])

        y_hat = B_u @ beta
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch['features'], batch['target']
        B_u = self.build_B_u(x)
        c = torch.abs(self.c)

        # iteratively update beta
        y_hat = self.forward(B_u)
        
        pi = 0.999 * torch.sigmoid(y_hat) + 0.0005      # Important for numerical stability

        w = pi * (1 - pi)
        dw = torch.diag(w.squeeze())
        dc = torch.diag(c.squeeze())
        
        X_tilde = B_u @ dc
        A = X_tilde.T @ dw @ X_tilde + torch.eye(self.input_dim, device=x.device)
        q = y_hat + (y - pi) / w
        b = X_tilde.T @ dw @ q
        gammma = torch.linalg.solve(A, b)
        new_beta = c * gammma

        B_u = torch.cat([torch.ones(B_u.size(0), 1, device=B_u.device), B_u], dim=1)
        beta = torch.cat([self.beta_0, new_beta])
        y_hat_new = B_u @ beta

        # compute losses
        bce_loss = torch.nn.BCEWithLogitsLoss()(y_hat_new, y)
        lasso_loss = self.lasso_penalty * torch.sum((self.c ** 2) ** self.lasso_norm)
        group_loss = self.group_penalty * torch.sum(torch.norm(self.fc1.weight[:, :self.input_dim], dim=0) ** self.group_norm)
        loss = bce_loss + lasso_loss + group_loss

        # update beta
        self.beta.data = new_beta

        # compute binary accuracy
        y_hat_new = torch.sigmoid(y_hat_new)
        pred = (y_hat_new > 0.5).float()
        acc = (pred == y).float().mean()

        # log
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_bce', bce_loss, on_epoch=True, prog_bar=True)
        self.log('train_lasso', lasso_loss, on_epoch=True, prog_bar=True)
        self.log('train_group', group_loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        return loss  
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.logger is not None:
            self.logger.experiment.add_histogram('beta', self.beta, global_step=self.global_step)
            self.logger.experiment.add_histogram('c', self.c, global_step=self.global_step)
        self.log('beta_0', self.beta_0, on_epoch=True, prog_bar=True)
        self.log('alpha2', self.alpha2, on_epoch=True, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['features'], batch['target']
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)

        # compute binary accuracy
        y_hat = torch.sigmoid(y_hat)
        pred = (y_hat > 0.5).float()
        acc = (pred == y).float().mean()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch['features'], batch['target']
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        
        # compute binary accuracy
        y_hat = torch.sigmoid(y_hat)
        pred = (y_hat > 0.5).float()
        acc = (pred == y).float().mean()

        print(f"Test loss: {loss}")
        print(f"Test ACC: {acc}")
    
    def predict_step(self, batch, batch_idx):
        x, y = batch['features'], batch['target']
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        print(f"Predict loss: {loss}")
        
        # return y_hat
        
        # compute binary accuracy
        y_hat = torch.sigmoid(y_hat)
        pred = (y_hat > 0.5).float()

        acc = (pred == y).float().mean()
        print(f"Predict ACC: {acc}")
        return pred

    def custom_prediction(self, x):
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.fc1.parameters()) + \
                                     list(self.fc2.parameters()) + \
                                     list(self.fc3.parameters()) + \
                                     [self.beta_0] + [self.c] + [self.alpha2], lr=self.lr)

        return optimizer


def run_nimo_baseline(
    X_train, y_train, X_test, y_test,
    rng, iteration, randomState, X_columns=None,
    *,
    max_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    lasso_penalty: float = 0.01,
    group_penalty: float = 1.0,
    lasso_norm: float = 0.5,
    group_norm: float = 0.25,
    dropout: float = 0.0,
    hidden_dim: int = None,
):
    """
    NIMO Baseline - Adaptive Ridge Logistic Regression with Lightning
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    # 1) DataLoader aufsetzen
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test,  dtype=torch.float32)
    yte = torch.tensor(y_test,  dtype=torch.float32)

    # Create custom dataset that returns dictionaries
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, features, targets):
            self.features = features
            self.targets = targets
            
        def __len__(self):
            return len(self.features)
            
        def __getitem__(self, idx):
            return {
                'features': self.features[idx],
                'target': self.targets[idx]  # Keep as scalar
            }

    ds_tr = DictDataset(Xtr, ytr)
    ds_va = DictDataset(Xte, yte)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size)

    # 2) Modell initialisieren
    model = AdaptiveRidgeLogisticRegression(
        input_dim = X_train.shape[1],
        output_dim= 1,
        learning_rate=learning_rate,
        lasso_penalty=lasso_penalty,
        group_penalty=group_penalty,
        lasso_norm=lasso_norm,
        group_norm=group_norm,
        dropout=dropout,
        hidden_dim=hidden_dim
    )

    # 3) Trainer konfigurieren
    trainer = L.Trainer(
        max_epochs = max_epochs,
        accelerator = device.type,
        devices = 1 if device.type=="cuda" else 1,  # Use 1 for CPU instead of None
        enable_checkpointing = False,
        logger = False,
        enable_model_summary = False,
    )

    # 4) Training
    trainer.fit(model, train_dataloaders=loader_tr, val_dataloaders=loader_va)

    # 5) Vorhersage auf Testset
    preds = trainer.predict(model, loader_va)  # Liste von 1-D Tensoren
    probs = torch.cat(preds, dim=0).squeeze().cpu().numpy()

    # 6) Threshold‑Optimierung
    thresholds = np.linspace(0,1,100)
    f1s = [f1_score(y_test, (probs>=t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    best_thr = thresholds[best_idx]

    # Feature selection based on beta coefficients
    beta_coeffs = model.beta.data.cpu().numpy()  # Get all beta coefficients
    beta_threshold = 0.01
    if X_columns is not None:
        selected_features = [X_columns[i] for i, beta in enumerate(beta_coeffs) if abs(beta) > beta_threshold]
    else:
        selected_features = [i for i, beta in enumerate(beta_coeffs) if abs(beta) > beta_threshold]

    result = {
      'model_name':      'nimo_baseline',
      'iteration':       iteration,
      'best_threshold':  float(best_thr),
      'best_f1':         float(f1s[best_idx]),
      'y_pred':          (probs>=best_thr).astype(int).tolist(),
      'y_prob':          probs.tolist(),
      'selected_features': selected_features,
      'method_has_selection': True,
      'n_selected': len(selected_features),
      # zum Debug: hyperparams
      'hp': {
        'epochs': max_epochs,
        'bs': batch_size,
        'lr': learning_rate,
        'lasso_pen': lasso_penalty,
        'group_pen': group_penalty
      }
    }
    
    return standardize_method_output(result) 