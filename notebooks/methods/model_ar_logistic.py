"""
PyTorch Lightning wrapper for Adaptive Ridege Logistic Regression.
"""

import torch
import torch.nn as nn
import lightning as L
import numpy as np
import torch.nn.functional as F

def to_bin(x, n_bits):
    return np.array([int(b) for b in format(x, f'0{n_bits}b')]) - 0.5   # -0.5 to make it -0.5 and 0.5


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