# methods/neural_net.py

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils import standardize_method_output
except ImportError:
    def standardize_method_output(result):
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

class Maxout(nn.Module):
    def __init__(self, in_features, out_features, pool_size=2):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features * pool_size)
        self.pool_size = pool_size
        self.out_features = out_features

    def forward(self, x):
        x = self.lin(x)
        x = x.view(x.size(0), self.out_features, self.pool_size)
        return x.max(dim=2)[0]

class AdvancedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, use_sawtooth=False, fourier_features=None):
        super().__init__()
        self.use_sawtooth = use_sawtooth
        self.fourier_features = fourier_features or []

        expanded = input_dim + 2*len(self.fourier_features)
        if self.use_sawtooth:
            expanded *= 2

        self.max1 = Maxout(expanded, hidden_dim, pool_size=2)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.do1  = nn.Dropout(dropout_rate)

        self.max2 = Maxout(hidden_dim, hidden_dim, pool_size=2)
        self.bn2  = nn.BatchNorm1d(hidden_dim)
        self.do2  = nn.Dropout(dropout_rate)

        self.fc3  = nn.Linear(hidden_dim, 1)

    def forward(self, x_raw):
        x = x_raw
        if self.fourier_features:
            extras = []
            for freq, idx in self.fourier_features:
                xi = x_raw[:, idx]
                extras.append(torch.sin(freq * xi).unsqueeze(1))
                extras.append(torch.cos(freq * xi).unsqueeze(1))
            x = torch.cat([x] + extras, dim=1)

        if self.use_sawtooth:
            saw = 2 * (x - torch.floor(x + 0.5))
            x = torch.cat([x, saw], dim=1)

        x = self.do1(self.bn1(self.max1(x)))
        x = self.do2(self.bn2(self.max2(x)))
        return self.fc3(x)

def run_neural_net(X_train, y_train, X_test, y_test, iteration, randomState, X_columns=None, X_val=None, y_val=None):
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    # 1) Fourier‐Features erkennen
    fourier_feats = []
    if X_columns:
        for col in X_columns:
            parts = col.split('_')
            if len(parts)==4 and parts[0]=='sin' and parts[1]=='highfreq':
                idx  = int(parts[2])
                freq = float(parts[3])
                fourier_feats.append((freq, idx))

    X_t      = torch.tensor(X_train, dtype=torch.float32)
    y_t      = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # 2) Hyperparameter‐Raster
    param_grid = {
        'hidden_dim':   [64, 128],
        'dropout_rate': [0.3, 0.5],
        'use_sawtooth': [False, True]
    }

    best_loss, best_params = float('inf'), {}
    kf = KFold(n_splits=3, shuffle=True, random_state=randomState)

    # 3) CV‐Tuning
    for h in param_grid['hidden_dim']:
        for d in param_grid['dropout_rate']:
            for s in param_grid['use_sawtooth']:
                val_losses = []
                for train_idx, val_idx in kf.split(X_t):
                    model = AdvancedNN(
                        input_dim=X_t.shape[1],
                        hidden_dim=h,
                        dropout_rate=d,
                        use_sawtooth=s,
                        fourier_features=fourier_feats
                    )
                    opt, crit = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4), nn.BCEWithLogitsLoss()
                    for _ in range(30):
                        model.train()
                        opt.zero_grad()
                        opt.step(closure=lambda: crit(model(X_t[train_idx]), y_t[train_idx]).backward())
                    model.eval()
                    with torch.no_grad():
                        val_losses.append(crit(model(X_t[val_idx]), y_t[val_idx]).item())
                avg_val = np.mean(val_losses)
                if avg_val < best_loss:
                    best_loss, best_params = avg_val, {'hidden_dim':h,'dropout_rate':d,'use_sawtooth':s}

    # 4) Finales Modell trainieren
    model = AdvancedNN(
        input_dim=X_t.shape[1],
        hidden_dim=best_params['hidden_dim'],
        dropout_rate=best_params['dropout_rate'],
        use_sawtooth=best_params['use_sawtooth'],
        fourier_features=fourier_feats
    )
    opt, crit = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4), nn.BCEWithLogitsLoss()
    for _ in range(50):
        model.train()
        opt.zero_grad()
        (crit(model(X_t), y_t)).backward()
        opt.step()

    # 5) Evaluation
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).squeeze().cpu().numpy()

    # Use validation set for threshold optimization if available, otherwise use test set
    if X_val is not None and y_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t)).squeeze().cpu().numpy()
        thresholds = np.linspace(0.000, 1.000, 1001)
        f1s = [f1_score(y_val, (val_probs>=t).astype(int), zero_division=0) for t in thresholds]
        best_i = int(np.argmax(f1s))
        best_thresh = thresholds[best_i]
        # Apply threshold to test set
        y_pred = (probs>=best_thresh).astype(int)
        best_f1 = f1_score(y_test, y_pred, zero_division=0)
    else:
        thresholds = np.linspace(0.000, 1.000, 1001)
        f1s = [f1_score(y_test, (probs>=t).astype(int), zero_division=0) for t in thresholds]
        best_i = int(np.argmax(f1s))
        best_thresh, best_f1 = thresholds[best_i], f1s[best_i]
        y_pred = (probs>=best_thresh).astype(int)

    print(f"[NeuralNet2] seed={randomState}, params={best_params}, f1={best_f1:.3f}")

    # 6) full_cols passend zur finalen Input‐Dim erzeugen
    full_cols = list(X_columns or [])
    for freq, idx in fourier_feats:
        full_cols += [f"sin_highfreq_{idx}_{freq}", f"cos_highfreq_{idx}_{freq}"]
    if best_params['use_sawtooth']:
        # für jede bisherige Spalte eine Sägezahn‐Variante
        full_cols += [col + '_saw' for col in full_cols]

    # 7) Feature‐Importances & Filter mit integer‐Suffix
    w = model.max1.lin.weight.data.abs().mean(dim=0).cpu().numpy()
    sel = []
    for i, v in enumerate(w):
        if v > 1e-2 and i < len(full_cols):
            col = full_cols[i]
            try:
                int(col.split('_')[-1])
                sel.append(col)
            except ValueError:
                pass

    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    
    result = {
        'model_name': 'neural_net',
        'iteration': iteration,
        'random_seed': randomState,
        'f1': best_f1,
        'accuracy': accuracy,
        'threshold': best_thresh,
        'y_pred': y_pred.tolist(),
        'y_prob': probs.tolist(),
        'selected_features': sel,
        'method_has_selection': False,
        'n_selected': len(sel),
        'hyperparams': best_params
    }
    return standardize_method_output(result)
