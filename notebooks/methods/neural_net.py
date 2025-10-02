# methods/neural_net.py

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import sys, os
import time

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

class ImprovedNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate, activation='relu'):
        super().__init__()
        self.activation = activation
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

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
    # Start timing
    start_time = time.perf_counter()
    
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    # GPU support for faster training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Feature scaling (like Lasso for fair comparison)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None

    # 1) Fourier‐Features erkennen
    fourier_feats = []
    if X_columns:
        for col in X_columns:
            parts = col.split('_')
            if len(parts)==4 and parts[0]=='sin' and parts[1]=='highfreq':
                idx  = int(parts[2])
                freq = float(parts[3])
                fourier_feats.append((freq, idx))

    X_t      = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_t      = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # 2) Fast Hyperparameter‐Grid - minimal grid for speed
    param_grid = {
        'hidden_dims': [[64], [64, 32]],  # Only 2 network sizes
        'dropout_rate': [0.3, 0.4],       # Only 2 dropout rates
        'activation': ['relu'],           # Only ReLU for simplicity
        'use_sawtooth': [False]           # Disable complex features initially
    }

    best_loss, best_params = float('inf'), {}
    kf = KFold(n_splits=2, shuffle=True, random_state=randomState)  # Reduced CV folds

    # 3) CV‐Tuning with Early Stopping (optimized for speed)
    for h_dims in param_grid['hidden_dims']:
        for d in param_grid['dropout_rate']:
            for act in param_grid['activation']:
                for s in param_grid['use_sawtooth']:
                    val_losses = []
                    for train_idx, val_idx in kf.split(X_t):
                        # Use improved architecture
                        if s or fourier_feats:  # Use advanced NN for complex features
                            model = AdvancedNN(
                                input_dim=X_t.shape[1],
                                hidden_dim=h_dims[0],
                                dropout_rate=d,
                                use_sawtooth=s,
                                fourier_features=fourier_feats
                            )
                        else:  # Use improved simple NN
                            model = ImprovedNN(
                                input_dim=X_t.shape[1],
                                hidden_dims=h_dims,
                                dropout_rate=d,
                                activation=act
                            )
                        
                        model = model.to(device)  # Move model to device
                        opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Balanced L2 regularization
                        crit = nn.BCEWithLogitsLoss()
                        
                        # Early stopping training with stricter patience
                        best_val_loss = float('inf')
                        patience = 5  # Reduced patience for faster training
                        patience_counter = 0
                        
                        for epoch in range(50):  # Reduced max epochs for CV
                            model.train()
                            opt.zero_grad()
                            loss = crit(model(X_t[train_idx]), y_t[train_idx])
                            loss.backward()
                            opt.step()
                            
                            # Validation check more frequently
                            if epoch % 2 == 0:  # Check every 2 epochs instead of 5
                                model.eval()
                                with torch.no_grad():
                                    val_loss = crit(model(X_t[val_idx]), y_t[val_idx]).item()
                                
                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                                
                                if patience_counter >= patience:
                                    break
                        
                        val_losses.append(best_val_loss)
                    
                    avg_val = np.mean(val_losses)
                    if avg_val < best_loss:
                        best_loss, best_params = avg_val, {
                            'hidden_dims': h_dims, 'dropout_rate': d, 
                            'activation': act, 'use_sawtooth': s
                        }

    # 4) Finales Modell trainieren mit Early Stopping (optimized)
    if best_params['use_sawtooth'] or fourier_feats:  # Use advanced NN for complex features
        model = AdvancedNN(
            input_dim=X_t.shape[1],
            hidden_dim=best_params['hidden_dims'][0],
            dropout_rate=best_params['dropout_rate'],
            use_sawtooth=best_params['use_sawtooth'],
            fourier_features=fourier_feats
        )
    else:  # Use improved simple NN
        model = ImprovedNN(
            input_dim=X_t.shape[1],
            hidden_dims=best_params['hidden_dims'],
            dropout_rate=best_params['dropout_rate'],
            activation=best_params['activation']
        )
    
    model = model.to(device)  # Move model to device
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Balanced L2 regularization
    crit = nn.BCEWithLogitsLoss()
    
    # Early stopping for final training with stricter patience
    best_val_loss = float('inf')
    patience = 5  # Much stricter patience
    patience_counter = 0
    
    for epoch in range(100):  # Reduced max epochs for final training
        model.train()
        opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward()
        opt.step()
        
        # Validation check more frequently
        if epoch % 2 == 0 and X_val is not None:  # Check every 2 epochs instead of 5
            model.eval()
            with torch.no_grad():
                val_loss = crit(model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device)), 
                               torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break

    # 5) Evaluation
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).squeeze().cpu().numpy()

    # Threshold optimization - STRICTLY use validation set only, never test set
    if X_val is not None and y_val is not None:
        with torch.no_grad():
            val_probs = torch.sigmoid(model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device))).squeeze().cpu().numpy()
        thresholds = np.linspace(0.000, 1.000, 1001)
        f1s = [f1_score(y_val, (val_probs>=t).astype(int), zero_division=0) for t in thresholds]
        best_i = int(np.argmax(f1s))
        best_thresh = thresholds[best_i]
        # Apply threshold to test set
        y_pred = (probs>=best_thresh).astype(int)
        best_f1 = f1_score(y_test, y_pred, zero_division=0)
    else:
        # NO FALLBACK - use fixed threshold to avoid data leakage
        # This ensures fair comparison and prevents overfitting to test set
        best_thresh = 0.5  # Fixed threshold
        y_pred = (probs>=best_thresh).astype(int)
        best_f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate training performance for overfitting analysis
    with torch.no_grad():
        train_probs = torch.sigmoid(model(X_t)).squeeze().cpu().numpy()
    train_pred = (train_probs >= best_thresh).astype(int)
    train_f1 = f1_score(y_train, train_pred, zero_division=0)

    print(f"[NeuralNet2] seed={randomState}, params={best_params}, f1={best_f1:.3f}")

    # 6) full_cols passend zur finalen Input‐Dim erzeugen
    full_cols = list(X_columns or [])
    for freq, idx in fourier_feats:
        full_cols += [f"sin_highfreq_{idx}_{freq}", f"cos_highfreq_{idx}_{freq}"]
    if best_params['use_sawtooth']:
        # für jede bisherige Spalte eine Sägezahn‐Variante
        full_cols += [col + '_saw' for col in full_cols]

    # 7) Feature‐Importances & Filter mit integer‐Suffix (balanced threshold)
    sel = []
    if hasattr(model, 'max1'):  # AdvancedNN has max1
        w = model.max1.lin.weight.data.abs().mean(dim=0).cpu().numpy()
        for i, v in enumerate(w):
            if v > 1e-2 and i < len(full_cols):  # Balanced threshold
                col = full_cols[i]
                try:
                    int(col.split('_')[-1])
                    sel.append(col)
                except ValueError:
                    pass
    elif hasattr(model, 'network'):  # ImprovedNN has network
        # For ImprovedNN, use the first layer weights
        first_layer = None
        for module in model.network:
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        
        if first_layer is not None:
            w = first_layer.weight.data.abs().mean(dim=0).cpu().numpy()
            for i, v in enumerate(w):
                if v > 1e-2 and i < len(full_cols):  # Balanced threshold
                    col = full_cols[i]
                    try:
                        int(col.split('_')[-1])
                        sel.append(col)
                    except ValueError:
                        pass

    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # End timing
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    result = {
        'model_name': 'NN',
        'iteration': iteration,
        'random_seed': randomState,
        'f1': best_f1,
        'accuracy': accuracy,
        'train_f1': train_f1,
        'train_accuracy': train_accuracy,
        'threshold': best_thresh,
        'y_pred': y_pred.tolist(),
        'y_prob': probs.tolist(),
        'selected_features': sel,
        'method_has_selection': False,
        'n_selected': len(sel),
        'hyperparams': best_params,
        
        # Timing information
        'execution_time': execution_time,
        'timing': {
            'total_seconds': execution_time,
            'start_time': start_time,
            'end_time': end_time
        }
    }
    return standardize_method_output(result)
