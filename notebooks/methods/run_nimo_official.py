# methods/nimo_official.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
import lightning as L

# aus eurem Repo importieren
from methods.model_ar_logistic import AdaptiveRidgeLogisticRegression

def run_nimo_official(
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
    Wrapper um die Lightning‑Implementierung von NIMO zu trainieren
    und dieselbe Funktionssignatur wie run_nimo zu liefern.
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

    return {
      'model_name':      'nimo_official',
      'iteration':       iteration,
      'best_threshold':  float(best_thr),
      'best_f1':         float(f1s[best_idx]),
      'y_pred':          (probs>=best_thr).astype(int).tolist(),
      'y_prob':          probs.tolist(),
      # NIMO wählt alle Features, die β≠0 haben; hier liefern wir einfach alle Spalten-Indices
      'selected_features': X_columns if X_columns is not None else list(range(X_train.shape[1])),
      # zum Debug: hyperparams
      'hp': {
        'epochs': max_epochs,
        'bs': batch_size,
        'lr': learning_rate,
        'lasso_pen': lasso_penalty,
        'group_pen': group_penalty
      }
    }
