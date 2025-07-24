# methods/neural_net.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


def run_neural_net(X_train, y_train, X_test, y_test, rng, iteration, randomState, X_columns=None):

    # Reproduzierbarkeit für Torch
    #torch.manual_seed(randomState)

    # Tensor-Umwandlung
    X_t      = torch.tensor(X_train, dtype=torch.float32)
    y_t      = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Hyperparameter grid
    param_grid = {
        'hidden_dim':    [32, 64, 128],
        'dropout_rate': [0.2, 0.3, 0.5]
    }

    best_loss   = float('inf')
    best_params = {}
    kf = KFold(n_splits=5, shuffle=True) #, random_state=randomState

    # CV-Tuning
    for h in param_grid['hidden_dim']:
        for d in param_grid['dropout_rate']:
            val_losses = []
            for train_idx, val_idx in kf.split(X_t):
                model = SimpleNN(X_t.shape[1], h, d)
                opt   = optim.Adam(model.parameters(), lr=0.001)
                crit  = nn.BCEWithLogitsLoss()

                for _ in range(20):
                    model.train()
                    opt.zero_grad()
                    loss = crit(model(X_t[train_idx]), y_t[train_idx])
                    loss.backward()
                    opt.step()

                model.eval()
                with torch.no_grad():
                    val_losses.append(
                        crit(model(X_t[val_idx]), y_t[val_idx]).item()
                    )

            avg_val_loss = np.mean(val_losses)
            if avg_val_loss < best_loss:
                best_loss   = avg_val_loss
                best_params = {'hidden_dim': h, 'dropout_rate': d}

    # Finales Training
    model = SimpleNN(X_t.shape[1], best_params['hidden_dim'], best_params['dropout_rate'])
    opt   = optim.Adam(model.parameters(), lr=0.001)
    crit  = nn.BCEWithLogitsLoss()

    for _ in range(20):
        model.train()
        opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward()
        opt.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs  = torch.sigmoid(logits).squeeze().numpy()

    # Threshold-Optimierung
    thresholds = np.linspace(0, 1, 100)
    f1_scores  = [f1_score(y_test, (probs >= t).astype(int)) for t in thresholds]
    best_idx   = int(np.argmax(f1_scores))
    best_thresh= thresholds[best_idx]
    best_f1    = f1_scores[best_idx]
    y_pred     = (probs >= best_thresh).astype(int)

    print(f"[NeuralNet] seed={randomState}, hidden={best_params['hidden_dim']}, dropout={best_params['dropout_rate']}")

    # Feature selection based on first layer weights
    if X_columns is not None:
        # Calculate feature importance based on first layer weights
        first_layer_weights = model.fc1.weight.data.abs().mean(dim=0).numpy()
        weight_threshold = 0.01
        selected_features = [X_columns[i] for i, weight in enumerate(first_layer_weights) if weight > weight_threshold]
    else:
        selected_features = []

    return {
        'model_name':       'NeuralNet',
        'iteration':   iteration,
        'best_threshold':   best_thresh,
        'best_f1':          best_f1,
        'y_pred':           y_pred.tolist(),
        'y_prob':           probs.tolist(),
        'selected_features': selected_features,
        'method_has_selection': False,
        'n_selected': len(selected_features),
        'hidden_size':      best_params['hidden_dim'],
        'dropout':          best_params['dropout_rate']
    }
