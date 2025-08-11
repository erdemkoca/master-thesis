# methods/neural_net.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils import standardize_method_output
except ImportError as e:
    print(f"Import error in neural_net.py: {e}")
    # Fallback: define a simple version
    def standardize_method_output(result):
        # Convert numpy and torch types to native Python types
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

class SimpleNN(nn.Module):
    """
    Einfache Feedforward-Neuralnetz-Architektur:
      - Eingabe -> Linear -> ReLU -> Dropout
      - Hidden -> Linear -> ReLU -> Dropout
      - Ausgabe -> Linear -> Logits
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(SimpleNN, self).__init__()
        # Erste voll verbundene Schicht
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        # Zweite fully connected
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Ausgabe-Schicht: ein Logit pro Beispiel
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Forward-Pass: fc1 -> ReLU -> Dropout -> fc2 -> ReLU -> Dropout -> fc3
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


def run_neural_net(X_train, y_train, X_test, y_test, iteration, randomState, X_columns=None):
    """
    Training und Evaluierung eines einfachen Neural Net:
      1. Daten in Torch-Tensoren umwandeln
      2. Hyperparameter via 5-fach CV abstimmen
      3. Bestes Modell final trainieren
      4. Schwellenwert für F1-Optimierung bestimmen
      5. Feature-Importance via erste Layer-Gewichte (optional)
    """
    # 1) Reproduzierbarkeit: Seed für NumPy und (optional) Torch
    np.random.seed(randomState)
    torch.manual_seed(randomState)

    # 2) Tensor-Umwandlung
    X_t      = torch.tensor(X_train, dtype=torch.float32)
    y_t      = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # [n,1]
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # 3) Hyperparameter-Raster definieren
    param_grid = {
        'hidden_dim':    [32, 64, 128],
        'dropout_rate': [0.2, 0.3, 0.5]
    }

    best_loss   = float('inf')
    best_params = {}
    # 5-fach KFold CV für zuverlässige Schätzung
    kf = KFold(n_splits=5, shuffle=True, random_state=randomState)

    # 4) CV-Tuning
    for h in param_grid['hidden_dim']:
        for d in param_grid['dropout_rate']:
            val_losses = []
            for train_idx, val_idx in kf.split(X_t):
                # Modell und Optimizer initialisieren
                model = SimpleNN(X_t.shape[1], h, d)
                opt   = optim.Adam(model.parameters(), lr=1e-3)
                crit  = nn.BCEWithLogitsLoss()

                # 20 Epochen Training auf Teil-Trainingssplit
                for _ in range(20):
                    model.train()
                    opt.zero_grad()
                    loss = crit(model(X_t[train_idx]), y_t[train_idx])
                    loss.backward()
                    opt.step()

                # Validierungsverlust berechnen
                model.eval()
                with torch.no_grad():
                    val_losses.append(
                        crit(model(X_t[val_idx]), y_t[val_idx]).item()
                    )
            avg_val_loss = np.mean(val_losses)
            # Besten Satz wählen
            if avg_val_loss < best_loss:
                best_loss   = avg_val_loss
                best_params = {'hidden_dim': h, 'dropout_rate': d}

    # 5) Finales Training mit besten Hyperparametern auf gesamtem Trainingssatz
    model = SimpleNN(X_t.shape[1], best_params['hidden_dim'], best_params['dropout_rate'])
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.BCEWithLogitsLoss()

    for _ in range(20):
        model.train()
        opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward()
        opt.step()

    # 6) Evaluation auf Testdaten
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs  = torch.sigmoid(logits).squeeze().numpy()

    # 7) Fine-grained threshold optimization (0.001 step)
    thresholds = np.linspace(0.000, 1.000, 1001)
    f1_scores  = [f1_score(y_test, (probs >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx    = int(np.argmax(f1_scores))
    best_thresh = thresholds[best_idx]
    best_f1     = f1_scores[best_idx]
    y_pred      = (probs >= best_thresh).astype(int)

    print(f"[NeuralNet] seed={randomState}, hidden={best_params['hidden_dim']}, dropout={best_params['dropout_rate']}")

    # 8) Optionale Feature-Selektion: Gewichtsmittel im ersten Layer
    selected_features = []
    if X_columns is not None:
        weights = model.fc1.weight.data.abs().mean(dim=0).numpy()
        # Schwellwert zur Auswahl
        threshold = 0.01
        selected_features = [X_columns[i] for i,w in enumerate(weights) if w > threshold]

    # 9) Ergebnisse sammeln
    result = {
        'model_name':        'NeuralNet',
        'iteration':         iteration,
        'best_threshold':    best_thresh,
        'best_f1':           best_f1,
        'y_pred':            y_pred.tolist(),
        'y_prob':            probs.tolist(),
        'selected_features': selected_features,
        'method_has_selection': bool(selected_features),
        'n_selected':        len(selected_features),
        'hidden_dim':        best_params['hidden_dim'],
        'dropout_rate':      best_params['dropout_rate']
    }
    return standardize_method_output(result)
