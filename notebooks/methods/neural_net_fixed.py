# methods/neural_net_fixed.py

import numpy as np
import pandas as pd
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from utils import standardize_method_output
except ImportError as e:
    print(f"Import error in neural_net_fixed.py: {e}")
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
            elif hasattr(v, 'item'):  # torch tensors
                converted[k] = v.item()
            else:
                converted[k] = v
        return converted

def calculate_hash(arr):
    """Calculate MD5 hash of array for debugging"""
    if isinstance(arr, np.ndarray):
        return hashlib.md5(arr.tobytes()).hexdigest()[:8]
    elif torch.is_tensor(arr):
        return hashlib.md5(arr.detach().cpu().numpy().tobytes()).hexdigest()[:8]
    else:
        return hashlib.md5(str(arr).encode()).hexdigest()[:8]

class SimpleNN(nn.Module):
    """
    Improved Neural Network Architecture with regularization:
      - Input -> Linear -> ReLU -> Dropout
      - Hidden -> Linear -> ReLU -> Dropout  
      - Output -> Linear -> Sigmoid
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Smaller second layer
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x))  # Output probabilities directly

def run_neural_net(X_train, y_train, X_test, y_test, rng, iteration, randomState, X_columns=None):
    """
    FIXED Neural Network with proper randomness and validation-based threshold optimization
    """
    # Set seeds for reproducibility
    np.random.seed(randomState)
    torch.manual_seed(randomState)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(randomState)

    # Split training data into train/val for threshold optimization
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=randomState)
    train_idx, val_idx = next(splitter.split(X_train, y_train))
    
    X_train_split = X_train[train_idx]
    y_train_split = y_train[train_idx]
    X_val_split = X_train[val_idx]
    y_val_split = y_train[val_idx]

    # Preprocessing
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train_split))
    X_val_processed = scaler.transform(imputer.transform(X_val_split))
    X_test_processed = scaler.transform(imputer.transform(X_test))

    # Convert to tensors
    X_train_t = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_split, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val_processed, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_split, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_processed, dtype=torch.float32)

    # Hyperparameter grid
    param_grid = {
        'hidden_dim': [16, 32, 64],
        'dropout_rate': [0.2, 0.3, 0.5],
        'lr': [1e-4, 1e-3, 1e-2],
        'weight_decay': [1e-5, 1e-4, 1e-3]
    }

    best_val_f1 = 0
    best_params = {}
    best_model_state = None

    # Manual hyperparameter search
    for hidden_dim in param_grid['hidden_dim']:
        for dropout_rate in param_grid['dropout_rate']:
            for lr in param_grid['lr']:
                for weight_decay in param_grid['weight_decay']:
                    # Create and train model
                    model = SimpleNN(X_train_t.shape[1], hidden_dim, dropout_rate)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    criterion = nn.BCELoss()

                    # Training loop
                    model.train()
                    for epoch in range(50):  # More epochs for better convergence
                        optimizer.zero_grad()
                        outputs = model(X_train_t)
                        loss = criterion(outputs, y_train_t)
                        loss.backward()
                        optimizer.step()

                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_probs = model(X_val_t).squeeze().numpy()
                        val_pred = (val_probs >= 0.5).astype(int)
                        val_f1 = f1_score(y_val_split, val_pred, zero_division=0)

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_params = {
                            'hidden_dim': hidden_dim,
                            'dropout_rate': dropout_rate,
                            'lr': lr,
                            'weight_decay': weight_decay
                        }
                        best_model_state = model.state_dict().copy()

    # Train final model with best parameters
    final_model = SimpleNN(X_train_t.shape[1], best_params['hidden_dim'], best_params['dropout_rate'])
    final_model.load_state_dict(best_model_state)
    
    # Get validation probabilities for threshold optimization
    final_model.eval()
    with torch.no_grad():
        y_val_probs = final_model(X_val_t).squeeze().numpy()

    # Fine-grained threshold optimization on validation set
    thresholds = np.linspace(0.000, 1.000, 1001)
    f1_scores_val = []

    for t in thresholds:
        y_val_pred = (y_val_probs >= t).astype(int)
        f1_val = f1_score(y_val_split, y_val_pred, zero_division=0)
        f1_scores_val.append(f1_val)

    best_threshold_idx = np.argmax(f1_scores_val)
    best_threshold = thresholds[best_threshold_idx]
    best_f1_val = f1_scores_val[best_threshold_idx]

    # Apply fixed threshold to test set
    with torch.no_grad():
        y_test_probs = final_model(X_test_t).squeeze().numpy()
    
    y_test_pred = (y_test_probs >= best_threshold).astype(int)

    # Calculate test metrics
    f1_test = f1_score(y_test, y_test_pred, zero_division=0)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, zero_division=0)
    recall_test = recall_score(y_test, y_test_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    # Feature importance via first layer weights (absolute mean)
    selected_features = []
    weights_hash = None
    if X_columns is not None:
        weights = final_model.fc1.weight.data.abs().mean(dim=0).numpy()
        weights_hash = calculate_hash(weights)
        # Dynamic threshold based on weight distribution
        threshold = np.mean(weights) + 0.5 * np.std(weights)
        selected_features = [X_columns[i] for i, w in enumerate(weights) if w > threshold]

    result = {
        'model_name': 'NeuralNet',
        'iteration': iteration,
        'seed_iter': randomState,
        'best_f1': f1_test,  # Test F1 score
        'best_threshold': best_threshold,
        'f1_val': best_f1_val,  # Validation F1 score used for threshold selection
        'accuracy': accuracy_test,
        'precision': precision_test,
        'recall': recall_test,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'y_pred': y_test_pred.tolist(),
        'y_prob': y_test_probs.tolist(),
        'selected_features': selected_features,
        'method_has_selection': bool(selected_features),
        'n_selected': len(selected_features),
        'weights_hash': weights_hash,
        'best_params': best_params,
        # Debug information
        'train_samples': len(X_train_split),
        'val_samples': len(X_val_split),
        'test_samples': len(X_test)
    }

    return standardize_method_output(result)
