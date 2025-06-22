#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from methods.lasso import run_lasso
from methods.lasso_Net import run_lassonet
from methods.random_forest import run_random_forest
from methods.neural_net import run_neural_net
import os
import json

# Load split data
print("Loading data...")
X_train = np.load("../data/splits/X_train.npy")
y_train = np.load("../data/splits/y_train.npy")
X_test = np.load("../data/splits/X_test.npy")
y_test = np.load("../data/splits/y_test.npy")

X_train_columns = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate',
                   'glucose', 'male', 'education', 'currentSmoker', 'BPMeds',
                   'prevalentStroke', 'prevalentHyp', 'diabetes']

# Create output dirs
os.makedirs("../results/", exist_ok=True)
os.makedirs("../results/preds/", exist_ok=True)
os.makedirs("../results/features/", exist_ok=True)

all_results = []

# Outer loop: subsampling by seed
for seed in range(20):
    print(f"\n=== Seed {seed} ===")

    # Subsample majority class
    df = pd.DataFrame(X_train, columns=X_train_columns)
    df['target'] = y_train
    df_minority = df[df['target'] == 1]
    df_majority = df[df['target'] == 0]

    df_majority_down = df_majority.sample(n=len(df_minority), replace=False, random_state=seed)
    df_subsampled = pd.concat([df_majority_down, df_minority], ignore_index=True)
    X_sub = df_subsampled.drop(columns='target').values
    y_sub = df_subsampled['target'].values

    # Methods
    methods = [
        run_lasso,
        # run_lassonet,
        run_random_forest,
        run_neural_net
    ]

    for method_fn in methods:
        print(f"Running {method_fn.__name__}...")
        result = method_fn(X_sub, y_sub, X_test, y_test, seed, X_train_columns)

        # Convert arrays/lists to JSON-safe format
        result['y_pred'] = json.dumps(result['y_pred'].tolist() if isinstance(result['y_pred'], np.ndarray) else result['y_pred'])
        result['y_prob'] = json.dumps(result['y_prob'].tolist() if isinstance(result['y_prob'], np.ndarray) else result['y_prob'])
        if 'selected_features' in result:
            result['selected_features'] = json.dumps(result['selected_features'])


        print(result)
        all_results.append(result)

        # Save raw arrays separately (optional)
        np.save(f"../results/preds/{result['model_name']}_seed{seed}_probs.npy", np.array(json.loads(result['y_prob'])))
        np.save(f"../results/preds/{result['model_name']}_seed{seed}_preds.npy", np.array(json.loads(result['y_pred'])))

        if 'selected_features' in result:
            with open(f"../results/features/{result['model_name']}_seed{seed}.json", "w") as f:
                f.write(result['selected_features'])

# Save all metrics to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("../results/all_model_results.csv", index=False)
print("\nAll experiment results saved to ../results/all_model_results.csv")




