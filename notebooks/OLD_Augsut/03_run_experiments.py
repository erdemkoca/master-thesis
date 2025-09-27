import numpy as np
import pandas as pd
from methods.lasso import run_lasso
from methods.lasso_Net import run_lassonet
from methods.random_forest import run_random_forest
from methods.neural_net import run_neural_net
from methods.nimo_variants.baseline import run_nimo_baseline
from methods.nimo_variants.variant import run_nimo_variant
import os
import json

from notebooks.methods.nimo_variants.nimo import run_nimo

# Load split data
print("Loading data...")
X_train = np.load("../../data/splits/X_train.npy")
y_train = np.load("../../data/splits/y_train.npy")
X_test  = np.load("../../data/splits/X_test.npy")
y_test  = np.load("../../data/splits/y_test.npy")

X_train_columns = [
    'age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate',
    'glucose', 'male', 'education', 'currentSmoker', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes'
]

os.makedirs("../../results/", exist_ok=True)
os.makedirs("../../results/preds/", exist_ok=True)
os.makedirs("../../results/features/", exist_ok=True)

all_results = []

# Outer loop: subsampling by seed
for iteration in range(20):

    print(f"\n=== Iteration {iteration} ===")

    rng = np.random.default_rng(iteration)

    # Subsample majority class
    df = pd.DataFrame(X_train, columns=X_train_columns)
    df['target'] = y_train
    df_minority  = df[df['target'] == 1]
    df_majority  = df[df['target'] == 0]

    df_majority_down = df_majority.sample(
        n=len(df_minority),
        replace=False,
        random_state=rng
    )
    df_subsampled = pd.concat([df_majority_down, df_minority], ignore_index=True)

    X_sub = df_subsampled.drop(columns='target').values
    y_sub = df_subsampled['target'].values

    # Methoden
    methods = [
        # run_lasso,
        # run_lassonet,
        # run_random_forest,
        # run_neural_net,
        # run_nimo_baseline,
        run_nimo_variant,
        run_nimo
    ]

    randomState = int(rng.integers(0, 2**32 - 1))

    for method_fn in methods:
        print(f"Running {method_fn.__name__}...")

        result = method_fn(
            X_sub, y_sub, X_test, y_test,
            rng,
            iteration,
            randomState,
            X_train_columns
        )

        preds = result['y_pred']
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
        result['y_pred'] = json.dumps(preds)

        probs = result['y_prob']
        if isinstance(probs, np.ndarray):
            probs = probs.tolist()
        result['y_prob'] = json.dumps(probs)

        if 'selected_features' in result:
            result['selected_features'] = json.dumps(result['selected_features'])

        all_results.append(result)

        # Speichern der Raw-Arrays
        np.save(f"../results/preds/{result['model_name']}_iteration{iteration}_probs.npy",
                np.array(json.loads(result['y_prob'])))
        np.save(f"../results/preds/{result['model_name']}_iteration{iteration}_preds.npy",
                np.array(json.loads(result['y_pred'])))

        if 'selected_features' in result:
            with open(f"../results/features/{result['model_name']}_iteration{iteration}.json", "w") as f:
                f.write(result['selected_features'])

# Alle Ergebnisse in CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("../results/all_model_results.csv", index=False)
print("\nAll experiment results saved to ../results/all_model_results.csv")


#All commands
#  jupyter nbconvert --to script notebooks/run_experiments.ipynb
# jupyter notebooks
# python -m ipykernel install --user --name thesis_env --display-name "Python (thesis_env)"
# conda activate thesis_env  - python -m pip install ipykernel


