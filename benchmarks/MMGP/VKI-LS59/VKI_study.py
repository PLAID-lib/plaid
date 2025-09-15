# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: 2dblade
#     language: python
#     name: python3
# ---

# %%
# %reload_ext autoreload
# %autoreload 2
import time

import numpy as np
from AM_POD import PolynomialManifoldApproximation
from data import extract_split_data, make_kfold_splits
from tqdm import tqdm

# 1) Load the train split and build 5 folds
inputs, outputs = extract_split_data("train")
folds = make_kfold_splits(inputs, outputs, n_splits=5, random_state=42)

# %%
# 2) Define the grid of (polynomial_order, r) to search
param_grid = [(p, r) for p in [1, 2, 3] for r in [5, 10, 15, 20, 25, 30, 35, 40]]

results_mach = {}
all_errors = []
print("=== Searching best parameters for MACH ===")
for poly_order, r in param_grid:
    print(f"Testing polynomial_order={poly_order}, r={r} for MACH")
    t0 = time.time()
    fold_errors = []

    with tqdm(
        total=len(folds), desc=f"  MACH p={poly_order}, r={r}", leave=False
    ) as pbar:
        for fold_idx, (train_in, train_out, val_in, val_out) in enumerate(folds):
            # prepare snapshots for mach
            S_train = np.stack(train_out["mach"], axis=0)
            S_val = np.stack(val_out["mach"], axis=0)

            pma = PolynomialManifoldApproximation(
                polynomial_order=poly_order,
                r=r,
            )
            pma.fit(S_train, S_val)
            err = pma.score(S_val)
            fold_errors.append(err)

            # update the tqdm bar
            avg_err = np.mean(fold_errors)
            pbar.set_postfix({"last_err": f"{err:.4e}", "avg_err": f"{avg_err:.4e}"})
            pbar.update()

            all_errors.append(
                {"poly_order": poly_order, "r": r, "fold": fold_idx, "error": err}
            )

    avg_err = np.mean(fold_errors)
    elapsed = time.time() - t0
    results_mach[(poly_order, r)] = avg_err
    print(
        f"--> Done p={poly_order}, r={r}: avg_error={avg_err:.4f} (time {elapsed:.1f}s)"
    )
    print("_" * 80)

best_mach = min(results_mach, key=results_mach.get)
print(
    f"\nBest for MACH → polynomial_order={best_mach[0]}, r={best_mach[1]} "
    f"(avg_val_error={results_mach[best_mach]:.4f})"
)

# %%
from utils import CVResults

results_mach = CVResults(all_errors)
results_mach.print_summary()

# %%
# 2) Define the grid of (polynomial_order, r) to search
param_grid = [(p, r) for p in [1, 2, 3] for r in [5, 10, 15, 20, 25, 30, 35, 40]]

# 3b) Search best params for 'nut'
results_nut = {}
all_errors_nut = []

print("=== Searching best parameters for NUT ===")
for poly_order, r in param_grid:
    print(f"\nTesting polynomial_order={poly_order}, r={r} for NUT")
    t0 = time.time()
    fold_errors = []

    with tqdm(
        total=len(folds), desc=f"  NUT p={poly_order}, r={r}", leave=False
    ) as pbar:
        for fold_idx, (train_in, train_out, val_in, val_out) in enumerate(folds):
            # prepare snapshots for nut
            S_train = np.stack(train_out["nut"], axis=0)
            S_val = np.stack(val_out["nut"], axis=0)

            pma = PolynomialManifoldApproximation(
                polynomial_order=poly_order,
                r=r,
                reg_ls=0.001,
            )
            pma.fit(S_train, S_val)
            err = pma.score(S_val)
            fold_errors.append(err)

            # update the tqdm bar
            avg_err = np.mean(fold_errors)
            pbar.set_postfix({"last_err": f"{err:.4e}", "avg_err": f"{avg_err:.4e}"})
            pbar.update()

            all_errors_nut.append(
                {"poly_order": poly_order, "r": r, "fold": fold_idx, "error": err}
            )

    avg_err = np.mean(fold_errors)
    elapsed = time.time() - t0
    results_nut[(poly_order, r)] = avg_err
    print(
        f"--> Done p={poly_order}, r={r}: avg_error={avg_err:.4f} (time {elapsed:.1f}s)"
    )
    print("_" * 80)

best_nut = min(results_nut, key=results_nut.get)
print(
    f"\nBest for NUT → polynomial_order={best_nut[0]}, r={best_nut[1]} "
    f"(avg_val_error={results_nut[best_nut]:.4f})"
)

# %%
results_nut = CVResults(all_errors_nut)
results_nut.print_summary()

# %%
# %reload_ext autoreload
# %autoreload 2
from processor import InputProcessor, OutputProcessor

inputprocessor = InputProcessor(explained_variance=0.99999)
X = inputprocessor.fit_transform(inputs)

# %%
print(inputprocessor.n_components_)

# %%
outputprocessor = OutputProcessor(
    mach_params=(3, 5), nut_params=(1, 40), verbose=True, max_iter_rot=22
)
Y = outputprocessor.fit_transform(outputs)

# %%
from model import GPyRegressor

# one single GP
gpmodel = GPyRegressor()
gpmodel.fit(X, Y)

# %%
inputs_test, outputs_test = extract_split_data("test")

# %%
X_test = inputprocessor.transform(inputs_test)
Y_test = gpmodel.predict(X_test)
outputs_test_pred = outputprocessor.inverse_transform(Y_test)

# %%
import matplotlib.pyplot as plt
from utils import plot_mach_nut, plot_scalars_pred_vs_true

fig, axs = plot_mach_nut(inputs_test, outputs_test, outputs_test_pred, idx=100)
plt.show()

# %%
fig, axs = plot_scalars_pred_vs_true(outputs_test, outputs_test_pred)

# %%
from data import dump_predictions

dump_predictions(outputs_test_pred, "predictions.pkl")

# %%
from joblib import Parallel, delayed
from model import GPyRegressor


def train_single_output(X, y):
    gp = GPyRegressor()
    gp.fit(X, y)
    return gp


# X: (n_samples, n_features)
# Y: (n_samples, n_outputs)
n_outputs = Y.shape[1]

# Parallel training: use all cores (n_jobs=-1), and print progress (verbose=10)
gp_models = Parallel(n_jobs=-1, verbose=10)(
    delayed(train_single_output)(X, Y[:, j]) for j in range(n_outputs)
)

# %%
len(gp_models)

# %%
X_test = inputprocessor.transform(inputs_test)
Y_test = np.column_stack([gp.predict(X_test) for gp in gp_models])
outputs_test_pred = outputprocessor.inverse_transform(Y_test)

# %%
import matplotlib.pyplot as plt
from utils import plot_mach_nut, plot_scalars_pred_vs_true

fig, axs = plot_mach_nut(inputs_test, outputs_test, outputs_test_pred, idx=100)

# %%
fig, axs = plot_scalars_pred_vs_true(outputs_test, outputs_test_pred)

# %%
from data import dump_predictions

dump_predictions(outputs_test_pred, "predictions.pkl")
