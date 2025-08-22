#!/usr/bin/env python
# coding: utf-8

# # Pipeline Examples
#
# This notebook demonstrates the end-to-end process of building a machine learning pipeline using PLAID datasets and PLAID’s scikit-learn-compatible blocks.

# ## PCA-GP for `mach` field prediction of `VKI-LS59` dataset
#
# Key steps covered:
#
# - **Loading the PLAID dataset** using Hugging Face integration and PLAID’s dataset classes
# - **Standardizing features** with PLAID-wrapped scikit-learn transformers for scalars
# - **Dimensionality reduction** of flow fields via Principal Component Analysis (PCA) to reduce output complexity
# - **Regression modeling** of PCA coefficients from scalar inputs using Gaussian Process regression
# - **Pipeline assembly** combining transformations and regressors into a single scikit-learn-compatible workflow
# - **Hyperparameter tuning** using Optuna and scikit-learn’s `GridSearchCV`
# - **Best practices** for working with PLAID datasets and pipelines in a reproducible and modular manner

# ### 📦 Imports

# In[ ]:


import warnings

warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings("ignore", message=".*IProgress not found.*")

import os
from pathlib import Path

import numpy as np
import optuna
import yaml
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from plaid.bridges.huggingface_bridge import (
    huggingface_dataset_to_plaid,
)
from plaid.pipelines.plaid_blocks import ColumnTransformer, TransformedTargetRegressor
from plaid.pipelines.sklearn_block_wrappers import (
    WrappedSklearnRegressor,
    WrappedSklearnTransformer,
)

disable_progress_bar()
n_processes = min(max(1, os.cpu_count()), 6)


# ### 📥 Load Dataset
#
# We load the `VKI-LS59` dataset from Hugging Face and restrict ourselves to the first 24 samples of the training set.

# In[ ]:


hf_dataset = load_dataset("PLAID-datasets/VKI-LS59", split="all_samples[:24]")
dataset_train, _ = huggingface_dataset_to_plaid(hf_dataset, processes_number = n_processes, verbose = False)


# We print the summary of dataset_train, which contains 24 samples, with 8 scalars and 8 fields, which is consistent with the `VKI-LS59` dataset:

# In[ ]:


print(dataset_train)


# ### ⚙️ Pipeline Configuration
#
# For convenience, the `in_features_identifiers` and `out_features_identifiers` for each pipeline block are defined in a `.yml` file. Here's an example of how the configuration might look:

# ```yaml
# pca_nodes:
#   in_features_identifiers:
#     - type: nodes
#       base_name: Base_2_2
#   out_features_identifiers:
#     - type: scalar
#       name: reduced_nodes_*
# ```

# In[ ]:


try:
    filename = Path(__file__).parent.parent.parent / "docs" / "source" / "notebooks" / "config_pipeline.yml"
except NameError:
    filename = "config_pipeline.yml"

with open(filename, 'r') as f:
    config = yaml.safe_load(f)

all_feature_id = config['input_scalar_scaler']['in_features_identifiers'] +\
    config['pca_nodes']['in_features_identifiers'] + config['pca_mach']['in_features_identifiers']


# In this example, we aim to predict the ``mach`` field based on two input scalars ``angle_in`` and ``mach_out``, and the mesh node coordinates. To contain memory consumption, we restrict the dataset to the features required for this example:

# In[ ]:


dataset_train = dataset_train.from_features_identifier(all_feature_id)
print("dataset_train:", dataset_train)
print("scalar names =", dataset_train.get_scalar_names())
print("field names =", dataset_train.get_field_names())


# We notive that only the 2 scalars and the field of interest are kept after restriction.

# #### 1. Preprocessor
#
# We now define a preprocessor: a `MinMaxScaler` of the 2 input scalars and a `PCA` on the nodes coordinates of the meshes:

# In[ ]:


preprocessor = ColumnTransformer(
    [
        ('input_scalar_scaler', WrappedSklearnTransformer(MinMaxScaler(), **config['input_scalar_scaler'])),
        ('pca_nodes', WrappedSklearnTransformer(PCA(), **config['pca_nodes'])),
    ]
)
preprocessor


# We use a `PlaidColumnTransformer` to apply independent transformations to different feature groups.
#
# To verify this behavior, we apply the `preprocessor` to `dataset_train`:

# In[ ]:


preprocessed_dataset = preprocessor.fit_transform(dataset_train)
print("preprocessed_dataset:", preprocessed_dataset)
print("scalar names =", preprocessed_dataset.get_scalar_names())
print("field names =", preprocessed_dataset.get_field_names())


# Using `MinMaxScaler`, we scaled the `angle_in` and `mach_out` features, replacing their original values. In contrast, `PCA` compressed the node coordinates and produced new scalar features named `reduced_nodes_*`, representing the PCA components. Alternatively, we could have specified `out_features_identifiers` in the `.yml` file configuring the `MinMaxScaler` block to generate new scalars without overwriting the original inputs.

# #### 2. Postprocessor
#
# Next, we define the postprocessor, which applies PCA to the `mach` field:

# In[ ]:


postprocessor = WrappedSklearnTransformer(PCA(), **config['pca_mach'])
postprocessor


# #### 3. TransformedTargetRegressor
#
# The Gaussian Process regressor takes the transformed `angle_in` and `mach_out` scalars, along with the PCA coefficients of the mesh node coordinates as inputs, and predicts the PCA coefficients of the `mach` field as outputs. This is facilitated by using a `PlaidTransformedTargetRegressor`.

# In[ ]:


kernel = Matern(length_scale_bounds=(1e-8, 1e8), nu = 2.5)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    optimizer='fmin_l_bfgs_b',
    n_restarts_optimizer=1,
    random_state=42)

reg = MultiOutputRegressor(gpr)

regressor = WrappedSklearnRegressor(reg, **config['regressor_mach'])

target_regressor = TransformedTargetRegressor(
    regressor=regressor,
    transformer=postprocessor
)
target_regressor


# `PlaidTransformedTargetRegressor` functions like scikit-learn’s `TransformedTargetRegressor` but operates directly on PLAID datasets.

# #### 4. Pipeline assembling
#
# We then define the complete pipeline as follows:

# In[ ]:


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", target_regressor),
    ]
)
pipeline


# ### 🎯 Optuna hyperparameter tuning
#
# We now use Optuna to optimize hyperparameters, specifically tuning the number of components for the two `PCA` blocks using three-fold cross-validation.

# In[ ]:


def objective(trial):
    # Suggest hyperparameters
    nodes_n_components = trial.suggest_int("preprocessor__pca_nodes__sklearn_block__n_components", 3, 4)
    mach_n_components = trial.suggest_int("regressor__transformer__sklearn_block__n_components", 4, 5)

    # Clone and configure pipeline
    pipeline_run = clone(pipeline)
    pipeline_run.set_params(
        preprocessor__pca_nodes__sklearn_block__n_components=nodes_n_components,
        regressor__transformer__sklearn_block__n_components=mach_n_components,
        regressor__regressor__sklearn_block__estimator__kernel=Matern(
                    length_scale_bounds=(1e-8, 1e8), nu=2.5, length_scale=np.ones(nodes_n_components + len(config['input_scalar_scaler']['in_features_identifiers']))
                )
    )

    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    scores = []

    indices = np.arange(len(dataset_train))

    for train_idx, val_idx in cv.split(indices):

        dataset_cv_train_ = dataset_train[train_idx]
        dataset_cv_val_   = dataset_train[val_idx]

        pipeline_run.fit(dataset_cv_train_)

        score = pipeline_run.score(dataset_cv_val_)

        scores.append(score)

    return np.mean(scores)


# We maximize the defined objective function over 4 trials selected by Optuna.

# In[ ]:


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=4)
print("best_params =", study.best_params)


# We retrieve the best hyperparameters found by Optuna and use them to define the `optimized_pipeline`.

# In[ ]:


optimized_pipeline = clone(pipeline).set_params(**study.best_params)
optimized_pipeline.set_params(regressor__regressor__sklearn_block__estimator__kernel=Matern(
                    length_scale_bounds=(1e-8, 1e8), nu=2.5, length_scale=np.ones(study.best_params['preprocessor__pca_nodes__sklearn_block__n_components'] + len(config['input_scalar_scaler']['in_features_identifiers']))
                )
)

optimized_pipeline.fit(dataset_train)
optimized_pipeline


# Next, we fit the `optimized_pipeline` to the `dataset_train` dataset and evaluate its performance on the same data.

# In[ ]:


dataset_pred = optimized_pipeline.predict(dataset_train)
score = optimized_pipeline.score(dataset_train)
print("score =", score, ", error =", 1. - score)


# We use an anisotropic kernel in the Gaussian Process. Its optimized `length_scale` is a vector with dimensions equal to 2 plus the number of PCA components from `preprocessor__pca_nodes__sklearn_block__n_components`, accounting for the two input scalars.

# In[ ]:


print(optimized_pipeline.named_steps["regressor"].regressor_.sklearn_block_.estimators_[0].kernel_.get_params()['length_scale'])


# In[ ]:


print("Dimension GP kernel length_scale =", len(optimized_pipeline.named_steps["regressor"].regressor_.sklearn_block_.estimators_[0].kernel_.get_params()['length_scale']))
print("Expected dimension =", 2 + study.best_params['preprocessor__pca_nodes__sklearn_block__n_components'])


# The error remains non-zero due to the approximation introduced by PCA. Since the Gaussian Process regressor interpolates, the error is expected to vanish on the training set if all PCA modes are retained.

# In[ ]:


exact_pipeline = clone(pipeline).set_params(
    preprocessor__pca_nodes__sklearn_block__n_components = 24,
    regressor__transformer__sklearn_block__n_components = 24
)
exact_pipeline.fit(dataset_train)
dataset_pred = exact_pipeline.predict(dataset_train)
score = exact_pipeline.score(dataset_train)
print("score =", score, ", error =", 1. - score)


# ### 🔍 GridSearchCV hyperparameter tuning
#
# Since our pipeline nodes conform to the scikit-learn API, the constructed pipeline can be used directly with `GridSearchCV`.

# In[ ]:


pca_n_components = [3, 4]
regressor_n_components = [4, 5]

param_grid = []
for n, m in zip(pca_n_components, regressor_n_components):
    param_grid.append(
        {
            "preprocessor__pca_nodes__sklearn_block__n_components": [n],
            "regressor__transformer__sklearn_block__n_components": [m],
            "regressor__regressor__sklearn_block__estimator__kernel": [
                Matern(
                    length_scale_bounds=(1e-8, 1e8), nu=2.5, length_scale=np.ones(n + 2)
                )
            ],
        }
    )

cv = KFold(n_splits=3, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, verbose=3, error_score='raise')
search.fit(dataset_train)


# We evaluate the performance of the optimized pipeline by computing its score on the training set.

# In[ ]:


print("best_params =", search.best_params_)
optimized_pipeline = clone(pipeline).set_params(**search.best_params_)
optimized_pipeline.fit(dataset_train)
dataset_pred = optimized_pipeline.predict(dataset_train)
score = optimized_pipeline.score(dataset_train)
print("score =", score, ", error =", 1. - score)

