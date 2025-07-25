#!/usr/bin/env python
# coding: utf-8

# # Pipeline Examples
#
# This notebook presents how the provided pipeline blocks can be used in a PCA-GP algorithm. Pipeline block directly link PLAID datasets, and hyperpamater tuning is available using scikit-learn's GridSearchCV or Optuna.
#
# We start by a few imports:

# In[ ]:


import os
import yaml
from pathlib import Path
import optuna
import numpy as np

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import KFold, GridSearchCV

import warnings
from tqdm import TqdmWarning
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', category=TqdmWarning)

from datasets import load_dataset, get_dataset_infos
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid, huggingface_description_to_problem_definition
from plaid.pipelines.sklearn_block_wrappers import WrappedPlaidSklearnTransformer, WrappedPlaidSklearnRegressor
from plaid.pipelines.plaid_blocks import PlaidTransformedTargetRegressor, PlaidColumnTransformer

nb_cpus = os.cpu_count()
n_processes = max(1, int(nb_cpus / 4))


# We load the `VKI-LS59` dataset from Hugging Face, and restrict ourselves to the first 24 samples of the training set.

# In[ ]:


hf_dataset = load_dataset("PLAID-datasets/VKI-LS59", split="all_samples[:24]")
dataset_train, _ = huggingface_dataset_to_plaid(hf_dataset, processes_number = n_processes, verbose = False)

del hf_dataset


# We print the summary of dataset_train:

# In[ ]:


print(dataset_train)


# There are 24 samples, with 8 scalars and 8 fields, which is consistent with the VKI-LS59` dataset. To contain memory consumption, we restrict the dataset to the features required for this example. Far praticity, in_features_identifiers and out_features_identifiers of each pipeline block are defined in a ``.yml`` file. In this example, we try to predict the ``mach`` based on two input scalars ``angle_in`` and ``mach_out``, and the mesh node coordinates.

# In[ ]:


try:
    filename = Path(__file__).parent.parent.parent / "docs" / "source" / "notebooks" / "config_pipeline.yml"
except NameError:
    filename = "config_pipeline.yml"

with open(filename, 'r') as f:
    config = yaml.safe_load(f)

all_feature_id = config['input_scalar_scaler']['in_features_identifiers'] +\
    config['pca_nodes']['in_features_identifiers'] +\
    config['pca_mach']['in_features_identifiers']


dataset_train = dataset_train.from_features_identifier(all_feature_id)
print(dataset_train)


# Hence, we keep only 2 scalars and 1 field of interest.
#
# We now define a preprocessor: a `MinMaxScaler` of the 2 input scalars and a `PCA` on the nodes coordinates of the meshes:

# In[ ]:


preprocessor = PlaidColumnTransformer([
    ('input_scalar_scaler', WrappedPlaidSklearnTransformer(MinMaxScaler(), **config['input_scalar_scaler'])),
    ('pca_nodes', WrappedPlaidSklearnTransformer(PCA(), **config['pca_nodes'])),
], remainder_feature_ids = config['pca_mach']['in_features_identifiers'])
preprocessor


# We use a `PlaidColumnTransformer`, which enable independant transformations of features. The `out_features_identifiers` of each transformer are appended to `remainder_feature_ids`, which specifies the feature that will be passed through, such that only these features are kept in the returned merged dataset.
#
# We check this by applying the `preprocessor` to `dataset_train`:

# In[ ]:


preprocessed_dataset = preprocessor.fit_transform(dataset_train)
print(preprocessed_dataset)
print("scalar names =", preprocessed_dataset.get_scalar_names())
print("field names =", preprocessed_dataset.get_field_names())


# With `MinMaxScaler`, we have scaled `angle_in` and `mach_out` and overridden their values, while with `PCA`, we have compressed the nodes coordinates and returned scalars with name `reduced_nodes_*'` containing the PCA coordinates. We could have specified `out_features_identifiers` in the `.yml` file to generate new scalars instead of overriding `in_features_identifiers`.
#
# We now define the postprocessor, which is here a PCA on the `mach` field:

# In[ ]:


postprocessor = WrappedPlaidSklearnTransformer(PCA(), **config['pca_mach'])
postprocessor


# The regressor in a Gaussian Process applied on the transformed ``angle_in`` and ``mach_out``, and the mesh node coordinates PCA coefficients as inputs, and the ``mach`` PCA coefficients as outputs. A ``PlaidTransformedTargetRegressor`` enable us to do this:

# In[ ]:


kernel = Matern(length_scale_bounds=(1e-8, 1e8), nu = 2.5)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    optimizer='fmin_l_bfgs_b',
    n_restarts_optimizer=1,
    random_state=42)

reg = MultiOutputRegressor(gpr)

def length_scale_init(X):
    return np.ones(X.shape[1])

dynamics_params_factory = {'estimator__kernel__length_scale':length_scale_init}

regressor = WrappedPlaidSklearnRegressor(reg, **config['regressor_mach'], dynamics_params_factory = dynamics_params_factory)

target_regressor = PlaidTransformedTargetRegressor(
    regressor=regressor,
    transformer=postprocessor,
    transformed_target_feature_id=config['pca_mach']['in_features_identifiers']
)
target_regressor


# `PlaidTransformedTargetRegressor` work as a scikit-learn `TransformedTargetRegressor`, but directly on PLAID datasets. The argument `transformed_target_feature_id` allows to specify which feature identifiers are concerned by the transformation.
#
# Finally, we define the complete pipeline as:

# In[ ]:


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", target_regressor),
    ]
)
pipeline


# ## Optuna
#
# We now use optune to optimze the hyperparmeters, by a research over the number of components of the two `PCA` blocks, and a three-fold cross validation:

# In[ ]:


def objective(trial):
    # Suggest hyperparameters
    nodes_n_components = trial.suggest_int("preprocessor__pca_nodes__sklearn_block__n_components", 3, 4)
    mach_n_components = trial.suggest_int("regressor__transformer__sklearn_block__n_components", 4, 5)

    # Clone and configure pipeline
    pipeline_run = clone(pipeline)
    pipeline_run.set_params(
        preprocessor__pca_nodes__sklearn_block__n_components=nodes_n_components,
        regressor__transformer__sklearn_block__n_components=mach_n_components
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


# We maximize the defined objective function over 4 trial runs chosen by optuna:

# In[ ]:


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=4)
print("best_params =", study.best_params)


# We retrieve the best found hyperparameters and define `optimized_pipeline` based on these values

# In[ ]:


optimized_pipeline = clone(pipeline).set_params(**study.best_params)
optimized_pipeline.fit(dataset_train)
optimized_pipeline


# Then, we fit the `dataset_train` dataset and compute the score on this same dataset:

# In[ ]:


dataset_pred = optimized_pipeline.predict(dataset_train)
score = optimized_pipeline.score(dataset_train)
print("score =", score, ", error =", 1. - score)


# We use an anisotropic kernel in the Gaussian Process: its optimized `length_scale` should be a vector with 2+preprocessor__pca_nodes__sklearn_block__n_components components (since we have appending 2 input scalars):

# In[ ]:


print("Dimension GP kernel length_scale =", len(optimized_pipeline.named_steps["regressor"].regressor_.sklearn_block_.estimators_[0].kernel_.get_params()['length_scale']))
print("Expected dimension =", 2 + study.best_params['preprocessor__pca_nodes__sklearn_block__n_components'])


# The error is non-zero on due to the PCA errors. Since we have an interpolating Gaussian Process, we expect the error to vanish on the training set if we keep all the PCA modes:

# In[ ]:


exact_pipeline = clone(pipeline).set_params(
    preprocessor__pca_nodes__sklearn_block__n_components = 24,
    regressor__transformer__sklearn_block__n_components = 24
)
exact_pipeline.fit(dataset_train)
dataset_pred = exact_pipeline.predict(dataset_train)
score = exact_pipeline.score(dataset_train)
print("score =", score, ", error =", 1. - score)


# ## GridSearchCV
#
# Our pipeline node design satisfying the scikit-learn API, the constructed pipeline is directly compatible with GridSearchCV:

# In[ ]:


param_grid = {
    'preprocessor__pca_nodes__sklearn_block__n_components': [3, 4],
    'regressor__transformer__sklearn_block__n_components': [4, 5],
}

search = GridSearchCV(pipeline, param_grid=param_grid, cv=3, verbose=3, error_score='raise')
search.fit(dataset_train)


# We check the score on the training set using the optimized pipeline:

# In[ ]:


print("best_params =", search.best_params_)
optimized_pipeline = clone(pipeline).set_params(**search.best_params_)
optimized_pipeline.fit(dataset_train)
dataset_pred = optimized_pipeline.predict(dataset_train)
score = optimized_pipeline.score(dataset_train)
print("score =", score, ", error =", 1. - score)

