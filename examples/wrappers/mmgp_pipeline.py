#!/usr/bin/env python
# coding: utf-8

# # Exemple of pipeline PCA-GP-PCA type

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.wrappers.sklearn import WrappedSklearnTransform

# ## Make fake Dataset
NB_SAMPLES = 103
NB_INPUT_SCALARS = 3
NB_OUTPUT_SCALARS = 5
FIELD_SIZE = 17

X, y = make_regression(n_samples=NB_SAMPLES, n_features=NB_INPUT_SCALARS, n_targets=NB_OUTPUT_SCALARS + FIELD_SIZE, noise=0.1)

dset = Dataset()
samples = []
for sample_id in range(NB_SAMPLES):
    sample = Sample()
    for scalar_id in range(NB_INPUT_SCALARS):
        sample.add_scalar(f"input_scalar_{scalar_id}", X[sample_id, scalar_id])
    for scalar_id in range(NB_OUTPUT_SCALARS):
        sample.add_scalar(f"output_scalar_{scalar_id}", y[sample_id, scalar_id])
    sample.init_base(topological_dim=3, physical_dim=3)
    sample.init_zone(np.array([0,0,0]))
    sample.add_field("output_field", y[sample_id, NB_OUTPUT_SCALARS:])
    samples.append(sample)
dset.add_samples(samples)

print(f"{dset.get_scalar_names()=}")
print(f"{dset.get_field_names()=}")

# ## PCA-GP-PCA as a sklearn `Pipeline`
# ### 1. Define the PCA for the shape embedding
#
# In this example we only apply PCA to the first 8 columns
#
# The last two columns are unchanged
NB_PCA_MODES = 8
pca = WrappedSklearnTransform(
            PCA(NB_PCA_MODES),
            in_keys='field::all',
            # in_keys=['omega', 'compression_rate'],
            out_keys=[f'scalar::pca{i_mode}' for i_mode in range(NB_PCA_MODES)],
        )

pca.fit(dset, problem_def)

feats_to_reduce = list(range(8))
preprocessor = ColumnTransformer(
    transformers=[
        (
            "pca",
            PCA(n_components=8),
            feats_to_reduce,
        ),
    ],
    remainder="passthrough",
)

# ### 2. Define the output scaler for the output fields (MinMaxScaler + PCA)
postprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=9)),
    ]
)

# ### 3. Define the regressor such that
# transformer(Y) = GP(X) where transformer(Y) = postprocessor(Y)
# then
# Y = transformer^{-1}(GP(X))
kernel = DotProduct() + WhiteKernel()
regressor = TransformedTargetRegressor(
    regressor=GaussianProcessRegressor(
        n_restarts_optimizer=3,
    ),
    check_inverse=False,
    transformer=postprocessor,
)

# ### 4. Combine with preprocessings to make the pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ]
)

# ## Fit the model
model.fit(dset, problem_def)

# ## Predict on the training data
y_pred = model.predict(dset)

# ## Other way to define the pipeline
# ### 1. Define the regressor
regressor = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("regressor", GaussianProcessRegressor(n_restarts_optimizer=3)),
    ]
)

# ### 2. Combine with target transform to make the pipeline
model = TransformedTargetRegressor(
    regressor=regressor,
    check_inverse=False,
    transformer=postprocessor,
)
