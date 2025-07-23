# from ml_pipeline_nodes_5 import PlaidTransformedTargetRegressor, WrappedPlaidSklearnTransformer, WrappedPlaidSklearnRegressor, PlaidColumnTransformer, PlaidSklearnBlockWrapper_
from ml_pipeline_nodes_5 import WrappedPlaidSklearnTransformer, PlaidColumnTransformer, WrappedPlaidSklearnRegressor, PlaidTransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import yaml
from sklearn.base import clone
from sklearn.pipeline import Pipeline

with open("config_5.yml") as f:
    config = yaml.safe_load(f)



conf1 = config['pca_nodes']



wrapped_transf = WrappedPlaidSklearnTransformer(PCA(**config['pca_nodes']['sklearn_block_params']), **config['pca_nodes']['plaid_params'])
print("wrapped_transf.get_params()['sklearn_block__n_components'] (3) =", wrapped_transf.get_params()['sklearn_block__n_components'])
assert wrapped_transf.get_params()['sklearn_block__n_components'] == config['pca_nodes']['sklearn_block_params']['n_components']

wrapped_transf_clone = clone(wrapped_transf)

wrapped_transf_clone.set_params(**{'sklearn_block__n_components':5})
print("wrapped_transf_clone.get_params()['sklearn_block__n_components'] (5) =", wrapped_transf_clone.get_params()['sklearn_block__n_components'])
# print("wrapped_transf_clone.sklearn_block.get_params() =", wrapped_transf_clone.sklearn_block.get_params())

print("==")

preprocessor = PlaidColumnTransformer([
    ('input_scalar_scaler', WrappedPlaidSklearnTransformer(MinMaxScaler(), **config['input_scalar_scaler']['plaid_params'])),
    ('pca_nodes', WrappedPlaidSklearnTransformer(PCA(**config['pca_nodes']['sklearn_block_params']), **config['pca_nodes']['plaid_params'])),
], remainder_feature_id = config['pca_mach']['plaid_params']['in_features_identifiers'])
print("preprocessor.get_params()['pca_nodes__sklearn_block__n_components'] (3) =", preprocessor.get_params()['pca_nodes__sklearn_block__n_components'])
assert preprocessor.get_params()['pca_nodes__sklearn_block__n_components'] == config['pca_nodes']['sklearn_block_params']['n_components']

preprocessor_clone = clone(preprocessor)
preprocessor_clone.set_params(**{'pca_nodes__sklearn_block__n_components':5})
print("preprocessor_clone.get_params()['pca_nodes__sklearn_block__n_components'] (5) =", preprocessor_clone.get_params()['pca_nodes__sklearn_block__n_components'])
# print("wrapped_transf_clone.get_params() =", wrapped_transf_clone.get_params())


print("==")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor


available_kernel_classes = {
    "Matern":Matern
}


kernel = ConstantKernel() * Matern(length_scale_bounds=(1e-8, 1e8), nu = 2.5) + WhiteKernel(noise_level_bounds=(1e-8, 1e8))

gpr = GaussianProcessRegressor(
    kernel=kernel,
    optimizer='fmin_l_bfgs_b',
    n_restarts_optimizer=1,
    random_state=42)

reg = MultiOutputRegressor(gpr)

# print("==")
# kernel = ConstantKernel()*Matern() + WhiteKernel()
# print("kernel.get_params() =", kernel.get_params()['k1__k2__nu'])
# kernel.set_params(**{'k1__k2__nu':2.5})
# print("kernel.get_params() =", kernel.get_params()['k1__k2__nu'])

# print("==")
# gp = GaussianProcessRegressor(ConstantKernel()*Matern() + WhiteKernel())
# print("gp.get_params() =", gp.get_params())
# print("gp.get_params() =", gp.get_params()['kernel__k1__k2__nu'])
# # mat.set_params(**{'nu':2.5})
# # print("mat.get_params() =", mat.get_params()['nu'])

# print("==")
# reg = MultiOutputRegressor(GaussianProcessRegressor(ConstantKernel()*Matern()+WhiteKernel()))
# print("reg.get_params() =", reg.get_params())
# print("reg.get_params() =", reg.get_params()['estimator__kernel__k1__k2__nu'])

postprocessor = WrappedPlaidSklearnTransformer(PCA(**config['pca_mach']['sklearn_block_params']), **config['pca_mach']['plaid_params'])


regressor = WrappedPlaidSklearnRegressor(reg, **config['regressor_mach']['plaid_params'])

print("regressor.get_params() (2.5) =", regressor.get_params()['sklearn_block__estimator__kernel__k1__k2__nu'])
assert regressor.get_params()['sklearn_block__estimator__kernel__k1__k2__nu'] == 2.5

regressor_clone = clone(regressor)

regressor_clone.set_params(**{'sklearn_block__estimator__kernel__k1__k2__nu':1.5})
print("regressor_clone.get_params() (1.5) =", regressor_clone.get_params()['sklearn_block__estimator__kernel__k1__k2__nu'])



print("==")


target_regressor = PlaidTransformedTargetRegressor(
    regressor=regressor,
    transformer=postprocessor,
    transformed_target_feature_id=config['pca_mach']['plaid_params']['in_features_identifiers']
)
print("target_regressor.get_params(nu) (2.5) =", target_regressor.get_params()['regressor__sklearn_block__estimator__kernel__k1__k2__nu'])
print("target_regressor.get_params(n_comp mach) (5) =", target_regressor.get_params()['transformer__sklearn_block__n_components'])

print("==")

target_regressor_clone = clone(target_regressor)
target_regressor_clone.set_params(**{'regressor__sklearn_block__estimator__kernel__k1__k2__nu':1.5})
target_regressor_clone.set_params(**{'transformer__sklearn_block__n_components':4})

print("target_regressor_clone.get_params(nu) (1.5) =", target_regressor_clone.get_params()['regressor__sklearn_block__estimator__kernel__k1__k2__nu'])
print("target_regressor_clone.get_params(n_comp mach) (4) =", target_regressor_clone.get_params()['transformer__sklearn_block__n_components'])

# WrappedPlaidSklearnRegressor