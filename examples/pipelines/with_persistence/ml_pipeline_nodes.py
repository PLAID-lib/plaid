import os
import json
import joblib
from joblib import Parallel, delayed
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


# def flatten_dict(d, prefix=''):
#     items = {}
#     for k, v in d.items():
#         new_key = f'{prefix}__{k}' if prefix else k
#         if isinstance(v, dict):
#             items.update(flatten_dict(v, new_key))
#         else:
#             items[f'param__{new_key}'] = v
#     return items

# def unflatten_dict(flat):
#     nested = {}
#     for flat_key, value in flat.items():
#         keys = flat_key.split('__')
#         d = nested
#         for k in keys[:-1]:
#             d = d.setdefault(k, {})
#         d[keys[-1]] = value
#     return nested


class PersistentNode(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, name, global_params, params):
        self.name = name
        self.global_params = global_params
        self.params = params
        self.save_path = Path(os.path.join(global_params['save_path'], f"{name}.joblib"))
        self.fitted_ = False
        self.model = None

    # def get_params(self, deep=True):
    #     return flatten_dict(self.params[self.name])

    # def set_params(self, **kwargs):
    #     if 'name' in kwargs:
    #         self.name = kwargs.pop('name')
    #     if 'params' in kwargs:
    #         self.params.update(kwargs.pop('params'))

    #     # Extract param__ keys and merge into self.params
    #     flat = {}
    #     for k, v in kwargs.items():
    #         if k.startswith('param__'):
    #             flat_key = k[len('param__'):]
    #             flat[flat_key] = v
    #     nested_update = unflatten_dict(flat)
    #     self.params = self._deep_merge(self.params, nested_update)
    #     return self

    def save(self, obj):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, self.save_path)
        self.fitted_ = True

    def load(self):
        print(f"Loading existing model from {self.save_path}")
        obj = joblib.load(self.save_path)
        self.fitted_ = True
        return obj

    def exists(self):
        return self.save_path.exists()

    def check_fitted_or_load(self):
        if self.fitted_:
            return
        if self.exists():
            self.set_model(self.load())
        else:
            raise ValueError("Model not fitted and no saved model found.")

    def set_model(self, obj):
        self.model = obj

    def fit(self, X, y=None):
        if self.exists():
            self.set_model(self.load())
            return self
        self._fit(X, y)
        self.save(self.model)
        return self

    def transform(self, X):
        self.check_fitted_or_load()
        return self._transform(X)

    def inverse_transform(self, X):
        self.check_fitted_or_load()
        return self._inverse_transform(X)

    def predict(self, X):
        self.check_fitted_or_load()
        return self._predict(X)

    def score(self, X, y):
        self.check_fitted_or_load()
        return self._score(X, y)

    # Protected methods to override in subclasses
    def _fit(self, X, y=None):
        raise NotImplementedError

    def _predict(self, X):
        raise NotImplementedError

    def _transform(self, X):
        raise NotImplementedError

    def _inverse_transform(self, X):
        raise NotImplementedError

    def _score(self, X, y):
        raise NotImplementedError



from sklearn.preprocessing import StandardScaler, MinMaxScaler

available_scalers = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}

class ScalarScalerNode(PersistentNode):

    def __init__(self, name, global_params, params):
        super().__init__(name, global_params, params)
        self.scalar_names = params['scalar_names']
        self.model = available_scalers[params['type']]()

    def get_scalars(self, dataset):
        return dataset.get_scalars_to_tabular(
            scalar_names = self.scalar_names,
            as_nparray = True
        )

    def set_scalars(self, dataset, scalars):
        for i in range(len(dataset)):
            for j, sn in enumerate(self.scalar_names):
                dataset[i].add_scalar(sn, scalars[i, j])

    def _fit(self, dataset, y=None):
        scalars = self.get_scalars(dataset)
        self.model.fit(scalars)

    def _transform(self, dataset):
        scalars = self.get_scalars(dataset)
        scaled_scalars = self.model.transform(scalars)
        self.set_scalars(dataset, scaled_scalars)

        return dataset

    def _inverse_transform(self, dataset):
        scaled_scalars = self.get_scalars(dataset)
        scalars = self.model.inverse_transform(scaled_scalars)
        self.set_scalars(dataset, scalars)

        return dataset


class PCAEmbeddingNode(PersistentNode):

    def __init__(self, name, global_params, params):
        super().__init__(name, global_params, params)

        self.zone_name = params["zone_name"] if "zone_name" in params else None
        self.base_name = params["base_name"] if "base_name" in params else None
        self.time      = params["time"]      if "time"      in params else None
        self.location  = params["location"]  if "location"  in params else "Vertex"

        assert params['type'] == "PCA"
        self.n_components = params['n_components']
        self.field_names = list(self.n_components.keys())
        self.model = {name: PCA(n_components = nc) for name, nc in self.n_components.items()}

    def get_params(self, deep=True):
        return {f"n_components__{key}":val for key, val in self.n_components.items()}

    def get_all_fields(self, dataset, fn):
        all_fields = []
        for sample in dataset:
            if fn == "nodes":
                field = sample.get_nodes(self.zone_name, self.base_name, self.time).flatten()
            else:
                field = sample.get_field(fn, self.zone_name, self.base_name, self.location, self.time)
            all_fields.append(field)
        return np.array(all_fields)


    def set_reduced_fields(self, dataset, fn, reduced_fields):
        for i in range(len(dataset)):
            for j in range(self.n_components[fn]):
                dataset[i].add_scalar(f"reduced_{fn}_{j}", reduced_fields[i, j])

    def get_reduced_fields(self, dataset, fn):
        return dataset.get_scalars_to_tabular(
            scalar_names = [f"reduced_{fn}_{j}" for j in range(self.n_components[fn])],
            as_nparray = True
        )

    def set_fields(self, dataset, fn, fields):
        for i in range(len(dataset)):
            dataset[i].add_field(fn, fields[i], self.zone_name, self.base_name, self.location, self.time)

    def _fit(self, dataset, y=None):
        for fn in self.field_names:
            all_fields = self.get_all_fields(dataset, fn)
            self.model[fn].fit(all_fields)

    def _transform(self, dataset):
        for fn in self.field_names:
            all_fields = self.get_all_fields(dataset, fn)
            reduced_fields = self.model[fn].transform(all_fields)
            self.set_reduced_fields(dataset, fn, reduced_fields)
        return dataset

    def _inverse_transform(self, dataset):
        for fn in self.field_names:
            reduced_fields = self.get_reduced_fields(dataset, fn)
            fields = self.model[fn].inverse_transform(reduced_fields)
            self.set_fields(dataset, fn, fields)
        return dataset


# from Muscat.Containers import MeshGraphTools as MGT
# from Muscat.Bridges.CGNSBridge import CGNSToMesh, MeshToCGNS
# from Muscat.Containers import MeshModificationTools as MMT

# import sys
# from contextlib import contextmanager

# @contextmanager
# def suppress_stdout():
#     original_stdout = sys.stdout
#     sys.stdout = open(os.devnull, 'w')
#     try:
#         yield
#     finally:
#         sys.stdout.close()
#         sys.stdout = original_stdout

# class TutteMorphing(PersistentNode):
#     def __init__(self, name, params):
#         super().__init__(params['save_path'], name)
#         self.prob_def = params['prob_def']
#         self.loc_params = params[name]
#         self.model = {}

#     def fit(self, X, y=None):
#         # No fitting needed here
#         return self

#     def transform(self, dataset):
#         return Parallel(n_jobs=self.loc_params['n_jobs'])(
#             delayed(self._process_row)(sample) for sample in dataset
#         )

#     # def inverse_transform(self, dataset):
#     #     return Parallel(n_jobs=self.n_jobs)(
#     #         delayed(self._process_row)(sample) for sample in dataset
#     #     )


#     def _process_row(self, sample):
#         # Your custom transformation logic

#         mesh = CGNSToMesh(sample.get_mesh())

#         with suppress_stdout():
#             mesh_renumb, renumbering, n_boundary = MGT.RenumberMeshForParametrization(
#                 mesh, inPlace=False)
#             mesh_renumb.elemFields = mesh_renumb.nodeFields = {}
#             morphed_mesh, _ = MGT.FloaterMeshParametrization(
#                 mesh_renumb, n_boundary)

#         # ---# Check invariance
#         assert (np.all(renumbering == np.argsort(np.argsort(renumbering))))
#         MMT.NodesPermutation(morphed_mesh, np.argsort(renumbering))

#         sample.del_tree(time = 0.)
#         sample.add_tree(MeshToCGNS(morphed_mesh))

#         return sample





from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor

available_kernel_classes = {
    "Matern":Matern
}

class GPRegressorNode(PersistentNode):

    def __init__(self, name, global_params, params):
        super().__init__(name, global_params, params)

        self.loc_params = params
        assert self.loc_params['type'] == "GaussianProcessRegressor"

        options = self.loc_params['options']
        assert options["kernel"] in available_kernel_classes.keys(), "scikit-learn kernel "+self.options["kernel"]+" not available"
        kernel_class = available_kernel_classes[options["kernel"]]

        self.input_names = self.loc_params['input']['names']
        self.output_names = self.loc_params['output']['names']

        if options["anisotropic"]:
            kernel = ConstantKernel() * kernel_class(length_scale=np.ones(len(self.input_names)), length_scale_bounds=(1e-8, 1e8),
                                    **options["kernel_options"]) + WhiteKernel(noise_level_bounds=(1e-8, 1e8))
        else:
            kernel = kernel_class(length_scale_bounds=(1e-8, 1e8), **options["kernel_options"]) \
                + WhiteKernel(noise_level_bounds=(1e-8, 1e8))

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=options["optim"],
            n_restarts_optimizer=options["num_restarts"],
            random_state = options["random_state"])

        self.model = MultiOutputRegressor(gpr)


    def get_scalars(self, dataset):
        return dataset.get_scalars_to_tabular(
            scalar_names = self.input_names,
            as_nparray = True
        )

    def _fit(self, dataset, y=None):
        X = dataset.get_scalars_to_tabular(
            scalar_names = self.input_names,
            as_nparray = True
        )
        y = dataset.get_scalars_to_tabular(
            scalar_names = self.output_names,
            as_nparray = True
        )
        self.model.fit(X, y)

    def _predict(self, dataset):
        X = dataset.get_scalars_to_tabular(
            scalar_names = self.input_names,
            as_nparray = True
        )

        pred= self.model.predict(X)
        if len(self.output_names) == 1:
            pred = pred.reshape((-1, 1))

        for i in range(len(dataset)):
            for j, sn in enumerate(self.output_names):
                dataset[i].add_scalar(sn, pred[i, j])

        return dataset

    def _transform(self, dataset):
        return dataset

    def _inverse_transform(self, dataset):
        return dataset

    def _score(self, dataset, dataset_ref):
        X = dataset.get_scalars_to_tabular(
            scalar_names = self.input_names,
            as_nparray = True
        )
        y = dataset_ref.get_scalars_to_tabular(
            scalar_names = self.output_names,
            as_nparray = True
        )
        return self.model.score(X, y)


class ScalerNode(PersistentNode):
    def __init__(self, name, save_path):
        super().__init__(save_path, name)

        self.model = StandardScaler()

    def _fit(self, X, y=None):
        self.model.fit(X)

    def _transform(self, X):
        return self.model.transform(X)

    def _inverse_transform(self, X):
        return self.model.inverse_transform(X)

    def _predict(self, X):
        raise AttributeError("ScalarScalerNode does not support predict.")


class RegressorNode(PersistentNode):
    def __init__(self, name, save_path, alpha=1.0):
        super().__init__(save_path, name)
        self.model = Ridge(alpha=alpha)

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _transform(self, X):
        raise AttributeError("RegressorNode does not support transform.")

    def _inverse_transform(self, X):
        raise AttributeError("RegressorNode does not support inverse_transform.")
