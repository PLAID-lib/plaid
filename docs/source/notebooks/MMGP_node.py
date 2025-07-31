
import copy
from typing import Union

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
)
from sklearn.utils.validation import check_is_fitted

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.containers.utils import get_feature_type_and_details_from_identifier, check_features_type_homogeneity

import Muscat.Containers.ElementsDescription as ED
import Muscat.Containers.Filters.FilterObjects as FilterObjects
import Muscat.Containers.MeshModificationTools as MMT
import networkx
import numpy as np
from Muscat.Containers.Mesh import Mesh
from scipy import sparse


import Muscat.Containers.Filters.FilterObjects as FilterObjects
import Muscat.Containers.MeshInspectionTools as MIT
import networkx
import numpy as np
from Muscat.Containers import MeshFieldOperations as MFO
from Muscat.Containers.Mesh import Mesh
from Muscat.FE import FETools as FT
from Muscat.FE.Fields import FEField as FF
from Muscat.Bridges.CGNSBridge import CGNSToMesh, MeshToCGNS
import time
from typing import Tuple, Callable




class MMGPPreparer(TransformerMixin, BaseEstimator):
    """Adapter for using a scikit-learn transformer on PLAID Datasets.

    Transforms tabular data extracted from homogeneous feature identifiers,
    and returns results as a `Dataset`. Supports forward and inverse transforms.

    Args:
        sklearn_block: A scikit-learn Transformer implementing fit/transform APIs.
        in_features_identifiers: List of feature identifiers to extract input data from.
        out_features_identifiers: List of feature identifiers used for outputs. If None,
            defaults to `in_features_identifiers`.
    """
    # TODO: check if restrict_to_features=True can be used to reduce further memory consumption
    def __init__(
        self,
        common_mesh_id: int = None
    ):
        self.common_mesh_id = common_mesh_id

    def morph(self, mesh):
        mesh_renumb, _, n_boundary = renumber_mesh_for_parametrization(
            mesh, in_place=False)
        mesh_renumb.elemFields = mesh_renumb.nodeFields = {}
        morphed_mesh, _ = floater_mesh_parametrization(
            mesh_renumb, n_boundary)
        return morphed_mesh

    def fit(self, dataset: Dataset, _y=None):
        """Fits the underlying scikit-learn transformer on selected input features.

        Args:
            dataset: A `Dataset` object containing the features to transform.
            _y: Ignored.

        Returns:
            self: The fitted transformer.
        """

        self.common_mesh_id_ = 0 if self.common_mesh_id is None else self.common_mesh_id

        self.morphed_common_mesh_ = self.morph(CGNSToMesh(dataset[self.common_mesh_id_].get_mesh()))
        self.morphed_common_tree_ = MeshToCGNS(self.morphed_common_mesh_, exportOriginalIDs=False, tagsAsFields=False)

        return self

    def transform(self, dataset: Dataset):
        """Applies the fitted transformer to the selected input features.

        Args:
            dataset: A `Dataset` object to transform.

        Returns:
            Dataset: Transformed features wrapped as a new `Dataset`.
        """
        check_is_fitted(self, "morphed_common_mesh_")

        # n_dim = self.morphed_common_mesh_.GetPointsDimensionality()
        # coord_names = ["coords_X", "coords_Y", "coords_Z"][:n_dim]

        samples = []
        for sample in dataset:
            # morph the mesh of the current sample
            mesh = CGNSToMesh(sample.get_mesh())
            morphed_mesh = self.morph(mesh)

            # compute the FE interpolation operator
            proj_operator = compute_FE_projection_operator(morphed_mesh, self.morphed_common_mesh_)
            inv_proj_operator = compute_FE_projection_operator(self.morphed_common_mesh_, morphed_mesh)

            # # initialize morph_proj_sample and add the common mesh as geometrical support of morph_proj_sample
            # morph_proj_sample = Sample()
            # morph_proj_sample.add_tree(copy.deepcopy(self.morphed_common_tree_))

            # # update all features (except nodes and fields) of morph_proj_sample from the ones of current sample
            # features_identifiers = sample.get_all_features_identifiers()
            # features_identifiers = [feat_id for feat_id in features_identifiers if feat_id.get("type") != "nodes"]
            # features = sample.get_features_from_identifiers(features_identifiers)
            # morph_proj_sample.update_features_from_identifier(features_identifiers, features, in_place=True)

            # # add morph_proj_sample coord fields as the node coordinates of current sample
            # for nodes_feat_id in sample.get_all_features_identifiers_by_type("nodes"):
            #     args = {key: value for key, value in nodes_feat_id.items() if key != 'type'}
            #     coords = sample.get_feature_from_identifier(nodes_feat_id).reshape((-1,n_dim))
            #     for dim in range(n_dim):
            #         morph_proj_sample.add_field(name = coord_names[dim], field = coords[:,dim], **args)

            # update all field of morph_proj_sample by finite element interpolation onto the common mesh
            # field_features_identifiers = morph_proj_sample.get_all_features_identifiers_by_type("field")
            # projected_field = [proj_operator.dot(field) for field in morph_proj_sample.get_features_from_identifiers(field_features_identifiers)]
            # morph_proj_sample.update_features_from_identifier(field_features_identifiers, projected_field, in_place=True)

            # # add input nodes coordinate to extra_sample for inverse transform needs
            # extra_sample = Sample()
            # morphed_tree = MeshToCGNS(morphed_mesh, exportOriginalIDs=False, tagsAsFields=False)
            # extra_sample.add_tree(morphed_tree)
            # for nodes_feat_id in sample.get_all_features_identifiers_by_type("nodes"):
            #     args = {key: value for key, value in nodes_feat_id.items() if key != 'type'}
            #     coords = sample.get_feature_from_identifier(nodes_feat_id).reshape((-1,n_dim))
            #     for dim in range(n_dim):
            #         extra_sample.add_field(name = coord_names[dim], field = coords[:,dim], **args)

            sample_ = sample.copy()
            sample_._extra_data = {
                    'init_mesh':mesh,
                    'proj_operator':proj_operator,
                    'inv_proj_operator':inv_proj_operator
                }

            samples.append(sample_)

        transformed_dataset = Dataset.from_list_of_samples(samples)

        return transformed_dataset


    def inverse_transform(self, dataset: Dataset):
        """Applies inverse transformation to the output features.

        Args:
            dataset: A `Dataset` object with transformed output features.

        Returns:
            Dataset: Dataset with inverse-transformed features.
        """
        check_is_fitted(self, "morphed_common_mesh_")

        dataset_ = dataset.copy()
        for sample in dataset_:
            sample._extra_data = None

        # n_dim = self.morphed_common_mesh_.GetPointsDimensionality()
        # coord_names = ["coords_X", "coords_Y", "coords_Z"][:n_dim]

        # samples = []
        # for transformed_sample in zip(dataset):
        #     # morph the mesh of the current sample
        #     mesh = transformed_sample._extra_data['init_mesh']

        #     # compute the FE interpolation operator
        #     inv_proj_operator = compute_FE_projection_operator(self.morphed_common_mesh_, mesh)

        #     # initialize sample and add the inverse morphed mesh
        #     sample = Sample()
        #     sample.add_tree(MeshToCGNS(mesh, exportOriginalIDs=False, tagsAsFields=False))
        #     sample = sample.del_all_fields()

        #     # update all features (except nodes and fields) of sample from the ones of current transformed_sample
        #     features_identifiers = extra_sample.get_all_features_identifiers()
        #     # features_identifiers = [feat_id for feat_id in features_identifiers if feat_id.get("type") not in ("nodes", "field")]
        #     features = extra_sample.get_features_from_identifiers(features_identifiers)
        #     sample.update_features_from_identifier(features_identifiers, features, in_place=True)

        #     # update all field of sample by finite element interpolation from the common mesh
        #     field_features_identifiers = transformed_sample.get_all_features_identifiers_by_type("field")
        #     field_features_identifiers = [feat_id for feat_id in field_features_identifiers if feat_id.get("name") not in coord_names]
        #     for field_id in field_features_identifiers:
        #         args = {key: value for key, value in field_id.items() if key != 'type'}
        #         field = transformed_sample.get_feature_from_identifier(field_id)
        #         sample.add_field(field = inv_proj_operator.dot(field), **args)

        #     samples.append(sample)

        # return Dataset.from_list_of_samples(samples)

        return dataset_



class MMGPTransformer(TransformerMixin, BaseEstimator):
    """Adapter for using a scikit-learn transformer on PLAID Datasets.

    Transforms tabular data extracted from homogeneous feature identifiers,
    and returns results as a `Dataset`. Supports forward and inverse transforms.

    Args:
        sklearn_block: A scikit-learn Transformer implementing fit/transform APIs.
        in_features_identifiers: List of feature identifiers to extract input data from.
        out_features_identifiers: List of feature identifiers used for outputs. If None,
            defaults to `in_features_identifiers`.
    """
    # TODO: check if restrict_to_features=True can be used to reduce further memory consumption
    def __init__(
        self,
        in_features_identifiers: list[dict] = None,
    ):
        self.in_features_identifiers = in_features_identifiers

    def fit(self, dataset: Dataset, _y=None):

        # print("self.in_features_identifiers =", self.in_features_identifiers )
        # print(dataset)
        # print(">>", dataset[0]._extra_data)
        # 1./0.

        self.n_dim_ = dataset[0]._extra_data['init_mesh'].GetPointsDimensionality()
        self.coord_names_ = ["coords_X", "coords_Y", "coords_Z"][:self.n_dim_]

        self.in_features_identifiers_ = copy.deepcopy(self.in_features_identifiers)

        return self


    def transform(self, dataset: Dataset):
        check_is_fitted(self, "n_dim_")

        transformed_samples = []
        for sample in dataset:

            proj_operator     = sample._extra_data['proj_operator']

            transformed_sample = sample.copy()

            for feat_id in self.in_features_identifiers:

                feature_type, feature_details = get_feature_type_and_details_from_identifier(feat_id)

                if feature_type == "nodes":
                    coords = sample.get_feature_from_identifier(feat_id).reshape((-1,self.n_dim_))
                    nodes = []
                    for dim in range(self.n_dim_):
                        nodes.append(proj_operator.dot(coords[:,dim]))
                        # transformed_sample.add_field(name = self.coord_names_[dim], field = proj_operator.dot(coords[:,dim]), **feature_details)
                    transformed_sample.set_nodes(nodes = np.stack(nodes).T, **feature_details)
                elif feature_type == "field":
                    field = sample.get_feature_from_identifier(feat_id)
                    transformed_sample.add_field(field = proj_operator.dot(field), **feature_details, warning_overwrite = False)
                else:
                    raise(f"Feature type {feat_id['type']} not compatible with MMGPTransformer")

            transformed_samples.append(transformed_sample)

        transformed_dataset = Dataset.from_list_of_samples(transformed_samples)

        return transformed_dataset


    def inverse_transform(self, dataset: Dataset):
        check_is_fitted(self, "n_dim_")

        samples = []
        for transformed_sample in dataset:

            inv_proj_operator = transformed_sample._extra_data['inv_proj_operator']

            sample = transformed_sample.copy()

            for feat_id in self.in_features_identifiers:

                feature_type, feature_details = get_feature_type_and_details_from_identifier(feat_id)

                if feature_type == "nodes":
                    coords = transformed_sample.get_feature_from_identifier(feat_id).reshape((-1,self.n_dim_))
                    nodes = []
                    for dim in range(self.n_dim_):
                        nodes.append(inv_proj_operator.dot(coords[:,dim]))
                    # nodes = []
                    # for dim in range(self.n_dim_):
                    #     nodes.append(inv_proj_operator.dot(transformed_sample.get_field(name = self.coord_names_[dim], **feature_details)))
                    #     sample.del_field(name = self.coord_names_[dim], **feature_details)
                    sample.set_nodes(nodes = np.stack(nodes).T, **feature_details)
                elif feature_type == "field":
                    field = transformed_sample.get_feature_from_identifier(feat_id)
                    sample.add_field(field = inv_proj_operator.dot(field), **feature_details, warning_overwrite = False)
                else:
                    raise(f"Feature type {feat_id['type']} not compatible with MMGPTransformer")

            samples.append(sample)

        transformed_dataset = Dataset.from_list_of_samples(samples)

        return transformed_dataset

class MeshMorphingInterpolationFieldTransformer(TransformerMixin, BaseEstimator):
    """Adapter for using a scikit-learn transformer on PLAID Datasets.

    Transforms tabular data extracted from homogeneous feature identifiers,
    and returns results as a `Dataset`. Supports forward and inverse transforms.

    Args:
        sklearn_block: A scikit-learn Transformer implementing fit/transform APIs.
        in_features_identifiers: List of feature identifiers to extract input data from.
        out_features_identifiers: List of feature identifiers used for outputs. If None,
            defaults to `in_features_identifiers`.
    """
    # TODO: check if restrict_to_features=True can be used to reduce further memory consumption
    def __init__(
        self,
        common_mesh_id: int = None
    ):
        self.common_mesh_id = common_mesh_id

    def morph(self, mesh):
        mesh_renumb, _, n_boundary = renumber_mesh_for_parametrization(
            mesh, in_place=False)
        mesh_renumb.elemFields = mesh_renumb.nodeFields = {}
        morphed_mesh, _ = floater_mesh_parametrization(
            mesh_renumb, n_boundary)
        return morphed_mesh

    def fit(self, dataset: Dataset, _y=None):
        """Fits the underlying scikit-learn transformer on selected input features.

        Args:
            dataset: A `Dataset` object containing the features to transform.
            _y: Ignored.

        Returns:
            self: The fitted transformer.
        """

        self.common_mesh_id_ = 0 if self.common_mesh_id is None else self.common_mesh_id

        self.morphed_common_mesh_ = self.morph(CGNSToMesh(dataset[self.common_mesh_id_].get_mesh()))
        self.morphed_common_tree_ = MeshToCGNS(self.morphed_common_mesh_, exportOriginalIDs=False, tagsAsFields=False)

        return self

    def transform(self, dataset: Dataset):
        """Applies the fitted transformer to the selected input features.

        Args:
            dataset: A `Dataset` object to transform.

        Returns:
            Dataset: Transformed features wrapped as a new `Dataset`.
        """
        check_is_fitted(self, "morphed_common_mesh_")

        n_dim = self.morphed_common_mesh_.GetPointsDimensionality()
        coord_names = ["coords_X", "coords_Y", "coords_Z"][:n_dim]

        transformed_samples = []
        extra_samples = []
        for sample in dataset:
            # morph the mesh of the current sample
            morphed_mesh = self.morph(CGNSToMesh(sample.get_mesh()))

            # compute the FE interpolation operator
            proj_operator = compute_FE_projection_operator(morphed_mesh, self.morphed_common_mesh_)

            # initialize morph_proj_sample and add the common mesh as geometrical support of morph_proj_sample
            morph_proj_sample = Sample()
            morph_proj_sample.add_tree(copy.deepcopy(self.morphed_common_tree_))

            # update all features (except nodes and fields) of morph_proj_sample from the ones of current sample
            features_identifiers = sample.get_all_features_identifiers()
            features_identifiers = [feat_id for feat_id in features_identifiers if feat_id.get("type") != "nodes"]
            features = sample.get_features_from_identifiers(features_identifiers)
            morph_proj_sample.update_features_from_identifier(features_identifiers, features, in_place=True)

            # add morph_proj_sample coord fields as the node coordinates of current sample
            for nodes_feat_id in sample.get_all_features_identifiers_by_type("nodes"):
                args = {key: value for key, value in nodes_feat_id.items() if key != 'type'}
                coords = sample.get_feature_from_identifier(nodes_feat_id).reshape((-1,n_dim))
                for dim in range(n_dim):
                    morph_proj_sample.add_field(name = coord_names[dim], field = coords[:,dim], **args)

            # update all field of morph_proj_sample by finite element interpolation onto the common mesh
            field_features_identifiers = morph_proj_sample.get_all_features_identifiers_by_type("field")
            projected_field = [proj_operator.dot(field) for field in morph_proj_sample.get_features_from_identifiers(field_features_identifiers)]
            morph_proj_sample.update_features_from_identifier(field_features_identifiers, projected_field, in_place=True)

            # add input nodes coordinate to extra_sample for inverse transform needs
            extra_sample = Sample()
            morphed_tree = MeshToCGNS(morphed_mesh, exportOriginalIDs=False, tagsAsFields=False)
            extra_sample.add_tree(morphed_tree)
            for nodes_feat_id in sample.get_all_features_identifiers_by_type("nodes"):
                args = {key: value for key, value in nodes_feat_id.items() if key != 'type'}
                coords = sample.get_feature_from_identifier(nodes_feat_id).reshape((-1,n_dim))
                for dim in range(n_dim):
                    extra_sample.add_field(name = coord_names[dim], field = coords[:,dim], **args)

            transformed_samples.append(morph_proj_sample)
            extra_samples.append(extra_sample)

        transformed_dataset = Dataset.from_list_of_samples(transformed_samples)
        transformed_dataset.extra_data = Dataset.from_list_of_samples(extra_samples)

        return transformed_dataset


    def inverse_transform(self, dataset: Dataset):
        """Applies inverse transformation to the output features.

        Args:
            dataset: A `Dataset` object with transformed output features.

        Returns:
            Dataset: Dataset with inverse-transformed features.
        """
        check_is_fitted(self, "morphed_common_mesh_")

        n_dim = self.morphed_common_mesh_.GetPointsDimensionality()
        coord_names = ["coords_X", "coords_Y", "coords_Z"][:n_dim]

        samples = []
        for transformed_sample, extra_sample in zip(dataset, dataset.extra_data):
            # morph the mesh of the current sample
            tree = extra_sample.get_mesh()
            mesh = CGNSToMesh(tree)

            # compute the FE interpolation operator
            inv_proj_operator = compute_FE_projection_operator(self.morphed_common_mesh_, mesh)

            nodes = [extra_sample.get_field(cn) for cn in coord_names]
            mesh.nodes = np.vstack(nodes).T

            # initialize sample and add the inverse morphed mesh
            sample = Sample()
            sample.add_tree(MeshToCGNS(mesh, exportOriginalIDs=False, tagsAsFields=False))
            sample = sample.del_all_fields()

            # update all features (except nodes and fields) of sample from the ones of current transformed_sample
            features_identifiers = extra_sample.get_all_features_identifiers()
            # features_identifiers = [feat_id for feat_id in features_identifiers if feat_id.get("type") not in ("nodes", "field")]
            features = extra_sample.get_features_from_identifiers(features_identifiers)
            sample.update_features_from_identifier(features_identifiers, features, in_place=True)

            # update all field of sample by finite element interpolation from the common mesh
            field_features_identifiers = transformed_sample.get_all_features_identifiers_by_type("field")
            field_features_identifiers = [feat_id for feat_id in field_features_identifiers if feat_id.get("name") not in coord_names]
            for field_id in field_features_identifiers:
                args = {key: value for key, value in field_id.items() if key != 'type'}
                field = transformed_sample.get_feature_from_identifier(field_id)
                sample.add_field(field = inv_proj_operator.dot(field), **args)

            samples.append(sample)

        return Dataset.from_list_of_samples(samples)







class MeshMorphingInterpolationTransformerOld(TransformerMixin, BaseEstimator):
    """Adapter for using a scikit-learn transformer on PLAID Datasets.

    Transforms tabular data extracted from homogeneous feature identifiers,
    and returns results as a `Dataset`. Supports forward and inverse transforms.

    Args:
        sklearn_block: A scikit-learn Transformer implementing fit/transform APIs.
        in_features_identifiers: List of feature identifiers to extract input data from.
        out_features_identifiers: List of feature identifiers used for outputs. If None,
            defaults to `in_features_identifiers`.
    """
    # TODO: check if restrict_to_features=True can be used to reduce further memory consumption
    def __init__(
        self,
        common_mesh_id: int = None
    ):
        self.common_mesh_id = common_mesh_id

    def morph(self, mesh):
        mesh_renumb, _, n_boundary = renumber_mesh_for_parametrization(
            mesh, in_place=False)
        mesh_renumb.elemFields = mesh_renumb.nodeFields = {}
        morphed_mesh, _ = floater_mesh_parametrization(
            mesh_renumb, n_boundary)
        return morphed_mesh

    def fit(self, dataset: Dataset, _y=None):
        """Fits the underlying scikit-learn transformer on selected input features.

        Args:
            dataset: A `Dataset` object containing the features to transform.
            _y: Ignored.

        Returns:
            self: The fitted transformer.
        """

        self.common_mesh_id_ = 0 if self.common_mesh_id is None else self.common_mesh_id

        self.morphed_common_mesh_ = self.morph(CGNSToMesh(dataset[self.common_mesh_id_].get_mesh()))
        self.morphed_common_tree_ = MeshToCGNS(self.morphed_common_mesh_, exportOriginalIDs=False, tagsAsFields=False)

        return self

    def transform(self, dataset: Dataset):
        """Applies the fitted transformer to the selected input features.

        Args:
            dataset: A `Dataset` object to transform.

        Returns:
            Dataset: Transformed features wrapped as a new `Dataset`.
        """
        check_is_fitted(self, "morphed_common_mesh_")

        n_dim = self.morphed_common_mesh_.GetPointsDimensionality()
        coord_names = ["coords_X", "coords_Y", "coords_Z"][:n_dim]

        transformed_samples = []
        extra_samples = []
        for sample in dataset:
            # morph the mesh of the current sample
            morphed_mesh = self.morph(CGNSToMesh(sample.get_mesh()))

            # compute the FE interpolation operator
            proj_operator = compute_FE_projection_operator(morphed_mesh, self.morphed_common_mesh_)

            # initialize morph_proj_sample and add the common mesh as geometrical support of morph_proj_sample
            morph_proj_sample = Sample()
            morph_proj_sample.add_tree(copy.deepcopy(self.morphed_common_tree_))

            # update all features (except nodes and fields) of morph_proj_sample from the ones of current sample
            features_identifiers = sample.get_all_features_identifiers()
            features_identifiers = [feat_id for feat_id in features_identifiers if feat_id.get("type") != "nodes"]
            features = sample.get_features_from_identifiers(features_identifiers)
            morph_proj_sample.update_features_from_identifier(features_identifiers, features, in_place=True)

            # add morph_proj_sample coord fields as the node coordinates of current sample
            for nodes_feat_id in sample.get_all_features_identifiers_by_type("nodes"):
                args = {key: value for key, value in nodes_feat_id.items() if key != 'type'}
                coords = sample.get_feature_from_identifier(nodes_feat_id).reshape((-1,n_dim))
                for dim in range(n_dim):
                    morph_proj_sample.add_field(name = coord_names[dim], field = coords[:,dim], **args)

            # update all field of morph_proj_sample by finite element interpolation onto the common mesh
            field_features_identifiers = morph_proj_sample.get_all_features_identifiers_by_type("field")
            projected_field = [proj_operator.dot(field) for field in morph_proj_sample.get_features_from_identifiers(field_features_identifiers)]
            morph_proj_sample.update_features_from_identifier(field_features_identifiers, projected_field, in_place=True)

            # add input nodes coordinate to extra_sample for inverse transform needs
            extra_sample = Sample()
            morphed_tree = MeshToCGNS(morphed_mesh, exportOriginalIDs=False, tagsAsFields=False)
            extra_sample.add_tree(morphed_tree)
            for nodes_feat_id in sample.get_all_features_identifiers_by_type("nodes"):
                args = {key: value for key, value in nodes_feat_id.items() if key != 'type'}
                coords = sample.get_feature_from_identifier(nodes_feat_id).reshape((-1,n_dim))
                for dim in range(n_dim):
                    extra_sample.add_field(name = coord_names[dim], field = coords[:,dim], **args)

            transformed_samples.append(morph_proj_sample)
            extra_samples.append(extra_sample)

        transformed_dataset = Dataset.from_list_of_samples(transformed_samples)
        transformed_dataset.extra_data = Dataset.from_list_of_samples(extra_samples)

        return transformed_dataset


    def inverse_transform(self, dataset: Dataset):
        """Applies inverse transformation to the output features.

        Args:
            dataset: A `Dataset` object with transformed output features.

        Returns:
            Dataset: Dataset with inverse-transformed features.
        """
        check_is_fitted(self, "morphed_common_mesh_")

        n_dim = self.morphed_common_mesh_.GetPointsDimensionality()
        coord_names = ["coords_X", "coords_Y", "coords_Z"][:n_dim]

        samples = []
        for transformed_sample, extra_sample in zip(dataset, dataset.extra_data):
            # morph the mesh of the current sample
            tree = extra_sample.get_mesh()
            mesh = CGNSToMesh(tree)

            # compute the FE interpolation operator
            inv_proj_operator = compute_FE_projection_operator(self.morphed_common_mesh_, mesh)

            nodes = [extra_sample.get_field(cn) for cn in coord_names]
            mesh.nodes = np.vstack(nodes).T

            # initialize sample and add the inverse morphed mesh
            sample = Sample()
            sample.add_tree(MeshToCGNS(mesh, exportOriginalIDs=False, tagsAsFields=False))
            sample = sample.del_all_fields()

            # update all features (except nodes and fields) of sample from the ones of current transformed_sample
            features_identifiers = extra_sample.get_all_features_identifiers()
            # features_identifiers = [feat_id for feat_id in features_identifiers if feat_id.get("type") not in ("nodes", "field")]
            features = extra_sample.get_features_from_identifiers(features_identifiers)
            sample.update_features_from_identifier(features_identifiers, features, in_place=True)

            # update all field of sample by finite element interpolation from the common mesh
            field_features_identifiers = transformed_sample.get_all_features_identifiers_by_type("field")
            field_features_identifiers = [feat_id for feat_id in field_features_identifiers if feat_id.get("name") not in coord_names]
            for field_id in field_features_identifiers:
                args = {key: value for key, value in field_id.items() if key != 'type'}
                field = transformed_sample.get_feature_from_identifier(field_id)
                sample.add_field(field = inv_proj_operator.dot(field), **args)

            samples.append(sample)

        return Dataset.from_list_of_samples(samples)


def compute_FE_projection_operator(origin_mesh: Mesh, target_mesh: Mesh,
                                    verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Finite Element Projection Operators.

    Args:
        origin_mesh (Mesh): The original mesh data.
        target_mesh (Mesh): The target mesh data.
        verbose (bool, optional): Whether to display verbose output. Defaults to False.

    Returns:
        tuple(np.ndarray, np.ndarray): A tuple containing two projection operators (projOperator, invProjOperator).
    """
    start = time.time()
    if verbose is True: # pragma: no cover
        print("Computing direct FE interpolation operator:")
    space, numberings, _, _ = FT.PrepareFEComputation(
        origin_mesh, numberOfComponents=1)
    input_FE_field = FF.FEField(
        name="dummy",
        mesh=origin_mesh,
        space=space,
        numbering=numberings[0])
    proj_operator = MFO.GetFieldTransferOp(
        input_FE_field, target_mesh.nodes, method="Interp/Clamp", verbose=verbose)[0]

    if verbose is True: # pragma: no cover
        print(
            "Duration computation of FE interpolation operator: {:#.6g} s".format(
                time.time() -
                start))

    return proj_operator


# def compute_FE_projection_operators(origin_mesh: Mesh, target_mesh: Mesh,
#                                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
#     """Compute Finite Element Projection Operators.

#     Args:
#         origin_mesh (Mesh): The original mesh data.
#         target_mesh (Mesh): The target mesh data.
#         verbose (bool, optional): Whether to display verbose output. Defaults to False.

#     Returns:
#         tuple(np.ndarray, np.ndarray): A tuple containing two projection operators (projOperator, invProjOperator).
#     """
#     start = time.time()
#     if verbose is True: # pragma: no cover
#         print("Computing direct FE interpolation operator:")
#     space, numberings, _, _ = FT.PrepareFEComputation(
#         origin_mesh, numberOfComponents=1)
#     input_FE_field = FF.FEField(
#         name="dummy",
#         mesh=origin_mesh,
#         space=space,
#         numbering=numberings[0])
#     proj_operator = MFO.GetFieldTransferOp(
#         input_FE_field, target_mesh.nodes, method="Interp/Clamp", verbose=verbose)[0]

#     if verbose is True: # pragma: no cover
#         print("Computing inverse FE interpolation operator:")
#     space, numberings, _, _ = FT.PrepareFEComputation(
#         target_mesh, numberOfComponents=1)
#     input_FE_field = FF.FEField(
#         name="dummy",
#         mesh=target_mesh,
#         space=space,
#         numbering=numberings[0])
#     inv_proj_operator = MFO.GetFieldTransferOp(
#         input_FE_field, origin_mesh.nodes, method="Interp/Clamp", verbose=verbose)[0]

#     if verbose is True: # pragma: no cover
#         print(
#             "Duration computation of FE interpolation operators: {:#.6g} s".format(
#                 time.time() -
#                 start))

#     return proj_operator, inv_proj_operator


def compute_node_to_node_graph(
        in_mesh: Mesh, dimensionality: int = None, dist_func: Callable=None) -> networkx.Graph:
    '''Creates a networkx graph from the node connectivity on a Mesh through edges

    Parameters
    ----------
    in_mesh : Mesh
        input mesh
    dimensionality : int
        dimension of the elements considered to initalize the graph
    dist_func : func
        function applied to the lengh of the edges of the mesh, and attached of the
        corresponding edge of the graph of the mesh

    Returns
    -------
    networkx.Graph
        Element to element graph
    '''
    if dimensionality is None: # pragma: no cover
        dimensionality = in_mesh.GetDimensionality()

    if dist_func is None:
        def dist_func(x):
            return x

    el_filter = FilterObjects.ElementFilter(dimensionality=dimensionality)
    mesh = MIT.ExtractElementsByElementFilter(in_mesh, el_filter)

    node_connectivity, _ = MIT.ComputeNodeToNodeConnectivity(mesh)

    G = initialize_graph_points_from_mesh_points(in_mesh)
    edges = []
    for i in range(node_connectivity.shape[0]):
        for j in node_connectivity[i][node_connectivity[i] > i]:
            length = np.linalg.norm(in_mesh.nodes[i] - in_mesh.nodes[j])
            edges.append((i, j, dist_func(length)))
    G.add_weighted_edges_from(edges)

    return G


def initialize_graph_points_from_mesh_points(
        in_mesh: Mesh) -> networkx.Graph:
    '''Initializes a networkx graph with nodes consistant with the number of nodes of a Mesh.
    This enables further edge addition compatible with the connectivity of the elements of the Mesh.

    Parameters
    ----------
    in_mesh : Mesh
        input mesh

    Returns
    -------
    networkx.Graph
        initialized graph
    '''
    G = networkx.Graph()
    G.add_nodes_from(np.arange(in_mesh.GetNumberOfNodes()))
    return G


def renumber_mesh_for_parametrization(in_mesh: Mesh, in_place: bool = True, boundary_orientation: str = "direct",
                                      fixed_boundary_points: list = None, starting_point_rank_on_boundary: int = None) -> Tuple[Mesh, np.ndarray, int]:
    """
    Only for linear triangle meshes
    Renumber the node IDs, such that the points on the boundary are placed at the
    end of the numbering. Serves as a preliminary step for mesh parametrization.

    Parameters
    ----------
    in_mesh : Mesh
        input triangular to be renumbered
    in_place : bool
        if "True", in_mesh is modified
        if "False", in_mesh is let unmodified, and a new mesh is produced
    boundary_orientation : str
        if "direct, the boundary of the parametrisation is constructed in the direct trigonometric order
        if "indirect", the boundary of the parametrisation is constructed in the indirect trigonometric orderc order
    fixed_boundary_points : list
        list containing lists of two np.ndarrays. Each 2-member list is used to identify one
        point on the boundary: the first array contains the specified components, and the second the
    starting_point_rank_on_boundary : int
        node id (in the complete mesh) of the point on the boundary where the mapping starts

    Returns
    -------
    Mesh
        renumbered mesh
    ndarray(1) of ints
        renumbering of the nodes of the returned renumbered mesh, with respect to in_mesh
    int
        number of node of the boundary of in_mesh

    """
    # assert mesh of linear triangles
    for name, data in in_mesh.elements.items():
        name == ED.Triangle_3

    if in_place == True:
        mesh = in_mesh
    else:
        import copy
        mesh = copy.deepcopy(in_mesh)

    # Retrieve the elements of the line boundary
    skin = MMT.ComputeSkin(mesh, md=2)
    skin.ComputeBoundingBox()

    # Create a path linking nodes of the line boundary, starting with the node with smallest coordinates
    # and going in the direction increasing the value of the second coordinate
    # the least

    bars = skin.elements[ED.Bar_2].connectivity

    node_graph_0 = compute_node_to_node_graph(skin, dimensionality=1)
    node_graph = [list(node_graph_0[i].keys())
                  for i in range(node_graph_0.number_of_nodes())]

    indices_bars = np.sort(np.unique(bars.flatten()))

    if fixed_boundary_points is None:

        if starting_point_rank_on_boundary is None:
            vec = in_mesh.nodes[indices_bars, 0]
            indices_nodes_X_min = vec == vec[np.argmin(vec)]
            nodes_X_min = in_mesh.nodes[indices_bars[indices_nodes_X_min], :]

            indices_nodes_min = nodes_X_min[:, 1] == nodes_X_min[np.argmin(
                nodes_X_min[:, 1]), 1]
            nodesmin = nodes_X_min[indices_nodes_min, :]

            if in_mesh.GetPointsDimensionality() == 3: # pragma: no cover
                indices_nodes_min = nodesmin[:, 2] == nodesmin[np.argmin(
                    nodesmin[:, 2]), 2]
                nodesmin = nodesmin[indices_nodes_min, :]

            index_in_bars = np.where(
                (in_mesh.nodes[indices_bars, :] == nodesmin).all(axis=1))[0]
            assert index_in_bars.shape == (1,)
            index_in_bars = index_in_bars[0]
            assert (
                in_mesh.nodes[indices_bars[index_in_bars], :] == nodesmin).all()

            p_min = indices_bars[index_in_bars]

        else: # pragma: no cover
            p_min = starting_point_rank_on_boundary
        # print("starting walking along line boundary at point... =", str(in_mesh.nodes[p_min,:]), " of rank:", str(p_min))

    else: # pragma: no cover
        inds, point = fixed_boundary_points[0][0], fixed_boundary_points[0][1]
        index_in_bars = (np.linalg.norm(np.subtract(
            in_mesh.nodes[indices_bars, :][:, inds], point), axis=1)).argmin()
        p_min = indices_bars[index_in_bars]
        # print("starting walking along line boundary at point... =", str(in_mesh.nodes[p_min,:]), " of rank:", str(p_min))

    p1 = p1init = p_min
    p2_candidate = [node_graph[p_min][0], node_graph[p_min][1]]

    if fixed_boundary_points is None:
        # choose direction
        p2 = p2_candidate[np.argmin(np.asarray(
            [in_mesh.nodes[p2_candidate[0], 1], in_mesh.nodes[p2_candidate[1], 1]]))]

    else:
        # choose direction from second point set on boundary
        inds = fixed_boundary_points[1][0]
        delta_fixedBoundaryPoints = fixed_boundary_points[1][1] - \
            fixed_boundary_points[0][1]
        delta_fixedBoundaryPoints /= np.linalg.norm(delta_fixedBoundaryPoints)

        delta_candidate = np.asarray(
            [in_mesh.nodes[p2c, inds] - in_mesh.nodes[p_min, inds] for p2c in p2_candidate])
        delta_candidate[0] /= np.linalg.norm(delta_candidate[0])
        delta_candidate[1] /= np.linalg.norm(delta_candidate[1])

        error_delta_candidate = []
        error_delta_candidate.append(
            np.subtract(
                delta_candidate[0],
                delta_fixedBoundaryPoints))
        error_delta_candidate.append(
            np.subtract(
                delta_candidate[1],
                delta_fixedBoundaryPoints))

        p2 = p2_candidate[np.linalg.norm(
            error_delta_candidate, axis=1).argmin()]

    # print("... walking toward point =", str(in_mesh.nodes[p2,:]), " of rank:", str(p2))

    path = [p1, p2]
    while p2 != p1init:
        p2save = p2
        temp_array = np.asarray(node_graph[p2])
        p2 = temp_array[temp_array != p1][0]
        p1 = p2save
        path.append(p2)
    path = path[:-1]

    if boundary_orientation == "indirect": # pragma: no cover
        path = path[::-1]

    # Renumber the node, keeping at the end the continuous path along the line
    # boundary
    N = mesh.GetNumberOfNodes()
    n_boundary = len(path)

    init_order = np.arange(N)
    interior_numberings = np.delete(init_order, path)

    renumb = np.hstack((interior_numberings, path))

    assert len(renumb) == N

    # UMMT.RenumberNodes(mesh, renumb) # Deprecated (not in Muscat)
    MMT.NodesPermutation(mesh, renumb)

    """inv_renumb = np.argsort(renumb)

    mesh.nodes = mesh.nodes[renumb,:]
    for _, data in mesh.elements.items():
        data.connectivity = inv_renumb[data.connectivity]
    mesh.ConvertDataForNativeTreatment()"""

    return mesh, renumb, n_boundary


def floater_mesh_parametrization(in_mesh: Mesh, n_boundary: int, out_shape: str = "circle", boundary_orientation: str = "direct",
                                 curv_abs_boundary: bool = True, fixed_interior_points: dict[str, list] = None, fixed_boundary_points: list = None) -> Tuple[Mesh, dict[str, float]]:
    """
    STILL LARGELY EXPERIMENTAL

    Only for linear triangular meshes

    Computes the Mesh Parametrization algorithm [1] proposed by Floater,
    in the case of target parametrization fitted to the unit 2D circle (R=1) or square (L=1).
    Adapted for ML need: the out_shape's boundary is sampled following the curvilinear abscissa along
    the boundary on in_mesh (only for out_shape = "circle" for the moment)

    Parameters
    ----------
    in_mesh : Mesh
        Renumbered triangular mesh to parametrize
    n_boundary : int
        number nodes on the line boundary
    out_shape : str
        if "circle", the boundary of in_mesh is mapped into the unit circle
        if "square", the boundary of in_mesh is mapped into the unit square
    boundary_orientation : str
        if "direct, the boundary of the parametrisation is constructed in the direct trigonometric order
        if "indirect", the boundary of the parametrisation is constructed in the indirect trigonometric order
    curv_abs_boundary : bool
        only if fixed_interior_points = None
        if True, the point density on the boundary of out_shape is the same as the point density on the boundary of in_mesh
        if False, the point density on the boundary is uniform
    fixed_interior_points : dict
        with one key, and corresponding value, a list: [ndarray(n), ndarray(n,2)],
        with n the number of interior points to be fixed; the first ndarray is the index of the considered
        interior point, the second ndarray is the corresponding prescribed positions
        if key is "mean", the interior points are displaced by the mean of the prescribed positions
        if key is "value", the interior points are displaced by the value of the prescribed positions
    fixed_boundary_points: list
        list of lists: [ndarray(2), ndarray(2)], helping definining a point in in_mesh; the first ndarray is the component
        of a point on the boundary, and the second array is the value of corresponding component. Tested for triangular meshes
        in the 3D space.

    Returns
    -------
    Mesh
        parametrization of mesh
    dict
        containing 3 keys: "minEdge", "maxEdge" and "weights", with values floats containing the minimal
        and maximal edged length of the parametrized mesh, and the weights (lambda) in the Floater algorithm

    Attention
    -----
        mesh must be a renumbered Mesh of triangles (either in
        a 2D or 3D ambiant space), with a line boundary (no closed surface in 3D).
        out_shape = "circle" is more robust in the sense that is in_mesh has a 2D square-like,
        for triangles may ended up flat with  out_shape = "square"

    References
    ----------
        [1] M. S. Floater. Parametrization and smooth approximation of surface
        triangulations, 1997. URL: https://www.sciencedirect.com/science/article/abs/pii/S0167839696000313
    """
    import copy
    mesh = copy.deepcopy(in_mesh)

    N = mesh.GetNumberOfNodes()
    n = N - n_boundary

    u = np.zeros((mesh.nodes.shape[0], 2))

    if out_shape == "square":
        print("!!! Warning, the implmentation out_shape == 'square' is *very* experimental !!!")
        if boundary_orientation == "indirect":
            raise NotImplementedError(
                "Cannot use 'square' out_shape with 'indirect' boundary_orientation")
        if fixed_interior_points is not None:
            raise NotImplementedError(
                "Cannot use 'square' out_shape with fixed_interior_points not None")
        if fixed_boundary_points is not None:
            raise NotImplementedError(
                "Cannot use 'square' out_shape with fixed_boundary_points not None")

        # Set the boundary on the parametrization on the unit square
        L = n_boundary // 4
        r = n_boundary % 4

        u[n:n + L, 0] = np.linspace(1 / L, 1, L)
        u[n:n + L, 1] = 0.
        u[n + L:n + 2 * L, 0] = 1.
        u[n + L:n + 2 * L, 1] = np.linspace(1 / L, 1, L)
        u[n + 2 * L:n + 3 * L, 0] = np.linspace(1 - 1 / L, 0, L)
        u[n + 2 * L:n + 3 * L, 1] = 1.
        u[n + 3 * L:n + 4 * L + r, 0] = 0.
        u[n + 3 * L:n + 4 * L + r,
            1] = np.linspace(1 - 1 / (L + r), 0, (L + r))

    elif out_shape == "circle":
        # Set the boundary on the parametrization on the unit circle

        length_along_boundary = [0]
        cumulative_length = 0.
        indices = np.arange(n + 1, N)
        for i in indices:
            p1 = mesh.nodes[i - 1, :]
            p2 = mesh.nodes[i, :]
            cumulative_length += np.linalg.norm(p2 - p1)
            length_along_boundary.append(cumulative_length)
        length_along_boundary = np.asarray(length_along_boundary)

        if fixed_boundary_points is not None:
            fixed_ranks_on_boundary = [0]
            n_fixed_points_on_boundary = 1
            for fixed_boundary_point in fixed_boundary_points[1:]:
                inds, point = fixed_boundary_point[0], fixed_boundary_point[1]
                # index_in_bars = np.where((in_mesh.nodes[n:,:][:,inds] == point).all(axis=1))[0]
                index_in_bars = (np.linalg.norm(np.subtract(
                    in_mesh.nodes[n:, :][:, inds], point), axis=1)).argmin()

                fixed_ranks_on_boundary.append(index_in_bars)
                n_fixed_points_on_boundary += 1
            fixed_ranks_on_boundary.append(-1)

            angles = []
            delta_angle = 2 * np.pi / n_fixed_points_on_boundary
            # print("delta_angle =", delta_angle)
            for k in range(n_fixed_points_on_boundary):
                save_length = length_along_boundary[fixed_ranks_on_boundary[k]:fixed_ranks_on_boundary[k + 1]]
                delta_length_along_boundary = save_length - \
                    length_along_boundary[fixed_ranks_on_boundary[k]]
                delta_unit_length_along_boundary = delta_length_along_boundary / \
                    (length_along_boundary[fixed_ranks_on_boundary[k + 1]
                                           ] - length_along_boundary[fixed_ranks_on_boundary[k]])
                res = (k + delta_unit_length_along_boundary) * delta_angle
                angles = np.hstack((angles, res))

            angles = np.hstack((angles, 2. * np.pi))

        else:
            if curv_abs_boundary == True:
                angles = (2 * np.pi) * (1 - 1 / n_boundary) * \
                    length_along_boundary / cumulative_length
            else:
                angles = np.linspace(
                    2 * np.pi / n_boundary, 2 * np.pi, n_boundary)

        if boundary_orientation == "direct":
            for i, a in enumerate(angles):
                u[n + i, 0] = np.cos(a)
                u[n + i, 1] = np.sin(a)
        else: # pragma: no cover
            for i, a in enumerate(angles):
                u[n + i, 0] = np.cos(a)
                u[n + i, 1] = -np.sin(a)

    else: # pragma: no cover
        raise NotImplementedError(
            "out_shape" +
            str(out_shape) +
            " not implemented")

    # Compute a node graphe for the mesh
    edges = set()
    el_filter = FilterObjects.ElementFilter(
         dimensionality=2, elementType=[
            ED.Triangle_3])
    for name, data, ids in el_filter(mesh):
        for face in ED.faces1[name]:
            for idd in ids:
                edge = np.sort(data.connectivity[idd][face[1]])
                edges.add((edge[0], edge[1]))

    G2 = initialize_graph_points_from_mesh_points(mesh)
    for edge in edges:
        G2.add_edge(edge[0], edge[1])

    # Compute the weights of each node of the mesh (number of edges linked to
    # each node): the inverse of the degrees
    ad = networkx.adjacency_matrix(G2)

    weights = np.zeros(N)
    for i in range(N):
        weights[i] = 1. / np.sum(ad[[i], :])

    # Construct the sparse linear system to solve to find the position of the
    # interior points in the parametrization
    A = sparse.eye(n).tolil()
    RHS_mat = sparse.lil_matrix((n, N))
    for edge in edges:
        for edg in [(edge[0], edge[1]), (edge[1], edge[0])]:
            if edg[0] < n and edg[1] < n:
                A[edg[0], edg[1]] = -weights[edg[0]]
            elif edg[0] < n:
                RHS_mat[edg[0], edg[1]] = weights[edg[0]]

    RHS = RHS_mat.dot(u)
    A = A.tocsr()

    # update the position of the interior points
    res = sparse.linalg.spsolve(A, RHS)
    u[:n, :] = res

    if fixed_interior_points is not None:
        mesh.nodes = u
        mesh.ConvertDataForNativeTreatment()

        displacement = None
        mask = None

        if "mean" in fixed_interior_points:

            mean_pos = np.mean(u[fixed_interior_points["mean"][0], :], axis=0)

            if displacement is None:
                displacement = - \
                    np.tile(
                        mean_pos, (fixed_interior_points["mean"][1].shape[0], 1))
            else:
                displacement = np.vstack(
                    (displacement, -np.tile(mean_pos, (fixed_interior_points["mean"][1].shape[0], 1))))

            if mask is None:
                mask = fixed_interior_points["mean"][0]
            else:
                mask = np.hstack((mask, fixed_interior_points["mean"][0]))

        if "value" in fixed_interior_points:

            if displacement is None:
                displacement = fixed_interior_points["value"][1] - \
                    u[fixed_interior_points["value"][0], :]
            else:
                displacement = np.vstack(
                    (displacement, fixed_interior_points["value"][1] - u[fixed_interior_points["value"][0], :]))

            if mask is None:
                mask = fixed_interior_points["value"][0]
            else:
                mask = np.hstack((mask, fixed_interior_points["value"][0]))

        if displacement is not None and mask is not None:
            displacement = np.vstack((displacement, np.zeros((N - n, 2))))
            mask = np.hstack((mask, np.arange(n, N)))

            new_nodes = MMT.Morphing(mesh, displacement, mask, radius=1.)
            mesh.nodes = new_nodes

    else:
        mesh.nodes = u
        mesh.ConvertDataForNativeTreatment()

    infos = {}
    endge_lengths = []
    for edge in edges:
        endge_lengths.append(np.linalg.norm(
            mesh.nodes[edge[1], :] - mesh.nodes[edge[0], :]))

    infos = {
        "minEdge": np.min(endge_lengths),
        "maxEdge": np.max(endge_lengths),
        "weights": weights}

    return mesh, infos
