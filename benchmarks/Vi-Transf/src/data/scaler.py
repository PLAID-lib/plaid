from copy import deepcopy

import torch
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from torch_geometric.data import Data


class Scaler:
    def __init__(self, *args, **kwargs):
        pass

    def partial_fit(self, *args, **kwargs):
        pass

    def fit(self, dataset: list[Data]) -> None:
        pass

    def transform(self, dataset: list[Data]) -> list[Data]:
        return dataset

    def fit_transform(self, dataset: list[Data]) -> list[Data]:
        return dataset

    def inverse_transform(self, dataset: list[Data]) -> list[Data]:
        return dataset


class NoScaler(Scaler):
    def __init__(self, *args, **kwargs):
        super().__init__()


class StandardScaler(Scaler):
    """StandardScaler class for scaling and inverse scaling of node features, output scalars, and output fields in
    PyTorch Geometric datasets.

    Args:
        scalers (Union[str, list[str]], optional): A string representing a scaler name, or a list of three scalers
            for node features, output scalars, and output fields.
            Defaults to "StandardScaler".
    """

    def __init__(self):
        self.scalers = [SklearnStandardScaler for _ in range(6)]

    def _reset(self):
        """Resets the scalers for node features, output scalars, and output fields."""
        (
            self.xf_scaler,
            self.xs_scaler,
            self.yf_scaler,
            self.ys_scaler,
            self.edge_attr_scaler,
            self.edge_weight_scaler,
        ) = [s() for s in self.scalers]

    def fit(self, dataset: list[Data]):
        """Fits the scalers to the provided dataset. Each Data object in the dataset must have
        `x` and `input_scalars`. `output_scalars` and `output_fields` are optional.

        Args:
            dataset (list[Data]): A list of PyTorch Geometric Data objects.
        """
        self._reset()

        for data in dataset:
            self.xf_scaler.partial_fit(data.x)

            if (
                hasattr(data, "input_scalars")
                and isinstance(data.input_scalars, torch.Tensor)
                and len(data.input_scalars.ravel()) > 0
            ):
                self.xs_scaler.partial_fit(data.input_scalars)

            # Check if output_fields exists before fitting
            if (
                hasattr(data, "output_fields")
                and isinstance(data.output_fields, torch.Tensor)
                and len(data.output_fields.ravel()) > 0
            ):
                self.yf_scaler.partial_fit(data.output_fields)

            # Check if output_scalars exists before fitting
            if (
                hasattr(data, "output_scalars")
                and isinstance(data.output_scalars, torch.Tensor)
                and len(data.output_scalars.ravel()) > 0
            ):
                self.ys_scaler.partial_fit(data.output_scalars)

            if data.edge_attr is not None:
                self.edge_attr_scaler.partial_fit(data.edge_attr)
            if data.edge_weight is not None:
                self.edge_weight_scaler.partial_fit(data.edge_weight.reshape(-1, 1))

    def transform(self, dataset: list[Data]):
        """Transforms the dataset based on the fitted scalers.

        Args:
            dataset (list[Data]): A list of PyTorch Geometric Data objects to be transformed.
        """
        dataset = deepcopy(dataset)

        for data in dataset:
            xf_dtype = data.x.dtype

            data.x = torch.tensor(
                self.xf_scaler.transform(data.x),
                dtype=xf_dtype,
            )
            if (
                hasattr(data, "input_scalars")
                and isinstance(data.input_scalars, torch.Tensor)
                and len(data.input_scalars.ravel()) > 0
            ):
                xs_dtype = data.input_scalars.dtype
                data.input_scalars = torch.tensor(
                    self.xs_scaler.transform(data.input_scalars),
                    dtype=xs_dtype,
                )

            # Transform output_fields if it exists
            if (
                hasattr(data, "output_fields")
                and isinstance(data.output_fields, torch.Tensor)
                and len(data.output_fields.ravel()) > 0
            ):
                yf_dtype = data.output_fields.dtype
                data.output_fields = torch.tensor(
                    self.yf_scaler.transform(data.output_fields),
                    dtype=yf_dtype,
                )

            # Transform output_scalars if it exists
            if (
                hasattr(data, "output_scalars")
                and isinstance(data.output_scalars, torch.Tensor)
                and len(data.output_scalars.ravel()) > 0
            ):
                ys_dtype = data.output_scalars.dtype
                data.output_scalars = torch.tensor(
                    self.ys_scaler.transform(data.output_scalars),
                    dtype=ys_dtype,
                )

            if (
                isinstance(data.edge_attr, torch.Tensor)
                and len(data.edge_attr.ravel()) > 0
            ):
                edge_attr_dtype = data.edge_attr.dtype
                data.edge_attr = torch.as_tensor(
                    self.edge_attr_scaler.transform(data.edge_attr),
                    dtype=edge_attr_dtype,
                )

            if (
                isinstance(data.edge_weight, torch.Tensor)
                and len(data.edge_weight.ravel()) > 0
            ):
                edge_weight_dtype = data.edge_weight.dtype
                data.edge_weight = torch.as_tensor(
                    self.edge_weight_scaler.transform(data.edge_weight.reshape(-1, 1)),
                    dtype=edge_weight_dtype,
                ).reshape(-1)

        return dataset

    def fit_transform(self, dataset: list[Data]):
        """Fits the scalers and then transforms the dataset. A combination of `fit` and `transform`.

        Args:
            dataset (list[Data]): A list of PyTorch Geometric Data objects.

        Returns:
            list[Data]: Transformed dataset.
        """
        self.fit(dataset)
        return self.transform(dataset)

    def inverse_transform(self, dataset: list[Data] | Data):
        """Reverts the scaling transformation applied on the dataset, bringing the data back to its original scale.

        Args:
            dataset (list[Data] | Data): A list or single PyTorch Geometric Data object to be inverse transformed.

        Returns:
            list[Data] | Data: Inverse transformed dataset or data point.
        """
        dataset = deepcopy(dataset)
        is_data = False
        if isinstance(dataset, Data):
            dataset = [dataset]
            is_data = True

        for data in dataset:
            xf_dtype = data.x.dtype

            data.x = torch.tensor(
                self.xf_scaler.inverse_transform(data.x),
                dtype=xf_dtype,
            )

            if (
                hasattr(data, "input_scalars")
                and isinstance(data.input_scalars, torch.Tensor)
                and len(data.input_scalars.ravel()) > 0
            ):
                xs_dtype = data.input_scalars.dtype
                data.input_scalars = torch.tensor(
                    self.xs_scaler.inverse_transform(data.input_scalars),
                    dtype=xs_dtype,
                )

            # Inverse transform output_fields if it exists
            if (
                hasattr(data, "output_fields")
                and isinstance(data.output_fields, torch.Tensor)
                and len(data.output_fields.ravel()) > 0
            ):
                yf_dtype = data.output_fields.dtype
                data.output_fields = torch.tensor(
                    self.yf_scaler.inverse_transform(data.output_fields),
                    dtype=yf_dtype,
                )

            # Inverse transform output_scalars if it exists
            if (
                hasattr(data, "output_scalars")
                and isinstance(data.output_scalars, torch.Tensor)
                and len(data.output_scalars.ravel()) > 0
            ):
                ys_dtype = data.output_scalars.dtype
                data.output_scalars = torch.tensor(
                    self.ys_scaler.inverse_transform(data.output_scalars),
                    dtype=ys_dtype,
                )

            if (
                isinstance(data.edge_attr, torch.Tensor)
                and len(data.edge_attr.ravel()) > 0
            ):
                edge_attr_dtype = data.edge_attr.dtype
                data.edge_attr = torch.as_tensor(
                    self.edge_attr_scaler.inverse_transform(data.edge_attr),
                    dtype=edge_attr_dtype,
                )

            if (
                isinstance(data.edge_weight, torch.Tensor)
                and len(data.edge_weight.ravel()) > 0
            ):
                edge_weight_dtype = data.edge_weight.dtype
                data.edge_weight = torch.as_tensor(
                    self.edge_weight_scaler.inverse_transform(
                        data.edge_weight.reshape(-1, 1)
                    ),
                    dtype=edge_weight_dtype,
                ).reshape(-1)

            if (
                hasattr(data, "fields_prediction")
                and isinstance(data.fields_prediction, torch.Tensor)
                and len(data.fields_prediction.ravel()) > 0
            ):
                data.fields_prediction = torch.tensor(
                    self.yf_scaler.inverse_transform(data.fields_prediction),
                    dtype=data.fields_prediction.dtype,
                )

            if (
                hasattr(data, "scalars_prediction")
                and isinstance(data.scalars_prediction, torch.Tensor)
                and len(data.scalars_prediction.ravel()) > 0
            ):
                data.scalars_prediction = torch.tensor(
                    self.ys_scaler.inverse_transform(data.scalars_prediction),
                    dtype=data.scalars_prediction.dtype,
                )

        if is_data:
            return dataset[0]

        return dataset

    def inverse_transform_prediction(self, dataset: list[Data]):
        """Inverse transforms the output_fields_prediction attribute of the dataset.

        Args:
            dataset (list[Data]): A list of PyTorch Geometric Data objects.

        Returns:
            list[Data]: Dataset with inverse transformed predictions.
        """
        for data in dataset:
            if (
                hasattr(data, "fields_prediction")
                and isinstance(data.fields_prediction, torch.Tensor)
                and len(data.fields_prediction.ravel()) > 0
            ):
                data.fields_prediction = torch.tensor(
                    self.yf_scaler.inverse_transform(
                        data.fields_prediction.detach().cpu().numpy()
                    ),
                    dtype=data.fields_prediction.dtype,
                )

            if (
                hasattr(data, "scalars_prediction")
                and isinstance(data.scalars_prediction, torch.Tensor)
                and len(data.scalars_prediction.ravel()) > 0
            ):
                data.scalars_prediction = torch.tensor(
                    self.ys_scaler.inverse_transform(
                        data.scalars_prediction.detach().cpu().numpy().reshape(1, -1)
                    ),
                    dtype=data.scalars_prediction.dtype,
                )

        return dataset
