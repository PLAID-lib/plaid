"""Visualization utilities for analyzing datasets."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from __future__ import annotations

# %% Imports
import logging
from typing import TYPE_CHECKING, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from plaid.containers.dataset import Dataset
    from plaid.containers.sample import Sample

logger = logging.getLogger(__name__)


# %% Functions


def scatter_plot(
    dataset: Union[Dataset, list[Sample]],
    feature_names: Optional[list[str]] = None,
    sample_ids: Optional[list[int]] = None,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    max_features_per_plot: int = 6,
    **kwargs,
) -> Union[Figure, list[Figure]]:
    """Create scatter plots of feature values vs sample IDs.

    This function visualizes how feature values vary across samples in the dataset.
    Each feature is plotted as a scatter plot with sample IDs on the x-axis and
    feature values on the y-axis. Useful for detecting trends, outliers, or patterns.

    Args:
        dataset (Union[Dataset, list[Sample]]): The dataset or list of samples to visualize.
        feature_names (list[str], optional): List of feature names to plot. If None, plots all scalar features.
            For field features, use the format "base_name/zone_name/location/field_name". Defaults to None.
        sample_ids (list[int], optional): List of sample IDs to include. If None, uses all samples. Defaults to None.
        figsize (tuple[float, float], optional): Figure size (width, height) in inches.
            If None, automatically calculated based on number of features. Defaults to None.
        title (str, optional): Main title for the plot. Defaults to None.
        max_features_per_plot (int, optional): Maximum number of features to display in a single figure.
            If more features are requested, multiple figures are created. Defaults to 6.
        **kwargs: Additional keyword arguments passed to matplotlib's scatter function.

    Returns:
        Union[plt.Figure, list[plt.Figure]]: The created figure(s). Returns a single figure if all features
            fit in one plot, otherwise returns a list of figures.

    Raises:
        TypeError: If dataset is not a Dataset or list[Sample].
        ValueError: If no features are found or feature_names contains invalid names.

    Example:
        >>> from plaid import Dataset
        >>> from plaid.utils.viz import scatter_plot
        >>> dataset = Dataset("path/to/dataset")
        >>> # Plot all scalar features
        >>> scatter_plot(dataset)
        >>> # Plot specific features
        >>> scatter_plot(dataset, feature_names=["temperature", "pressure"])
        >>> # Customize appearance
        >>> scatter_plot(dataset, feature_names=["velocity"], figsize=(12, 6), alpha=0.6)
    """
    # Lazy import to avoid circular dependency
    from plaid.containers.dataset import Dataset

    # Input validation
    if isinstance(dataset, list):
        # Convert list of samples to Dataset for easier handling
        temp_dataset = Dataset()
        temp_dataset.add_samples(dataset)
        dataset = temp_dataset
    elif not isinstance(dataset, Dataset):
        raise TypeError(
            f"dataset must be a Dataset or list[Sample], got {type(dataset)}"
        )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    # Get sample IDs
    if sample_ids is None:
        sample_ids = dataset.get_sample_ids()
    else:
        # Validate sample_ids
        available_ids = dataset.get_sample_ids()
        invalid_ids = [sid for sid in sample_ids if sid not in available_ids]
        if invalid_ids:
            raise ValueError(f"Invalid sample IDs: {invalid_ids}")

    # Get feature names if not provided
    if feature_names is None:
        feature_names = dataset.get_scalar_names(ids=sample_ids)
        if not feature_names:
            raise ValueError("No scalar features found in dataset")

    # Validate feature names
    all_scalar_names = dataset.get_scalar_names(ids=sample_ids)
    all_field_names = dataset.get_field_names(ids=sample_ids)
    all_feature_names = all_scalar_names + all_field_names

    invalid_features = [f for f in feature_names if f not in all_feature_names]
    if invalid_features:
        raise ValueError(
            f"Invalid feature names: {invalid_features}. Available features: {all_feature_names}"
        )

    # Collect feature data
    feature_data = {}
    for feature_name in feature_names:
        values = []
        valid_ids = []

        for sid in sample_ids:
            sample = dataset[sid]

            # Try to get as scalar first
            if feature_name in sample.get_scalar_names():
                value = sample.get_scalar(feature_name)
                if value is not None:
                    # Handle both scalar and array scalars
                    if isinstance(value, np.ndarray):
                        # For array scalars, take the mean
                        values.append(float(np.mean(value)))
                    else:
                        values.append(float(value))
                    valid_ids.append(sid)
            # Try to get as field
            elif "/" in feature_name:
                # Parse field identifier: base_name/zone_name/location/field_name
                parts = feature_name.split("/")
                if len(parts) >= 4:
                    base_name, zone_name, location, field_name = (
                        parts[0],
                        parts[1],
                        parts[2],
                        "/".join(parts[3:]),
                    )
                    try:
                        field = sample.get_field(
                            field_name,
                            location=location,
                            zone_name=zone_name,
                            base_name=base_name,
                        )
                        if field is not None:
                            # For fields, take the mean value
                            values.append(float(np.mean(field)))
                            valid_ids.append(sid)
                    except Exception:
                        # Field not found in this sample, skip
                        pass

        if values:
            feature_data[feature_name] = (np.array(valid_ids), np.array(values))
        else:
            logger.warning(f"No valid data found for feature '{feature_name}'")

    if not feature_data:
        raise ValueError("No valid feature data found")

    # Create plots
    n_features = len(feature_data)
    n_plots = (n_features + max_features_per_plot - 1) // max_features_per_plot
    figures = []

    for plot_idx in range(n_plots):
        start_idx = plot_idx * max_features_per_plot
        end_idx = min((plot_idx + 1) * max_features_per_plot, n_features)
        plot_features = list(feature_data.keys())[start_idx:end_idx]
        n_subplot_features = len(plot_features)

        # Calculate subplot layout
        n_cols = min(2, n_subplot_features)
        n_rows = (n_subplot_features + n_cols - 1) // n_cols

        # Set figure size
        if figsize is None:
            fig_width = 6 * n_cols
            fig_height = 4 * n_rows
            fig_size = (fig_width, fig_height)
        else:
            fig_size = figsize

        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, squeeze=False)

        # Set main title
        if title:
            if n_plots > 1:
                fig.suptitle(f"{title} (Part {plot_idx + 1}/{n_plots})", fontsize=14)
            else:
                fig.suptitle(title, fontsize=14)

        # Plot each feature
        for idx, feature_name in enumerate(plot_features):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            ids, values = feature_data[feature_name]
            ax.scatter(ids, values, **kwargs)
            ax.set_xlabel("Sample ID", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.set_title(feature_name, fontsize=11)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_subplot_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()
        figures.append(fig)

    return figures[0] if len(figures) == 1 else figures


def pairplot(
    dataset: Union[Dataset, list[Sample]],
    scalar_names: Optional[list[str]] = None,
    sample_ids: Optional[list[int]] = None,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    diag_kind: str = "hist",
    corner: bool = False,
    **kwargs,
) -> Figure:
    """Create a pairplot matrix showing relationships between scalar features.

    This function creates a grid of plots where each off-diagonal subplot shows
    a scatter plot of two features, and diagonal subplots show the distribution
    of individual features. Useful for detecting correlations and understanding
    multivariate relationships.

    Args:
        dataset (Union[Dataset, list[Sample]]): The dataset or list of samples to visualize.
        scalar_names (list[str], optional): List of scalar feature names to include in the pairplot.
            If None, uses all scalar features. Defaults to None.
        sample_ids (list[int], optional): List of sample IDs to include. If None, uses all samples. Defaults to None.
        figsize (tuple[float, float], optional): Figure size (width, height) in inches.
            If None, automatically calculated based on number of features. Defaults to None.
        title (str, optional): Main title for the plot. Defaults to None.
        diag_kind (str, optional): Type of plot for diagonal subplots. Options are:
            - "hist": Histogram
            - "kde": Kernel Density Estimation
            Defaults to "hist".
        corner (bool, optional): If True, only shows the lower triangle of the pairplot.
            This reduces redundancy since scatter(x, y) and scatter(y, x) show the same information.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to matplotlib's scatter function.

    Returns:
        Figure: The created figure containing the pairplot.

    Raises:
        TypeError: If dataset is not a Dataset or list[Sample].
        ValueError: If dataset is empty, no scalar features found, or invalid parameters.

    Example:
        >>> from plaid import Dataset
        >>> from plaid.utils.viz import pairplot
        >>> dataset = Dataset("path/to/dataset")
        >>> # Create pairplot for all scalars
        >>> pairplot(dataset)
        >>> # Create pairplot for specific scalars
        >>> pairplot(dataset, scalar_names=["temperature", "pressure", "density"])
        >>> # Create corner pairplot with KDE on diagonal
        >>> pairplot(dataset, diag_kind="kde", corner=True)
    """
    # Lazy import to avoid circular dependency
    from plaid.containers.dataset import Dataset

    # Input validation
    if isinstance(dataset, list):
        # Convert list of samples to Dataset for easier handling
        temp_dataset = Dataset()
        temp_dataset.add_samples(dataset)
        dataset = temp_dataset
    elif not isinstance(dataset, Dataset):
        raise TypeError(
            f"dataset must be a Dataset or list[Sample], got {type(dataset)}"
        )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    if diag_kind not in ["hist", "kde"]:
        raise ValueError(f"diag_kind must be 'hist' or 'kde', got '{diag_kind}'")

    # Get sample IDs
    if sample_ids is None:
        sample_ids = dataset.get_sample_ids()
    else:
        # Validate sample_ids
        available_ids = dataset.get_sample_ids()
        invalid_ids = [sid for sid in sample_ids if sid not in available_ids]
        if invalid_ids:
            raise ValueError(f"Invalid sample IDs: {invalid_ids}")

    # Get scalar names if not provided
    if scalar_names is None:
        scalar_names = dataset.get_scalar_names(ids=sample_ids)
        if not scalar_names:
            raise ValueError("No scalar features found in dataset")
    else:
        # Validate scalar names
        all_scalar_names = dataset.get_scalar_names(ids=sample_ids)
        invalid_scalars = [s for s in scalar_names if s not in all_scalar_names]
        if invalid_scalars:
            raise ValueError(
                f"Invalid scalar names: {invalid_scalars}. Available scalars: {all_scalar_names}"
            )

    # Collect scalar data
    scalar_data = {}
    for scalar_name in scalar_names:
        values = []
        for sid in sample_ids:
            sample = dataset[sid]
            value = sample.get_scalar(scalar_name)
            if value is not None:
                # Handle both scalar and array scalars
                if isinstance(value, np.ndarray):
                    # For array scalars, take the mean
                    values.append(float(np.mean(value)))
                else:
                    values.append(float(value))
            else:
                # Use NaN for missing values
                values.append(np.nan)

        scalar_data[scalar_name] = np.array(values)

    if not scalar_data:
        raise ValueError("No valid scalar data found")

    # Remove samples with any NaN values
    data_matrix = np.column_stack([scalar_data[name] for name in scalar_names])
    valid_mask = ~np.any(np.isnan(data_matrix), axis=1)
    data_matrix = data_matrix[valid_mask]

    if len(data_matrix) == 0:
        raise ValueError("No samples with complete scalar data found")

    n_features = len(scalar_names)

    # Set figure size
    if figsize is None:
        fig_size = (3 * n_features, 3 * n_features)
    else:
        fig_size = figsize

    # Create figure and axes
    fig, axes = plt.subplots(n_features, n_features, figsize=fig_size)

    # Handle single feature case
    if n_features == 1:
        axes = np.array([[axes]])

    # Set main title
    if title:
        fig.suptitle(title, fontsize=16, y=0.995)

    # Create pairplot
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]

            if corner and j > i:
                # Hide upper triangle if corner=True
                ax.axis("off")
                continue

            if i == j:
                # Diagonal: plot distribution
                data = data_matrix[:, i]

                if diag_kind == "hist":
                    ax.hist(data, bins=20, edgecolor="black", alpha=0.7)
                elif diag_kind == "kde":
                    # Simple KDE using histogram as approximation
                    from scipy import stats

                    try:
                        kde = stats.gaussian_kde(data)
                        x_range = np.linspace(data.min(), data.max(), 100)
                        ax.plot(x_range, kde(x_range), linewidth=2)
                        ax.fill_between(x_range, kde(x_range), alpha=0.3)
                    except Exception:
                        # Fallback to histogram if KDE fails
                        ax.hist(data, bins=20, edgecolor="black", alpha=0.7)
                        logger.warning(
                            f"KDE failed for {scalar_names[i]}, using histogram instead"
                        )

                ax.set_ylabel("Frequency" if diag_kind == "hist" else "Density")
            else:
                # Off-diagonal: scatter plot
                x_data = data_matrix[:, j]
                y_data = data_matrix[:, i]
                # Set default alpha if not provided in kwargs
                scatter_kwargs = {"alpha": 0.5}
                scatter_kwargs.update(kwargs)
                ax.scatter(x_data, y_data, **scatter_kwargs)

            # Set labels
            if i == n_features - 1:
                ax.set_xlabel(scalar_names[j], fontsize=10)
            else:
                ax.set_xticklabels([])

            if j == 0 and not (corner and i == 0):
                ax.set_ylabel(scalar_names[i], fontsize=10)
            elif i != j:
                ax.set_yticklabels([])

            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def kdeplot(
    dataset: Union[Dataset, list[Sample]],
    feature_names: Optional[list[str]] = None,
    sample_ids: Optional[list[int]] = None,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    fill: bool = True,
    bw_method: Optional[Union[str, float]] = None,
    **kwargs,
) -> Figure:
    """Create kernel density estimation plots for feature distributions.

    This function visualizes the probability density of features using KDE,
    which provides a smooth continuous estimate of the distribution. Multiple
    features can be overlaid on the same plot for comparison.

    Args:
        dataset (Union[Dataset, list[Sample]]): The dataset or list of samples to visualize.
        feature_names (list[str], optional): List of feature names to plot. If None, plots all scalar features.
            For field features, use the format "base_name/zone_name/location/field_name". Defaults to None.
        sample_ids (list[int], optional): List of sample IDs to include. If None, uses all samples. Defaults to None.
        figsize (tuple[float, float], optional): Figure size (width, height) in inches.
            If None, defaults to (10, 6). Defaults to None.
        title (str, optional): Main title for the plot. Defaults to None.
        fill (bool, optional): If True, fills the area under the KDE curve. Defaults to True.
        bw_method (str or float, optional): Bandwidth selection method for KDE.
            Can be 'scott', 'silverman', or a scalar. If None, uses 'scott'. Defaults to None.
        **kwargs: Additional keyword arguments passed to matplotlib's plot function.

    Returns:
        Figure: The created figure containing the KDE plots.

    Raises:
        TypeError: If dataset is not a Dataset or list[Sample].
        ValueError: If dataset is empty or no valid features found.

    Example:
        >>> from plaid import Dataset
        >>> from plaid.utils.viz import kdeplot
        >>> dataset = Dataset("path/to/dataset")
        >>> # Create KDE plot for all scalars
        >>> kdeplot(dataset)
        >>> # Compare distributions of specific features
        >>> kdeplot(dataset, feature_names=["temperature", "pressure"])
        >>> # Customize appearance
        >>> kdeplot(dataset, fill=False, bw_method='silverman', linewidth=2)
    """
    # Lazy import to avoid circular dependency
    from plaid.containers.dataset import Dataset

    # Input validation
    if isinstance(dataset, list):
        # Convert list of samples to Dataset for easier handling
        temp_dataset = Dataset()
        temp_dataset.add_samples(dataset)
        dataset = temp_dataset
    elif not isinstance(dataset, Dataset):
        raise TypeError(
            f"dataset must be a Dataset or list[Sample], got {type(dataset)}"
        )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    # Get sample IDs
    if sample_ids is None:
        sample_ids = dataset.get_sample_ids()
    else:
        # Validate sample_ids
        available_ids = dataset.get_sample_ids()
        invalid_ids = [sid for sid in sample_ids if sid not in available_ids]
        if invalid_ids:
            raise ValueError(f"Invalid sample IDs: {invalid_ids}")

    # Get feature names if not provided
    if feature_names is None:
        feature_names = dataset.get_scalar_names(ids=sample_ids)
        if not feature_names:
            raise ValueError("No scalar features found in dataset")

    # Validate feature names
    all_scalar_names = dataset.get_scalar_names(ids=sample_ids)
    all_field_names = dataset.get_field_names(ids=sample_ids)
    all_feature_names = all_scalar_names + all_field_names

    invalid_features = [f for f in feature_names if f not in all_feature_names]
    if invalid_features:
        raise ValueError(
            f"Invalid feature names: {invalid_features}. Available features: {all_feature_names}"
        )

    # Collect feature data
    feature_data = {}
    for feature_name in feature_names:
        values = []

        for sid in sample_ids:
            sample = dataset[sid]

            # Try to get as scalar first
            if feature_name in sample.get_scalar_names():
                value = sample.get_scalar(feature_name)
                if value is not None:
                    # Handle both scalar and array scalars
                    if isinstance(value, np.ndarray):
                        # For array scalars, flatten all values
                        values.extend(value.flatten().tolist())
                    else:
                        values.append(float(value))
            # Try to get as field
            elif "/" in feature_name:
                # Parse field identifier: base_name/zone_name/location/field_name
                parts = feature_name.split("/")
                if len(parts) >= 4:
                    base_name, zone_name, location, field_name = (
                        parts[0],
                        parts[1],
                        parts[2],
                        "/".join(parts[3:]),
                    )
                    try:
                        field = sample.get_field(
                            field_name,
                            location=location,
                            zone_name=zone_name,
                            base_name=base_name,
                        )
                        if field is not None:
                            # For fields, flatten all values
                            values.extend(field.flatten().tolist())
                    except Exception:
                        # Field not found in this sample, skip
                        pass

        if values:
            feature_data[feature_name] = np.array(values)
        else:
            logger.warning(f"No valid data found for feature '{feature_name}'")

    if not feature_data:
        raise ValueError("No valid feature data found")

    # Set figure size
    if figsize is None:
        figsize = (10, 6)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set main title
    if title:
        fig.suptitle(title, fontsize=14)

    # Import scipy for KDE
    # Plot KDE for each feature
    from matplotlib import cm
    from scipy import stats

    colors = cm.get_cmap("tab10")(np.linspace(0, 1, len(feature_data)))

    for idx, (feature_name, data) in enumerate(feature_data.items()):
        # Remove NaN values
        data = data[~np.isnan(data)]

        if len(data) < 2:
            logger.warning(
                f"Skipping '{feature_name}': need at least 2 data points for KDE"
            )
            continue

        try:
            # Compute KDE
            kde = stats.gaussian_kde(data, bw_method=bw_method)

            # Create evaluation points
            x_min, x_max = data.min(), data.max()
            x_range = x_max - x_min
            x_eval = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 200)

            # Evaluate KDE
            density = kde(x_eval)

            # Plot
            color = colors[idx]
            ax.plot(x_eval, density, label=feature_name, color=color, **kwargs)

            if fill:
                ax.fill_between(x_eval, density, alpha=0.3, color=color)

        except Exception as e:
            logger.warning(f"KDE failed for '{feature_name}': {e}")
            continue

    # Configure plot
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    return fig
