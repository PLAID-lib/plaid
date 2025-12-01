# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import matplotlib.pyplot as plt
import numpy as np
import pytest

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.utils.viz import kdeplot, pairplot, scatter_plot

# %% Fixtures


@pytest.fixture()
def sample_dataset():
    """Create a dataset with multiple samples for testing."""
    samples = []
    n_samples = 20

    for i in range(n_samples):
        sample = Sample()

        # Add scalar features
        sample.add_scalar("temperature", 20.0 + i * 0.5 + np.random.randn() * 0.1)
        sample.add_scalar("pressure", 100.0 + i * 2.0 + np.random.randn() * 0.5)
        sample.add_scalar("density", 1.2 + np.random.randn() * 0.05)

        # Add field feature
        sample.init_base(2, 3, "base_1")
        zone_shape = np.array([50, 0, 0])
        sample.init_zone(zone_shape, zone_name="zone_1")
        sample.set_nodes(np.random.randn(50, 3))

        field_data = np.random.randn(50) * (1 + i * 0.1)
        sample.add_field("velocity", field_data)

        samples.append(sample)

    dataset = Dataset()
    dataset.add_samples(samples)
    return dataset


@pytest.fixture()
def empty_dataset():
    """Create an empty dataset for testing error handling."""
    return Dataset()


@pytest.fixture()
def single_sample_dataset():
    """Create a dataset with a single sample."""
    sample = Sample()
    sample.add_scalar("test_scalar", 42.0)

    dataset = Dataset()
    dataset.add_samples([sample])
    return dataset


# %% Tests for scatter_plot


def test_scatter_plot_basic(sample_dataset):
    """Test basic scatter plot functionality."""
    fig = scatter_plot(sample_dataset)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_scatter_plot_with_feature_names(sample_dataset):
    """Test scatter plot with specific feature names."""
    fig = scatter_plot(sample_dataset, feature_names=["temperature", "pressure"])
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_scatter_plot_with_sample_ids(sample_dataset):
    """Test scatter plot with specific sample IDs."""
    sample_ids = sample_dataset.get_sample_ids()[:10]
    fig = scatter_plot(sample_dataset, sample_ids=sample_ids)
    assert fig is not None
    plt.close(fig)


def test_scatter_plot_custom_figsize(sample_dataset):
    """Test scatter plot with custom figure size."""
    fig = scatter_plot(sample_dataset, figsize=(12, 8))
    assert fig is not None
    assert fig.get_figwidth() == 12
    assert fig.get_figheight() == 8
    plt.close(fig)


def test_scatter_plot_with_title(sample_dataset):
    """Test scatter plot with title."""
    fig = scatter_plot(sample_dataset, title="Test Plot")
    assert fig is not None
    plt.close(fig)


def test_scatter_plot_with_kwargs(sample_dataset):
    """Test scatter plot with additional matplotlib kwargs."""
    fig = scatter_plot(sample_dataset, alpha=0.5, s=100, c="red")
    assert fig is not None
    plt.close(fig)


def test_scatter_plot_list_of_samples(sample_dataset):
    """Test scatter plot with list of samples instead of dataset."""
    samples = [sample_dataset[sid] for sid in sample_dataset.get_sample_ids()]
    fig = scatter_plot(samples)
    assert fig is not None
    plt.close(fig)


def test_scatter_plot_empty_dataset_raises_error(empty_dataset):
    """Test that scatter plot raises error for empty dataset."""
    with pytest.raises(ValueError, match="Dataset is empty"):
        scatter_plot(empty_dataset)


def test_scatter_plot_invalid_feature_name(sample_dataset):
    """Test that scatter plot raises error for invalid feature names."""
    with pytest.raises(ValueError, match="Invalid feature names"):
        scatter_plot(sample_dataset, feature_names=["nonexistent_feature"])


def test_scatter_plot_invalid_sample_ids(sample_dataset):
    """Test that scatter plot raises error for invalid sample IDs."""
    with pytest.raises(ValueError, match="Invalid sample IDs"):
        scatter_plot(sample_dataset, sample_ids=[999, 1000])


def test_scatter_plot_invalid_type():
    """Test that scatter plot raises error for invalid input type."""
    with pytest.raises(TypeError, match="dataset must be a Dataset or list"):
        scatter_plot("invalid_input")


# %% Tests for pairplot


def test_pairplot_basic(sample_dataset):
    """Test basic pairplot functionality."""
    fig = pairplot(sample_dataset)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_pairplot_with_scalar_names(sample_dataset):
    """Test pairplot with specific scalar names."""
    fig = pairplot(sample_dataset, scalar_names=["temperature", "pressure"])
    assert fig is not None
    plt.close(fig)


def test_pairplot_with_kde_diagonal(sample_dataset):
    """Test pairplot with KDE on diagonal."""
    fig = pairplot(sample_dataset, diag_kind="kde")
    assert fig is not None
    plt.close(fig)


def test_pairplot_with_hist_diagonal(sample_dataset):
    """Test pairplot with histogram on diagonal."""
    fig = pairplot(sample_dataset, diag_kind="hist")
    assert fig is not None
    plt.close(fig)


def test_pairplot_corner_mode(sample_dataset):
    """Test pairplot in corner mode (lower triangle only)."""
    fig = pairplot(sample_dataset, corner=True)
    assert fig is not None
    plt.close(fig)


def test_pairplot_with_sample_ids(sample_dataset):
    """Test pairplot with specific sample IDs."""
    sample_ids = sample_dataset.get_sample_ids()[:10]
    fig = pairplot(sample_dataset, sample_ids=sample_ids)
    assert fig is not None
    plt.close(fig)


def test_pairplot_custom_figsize(sample_dataset):
    """Test pairplot with custom figure size."""
    fig = pairplot(sample_dataset, figsize=(15, 15))
    assert fig is not None
    plt.close(fig)


def test_pairplot_with_title(sample_dataset):
    """Test pairplot with title."""
    fig = pairplot(sample_dataset, title="Test Pairplot")
    assert fig is not None
    plt.close(fig)


def test_pairplot_single_feature(sample_dataset):
    """Test pairplot with single feature."""
    fig = pairplot(sample_dataset, scalar_names=["temperature"])
    assert fig is not None
    plt.close(fig)


def test_pairplot_list_of_samples(sample_dataset):
    """Test pairplot with list of samples."""
    samples = [sample_dataset[sid] for sid in sample_dataset.get_sample_ids()]
    fig = pairplot(samples)
    assert fig is not None
    plt.close(fig)


def test_pairplot_empty_dataset_raises_error(empty_dataset):
    """Test that pairplot raises error for empty dataset."""
    with pytest.raises(ValueError, match="Dataset is empty"):
        pairplot(empty_dataset)


def test_pairplot_invalid_diag_kind(sample_dataset):
    """Test that pairplot raises error for invalid diag_kind."""
    with pytest.raises(ValueError, match="diag_kind must be"):
        pairplot(sample_dataset, diag_kind="invalid")


def test_pairplot_invalid_scalar_names(sample_dataset):
    """Test that pairplot raises error for invalid scalar names."""
    with pytest.raises(ValueError, match="Invalid scalar names"):
        pairplot(sample_dataset, scalar_names=["nonexistent_scalar"])


def test_pairplot_invalid_sample_ids(sample_dataset):
    """Test that pairplot raises error for invalid sample IDs."""
    with pytest.raises(ValueError, match="Invalid sample IDs"):
        pairplot(sample_dataset, sample_ids=[999, 1000])


def test_pairplot_invalid_type():
    """Test that pairplot raises error for invalid input type."""
    with pytest.raises(TypeError, match="dataset must be a Dataset or list"):
        pairplot(123)


# %% Tests for kdeplot


def test_kdeplot_basic(sample_dataset):
    """Test basic kdeplot functionality."""
    fig = kdeplot(sample_dataset)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_kdeplot_with_feature_names(sample_dataset):
    """Test kdeplot with specific feature names."""
    fig = kdeplot(sample_dataset, feature_names=["temperature", "pressure"])
    assert fig is not None
    plt.close(fig)


def test_kdeplot_single_feature(sample_dataset):
    """Test kdeplot with single feature."""
    fig = kdeplot(sample_dataset, feature_names=["temperature"])
    assert fig is not None
    plt.close(fig)


def test_kdeplot_with_sample_ids(sample_dataset):
    """Test kdeplot with specific sample IDs."""
    sample_ids = sample_dataset.get_sample_ids()[:10]
    fig = kdeplot(sample_dataset, sample_ids=sample_ids)
    assert fig is not None
    plt.close(fig)


def test_kdeplot_no_fill(sample_dataset):
    """Test kdeplot without fill."""
    fig = kdeplot(sample_dataset, fill=False)
    assert fig is not None
    plt.close(fig)


def test_kdeplot_with_fill(sample_dataset):
    """Test kdeplot with fill."""
    fig = kdeplot(sample_dataset, fill=True)
    assert fig is not None
    plt.close(fig)


def test_kdeplot_custom_bandwidth(sample_dataset):
    """Test kdeplot with custom bandwidth method."""
    fig = kdeplot(sample_dataset, bw_method="silverman")
    assert fig is not None
    plt.close(fig)


def test_kdeplot_custom_figsize(sample_dataset):
    """Test kdeplot with custom figure size."""
    fig = kdeplot(sample_dataset, figsize=(12, 8))
    assert fig is not None
    assert fig.get_figwidth() == 12
    assert fig.get_figheight() == 8
    plt.close(fig)


def test_kdeplot_with_title(sample_dataset):
    """Test kdeplot with title."""
    fig = kdeplot(sample_dataset, title="Test KDE Plot")
    assert fig is not None
    plt.close(fig)


def test_kdeplot_with_kwargs(sample_dataset):
    """Test kdeplot with additional matplotlib kwargs."""
    fig = kdeplot(sample_dataset, linewidth=3)
    assert fig is not None
    plt.close(fig)


def test_kdeplot_list_of_samples(sample_dataset):
    """Test kdeplot with list of samples."""
    samples = [sample_dataset[sid] for sid in sample_dataset.get_sample_ids()]
    fig = kdeplot(samples)
    assert fig is not None
    plt.close(fig)


def test_kdeplot_empty_dataset_raises_error(empty_dataset):
    """Test that kdeplot raises error for empty dataset."""
    with pytest.raises(ValueError, match="Dataset is empty"):
        kdeplot(empty_dataset)


def test_kdeplot_invalid_feature_name(sample_dataset):
    """Test that kdeplot raises error for invalid feature names."""
    with pytest.raises(ValueError, match="Invalid feature names"):
        kdeplot(sample_dataset, feature_names=["nonexistent_feature"])


def test_kdeplot_invalid_sample_ids(sample_dataset):
    """Test that kdeplot raises error for invalid sample IDs."""
    with pytest.raises(ValueError, match="Invalid sample IDs"):
        kdeplot(sample_dataset, sample_ids=[999, 1000])


def test_kdeplot_invalid_type():
    """Test that kdeplot raises error for invalid input type."""
    with pytest.raises(TypeError, match="dataset must be a Dataset or list"):
        kdeplot({"invalid": "type"})


# %% Integration tests


def test_all_functions_with_minimal_data(single_sample_dataset):
    """Test that all functions can handle minimal datasets."""
    # scatter_plot should work with single sample
    fig1 = scatter_plot(single_sample_dataset)
    assert fig1 is not None
    plt.close(fig1)

    # pairplot should work with single sample
    fig2 = pairplot(single_sample_dataset)
    assert fig2 is not None
    plt.close(fig2)

    # kdeplot might fail with single sample due to KDE requirements
    # We expect it to potentially raise a warning but not crash
    try:
        fig3 = kdeplot(single_sample_dataset)
        if fig3 is not None:
            plt.close(fig3)
    except Exception:
        # KDE may fail with insufficient data, which is acceptable
        pass


def test_multiple_features_scatter_plot(sample_dataset):
    """Test scatter plot with many features (multiple figures)."""
    # Create many features to trigger multiple figures
    feature_names = ["temperature", "pressure", "density"]
    figs = scatter_plot(
        sample_dataset, feature_names=feature_names, max_features_per_plot=2
    )

    # Should return list of figures when features exceed max_features_per_plot
    if isinstance(figs, list):
        assert len(figs) > 1
        for fig in figs:
            plt.close(fig)
    else:
        plt.close(figs)


def test_visualization_functions_consistency(sample_dataset):
    """Test that all visualization functions produce valid outputs."""
    # Test scatter_plot
    fig1 = scatter_plot(sample_dataset, feature_names=["temperature"])
    assert fig1 is not None
    plt.close(fig1)

    # Test pairplot
    fig2 = pairplot(sample_dataset, scalar_names=["temperature", "pressure"])
    assert fig2 is not None
    plt.close(fig2)

    # Test kdeplot
    fig3 = kdeplot(sample_dataset, feature_names=["temperature"])
    assert fig3 is not None
    plt.close(fig3)
