import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CVResults:
    def __init__(self, records: list):
        """
        records: list of dicts with keys 'poly_order','r','fold','error'
        """
        import pandas as pd
        self.df = pd.DataFrame(records)

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the raw DataFrame."""
        return self.df

    def print_summary(self):
        """Print per‑fold errors with scientific‑notation, plus avg and max."""
        import pandas as pd

        # pivot: rows = (poly_order, r), cols = fold
        pivot = self.df.pivot_table(
            index=['poly_order','r'],
            columns='fold',
            values='error'
        )
        # add avg and max
        pivot['avg_error'] = pivot.mean(axis=1)
        pivot['max_error'] = pivot.max(axis=1)

        # Set pandas to use scientific notation for floats
        pd.set_option('display.float_format', '{:.4e}'.format)

        print("\nPer‑fold errors with average and maximum (scientific notation):\n")
        print(pivot)

        # Reset to default float format
        pd.reset_option('display.float_format')
        
def plot_mach_nut(
    inputs: dict,
    outputs_true: dict,
    outputs_pred: dict,
    idx: int,
    grid_shape: tuple = (301, 121),
    figsize: tuple = (20, 20),
    levels: int = 200
) -> tuple:
    """
    Plot side-by-side contour plots for 'mach' and 'nut' fields of a given sample,
    and their prediction errors below with a symmetric red-blue colormap (white at zero).

    Args:
        inputs (dict): Must contain key 'nodes' -> list of flattened arrays shape (nx*ny*2,).
        outputs_true (dict): True outputs with keys 'mach', 'nut' -> lists of flattened arrays shape (nx*ny,).
        outputs_pred (dict): Predicted outputs with keys 'mach', 'nut' -> lists of flattened arrays shape (nx*ny,).
        idx (int): Index of the sample to plot.
        grid_shape (tuple): (nx, ny) dimensions to reshape nodes and fields.
        figsize (tuple): Size of the figure as (width, height).
        levels (int): Number of contour levels for plotting.

    Returns:
        fig (plt.Figure), axs (np.ndarray): Matplotlib figure and axes array.
    """
    nx, ny = grid_shape
    node_flat = inputs['nodes'][idx]
    nodes = node_flat.reshape(nx, ny, 2)

    mach_pred = outputs_pred['mach'][idx].reshape(nx, ny)
    nut_pred  = outputs_pred['nut'][idx].reshape(nx, ny)

    mach_true = outputs_true['mach'][idx].reshape(nx, ny)
    nut_true  = outputs_true['nut'][idx].reshape(nx, ny)

    # Compute vertical shift between first and last rows
    s = nodes[0, -1, 1] - nodes[0, 0, 1]

    # Create 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Row 1: Predicted fields
    cf0 = axs[0, 0].contourf(
        nodes[:, :, 0], nodes[:, :, 1], mach_pred, levels
    )
    axs[0, 0].contourf(
        nodes[:, :, 0], nodes[:, :, 1] - s, mach_pred, levels
    )
    fig.colorbar(cf0, ax=axs[0, 0], orientation='vertical', aspect=10)
    axs[0, 0].set_title(f"Sample {idx} - MACH Prediction")
    axs[0, 0].set_aspect('equal')

    cf1 = axs[0, 1].contourf(
        nodes[:, :, 0], nodes[:, :, 1], nut_pred, levels
    )
    axs[0, 1].contourf(
        nodes[:, :, 0], nodes[:, :, 1] - s, nut_pred, levels
    )
    fig.colorbar(cf1, ax=axs[0, 1], orientation='vertical', aspect=10)
    axs[0, 1].set_title(f"Sample {idx} - NUT Prediction")
    axs[0, 1].set_aspect('equal')

    # Row 2: Prediction errors with symmetric colormap around zero
    err_mach = mach_pred - mach_true
    # Determine symmetric range
    max_err_m = np.max(np.abs(err_mach))
    cf2 = axs[1, 0].contourf(
        nodes[:, :, 0], nodes[:, :, 1], err_mach, levels,
        cmap='RdBu', vmin=-max_err_m, vmax=max_err_m
    )
    axs[1, 0].contourf(
        nodes[:, :, 0], nodes[:, :, 1] - s, err_mach, levels,
        cmap='RdBu', vmin=-max_err_m, vmax=max_err_m
    )
    fig.colorbar(cf2, ax=axs[1, 0], orientation='vertical', aspect=10)
    axs[1, 0].set_title(f"Sample {idx} - MACH Error")
    axs[1, 0].set_aspect('equal')

    err_nut = nut_pred - nut_true
    max_err_n = np.max(np.abs(err_nut))
    cf3 = axs[1, 1].contourf(
        nodes[:, :, 0], nodes[:, :, 1], err_nut, levels,
        cmap='RdBu', vmin=-max_err_n, vmax=max_err_n
    )
    axs[1, 1].contourf(
        nodes[:, :, 0], nodes[:, :, 1] - s, err_nut, levels,
        cmap='RdBu', vmin=-max_err_n, vmax=max_err_n
    )
    fig.colorbar(cf3, ax=axs[1, 1], orientation='vertical', aspect=10)
    axs[1, 1].set_title(f"Sample {idx} - NUT Error")
    axs[1, 1].set_aspect('equal')

    plt.tight_layout()
    return fig, axs


def plot_scalars_pred_vs_true(
    outputs_true: dict,
    outputs_pred: dict,
    scalar_keys: list = None,
    figsize: tuple = (15, 10),
    marker: str = 'o',
    diagonal_color: str = 'r'
) -> tuple:
    """
    Plot predicted vs true scatter for each scalar with a diagonal reference line.

    Args:
        outputs_true (dict): True scalar values, keys = scalar names, values = lists of floats or scalars
        outputs_pred (dict): Predicted scalar values, same structure as outputs_true
        scalar_keys (list, optional): List of scalar keys to plot. If None, automatically
            select only those keys whose values are lists of scalars.
        figsize (tuple): Figure size (width, height).
        marker (str): Marker style for scatter.
        diagonal_color (str): Color of the y=x diagonal line.

    Returns:
        fig (plt.Figure), axs (np.ndarray): Matplotlib figure and axes array.
    """
    # Auto-select keys if not provided: only lists of scalar values
    if scalar_keys is None:
        scalar_keys = [
            key for key, vals in outputs_true.items()
            if isinstance(vals, list) and vals and all(np.isscalar(v) for v in vals)
        ]
    else:
        # Filter provided keys
        scalar_keys = [
            key for key in scalar_keys
            if key in outputs_true
            and isinstance(outputs_true[key], list)
            and outputs_true[key]
            and all(np.isscalar(v) for v in outputs_true[key])
        ]

    n = len(scalar_keys)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, key in enumerate(scalar_keys):
        row = idx // ncols
        col = idx % ncols
        ax = axs[row][col]

        y_true = np.array(outputs_true[key])
        y_pred = np.array(outputs_pred[key])

        ax.scatter(y_true, y_pred, marker=marker)

        # plot diagonal
        vmin = min(y_true.min(), y_pred.min())
        vmax = max(y_true.max(), y_pred.max())
        ax.plot([vmin, vmax], [vmin, vmax], diagonal_color)

        ax.set_title(key)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        fig.delaxes(axs.flatten()[idx])

    plt.tight_layout()
    return fig, axs

