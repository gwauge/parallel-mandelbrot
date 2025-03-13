#!/usr/bin/env python
import os
import shutil
import argparse
import re
import logging

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

SHOW_PLOTS = False

DPI = 1200

logging.basicConfig(
    # level=logging.INFO,
    format='[%(levelname)s - %(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Read performance metrics from CSV
def read_metrics(filename):
    return pd.read_csv(filename)


def reorder_file_name(file_name):
    """ Fix incorrect file names """
    match = re.match(r'^(.*)_(sp|dp)_(optimized|simd)(.*)$', file_name)
    if match:
        prefix, precision, optimization, suffix = match.groups()
        return f"{prefix}_{optimization}_{precision}{suffix}"
    return file_name


def parse_filename(filename):
    """ Parse filename into implementation, precision, optimization, and precision mode """
    match = re.match(r'^(.*)_(sp|dp)_(O[0-3])(?:-(.*))?$', filename)
    if match:
        implementation, precision, optimization, precision_mode = match.groups()
    else:
        implementation, precision, optimization, precision_mode = None, None, None, None
    return implementation, precision, optimization, precision_mode


def fix_data(data):
    # Group by 'file' and 'threads' and calculate the median of 'time'
    data = data.groupby(['file', 'threads'])['time'].agg(["mean", "median", "std"]).reset_index()

    # fix incorrect file names
    data['file'] = data['file'].apply(reorder_file_name)

    # parse filenames
    data[['implementation', 'precision', 'optimization', 'precision_mode']] = data['file'].apply(
        lambda x: pd.Series(parse_filename(x))
    )

    return data


def scatter_algos(data, output_file):
    # check if output directory exists
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    # Plotting each category and type combination in its respective subplot
    for _, category in enumerate(data["algo"].unique()):
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        fig.tight_layout(pad=2.0)
        for type_ in data["options"].unique():

            # Filter the data based on category and type
            filtered_data = data[(data["options"] == type_) & (data["algo"] == category)]

            # Plot the scatter plot for this category and type
            ax.scatter(filtered_data['threads'], filtered_data['medain'], label=f'{type_}')

            # Set labels and title
            ax.set_xlabel('Threads')
            ax.set_ylabel('Time')
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter("{x:.0f}")
            ax.set_title(f'{category.capitalize()}')
            ax.grid(True)
            ax.legend()

        plt.savefig(os.path.join(output_file, f"{category}.png"), dpi=1200)
        plt.close(fig)


def scatter_options(data, output_file):
    fig, axs = plt.subplots(1, len(data["options"].unique()), figsize=(15*len(data["options"].unique()), 15))
    fig.tight_layout(pad=2.0)
    for i, type_ in enumerate(data["options"].unique()):
        ax = axs[i]
        for category in data["algo"].unique():

            # Filter the data based on category and type
            filtered_data = data[(data["options"] == type_) & (data["algo"] == category)]

            # Plot the scatter plot for this category and type
            ax.errorbar(filtered_data['threads'], filtered_data['median'], yerr=filtered_data["std"], label=f'{category}')

            # Set labels and title
            ax.set_xlabel('Threads')
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter("{x:.0f}")
            ax.set_ylabel('Time')
            ax.set_title(f'{type_}')
            ax.grid(True)
            ax.legend()

    plt.savefig(output_file, dpi=1200)
    plt.close(fig)


def boxplots(data, output_file):
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    num_threads = data["threads"].unique()
    algos = data["algo"].unique()
    options = data["options"].unique()
    for _, algo in enumerate(algos):
        fig, axs = plt.subplots(1, len(options), figsize=(15*len(options), 15))
        fig.tight_layout(pad=5.0)
        maximum = data[data["algo"] == algo].max()["median"]
        for y, option in enumerate(options):
            ax = axs[y]
            filtered_data = []
            for num_thread in num_threads:
                filtered_data.append(data[(data["algo"] == algo) & (data["options"] == option) & (data["threads"] == num_thread)]["median"])
            ax.boxplot(filtered_data)
            ax.set_xticks(
                [y + 1 for y in range(len(filtered_data))],
                labels=num_threads)
            ax.set_ylim(bottom=0, top=maximum)
            ax.set_xlabel('Number of threads')
            ax.set_ylabel('Time in ms')
            ax.set_title(f"{algo} - {option}")

        plt.savefig(os.path.join(output_file, f"{algo}.png"), dpi=1200)
        plt.close(fig)


def precision_mode_str(precision_mode):
    return "N/A" if not precision_mode else precision_mode


def scatter_implementation_with_fixed_params(
    data: pd.DataFrame,
    output_file: str,
    implementations: list[str] | dict[str, str],
    precision: str = "dp",
    optimization: str = "O3",
    precision_mode: str | None = None
):
    fig, ax = plt.subplots(1, 1)  # use figsize=(15, 15)) for square plot

    keys = list(implementations.keys()) if isinstance(implementations, dict) else implementations
    logging.info(f"Creating plot '{output_file}' with keys: {', '.join(keys)}")

    # Filter data
    filtered_df = data[
        (data['implementation'].isin(keys)) &
        (data['precision'] == precision) &
        (data['optimization'] == optimization) &
        (data['precision_mode'].isna() if precision_mode is None
         else data['precision_mode'] == precision_mode)
    ]

    # Generate scatter plot with different colors and labels for each implementation
    for implementation in filtered_df['implementation'].unique():
        subset = filtered_df[filtered_df['implementation'] == implementation]
        label = implementations[implementation] if isinstance(implementations, dict) else implementation
        zorder = 5 + 1/subset["std"].max()
        ax.errorbar(
            subset['threads'], subset['median'],
            yerr=subset["std"], zorder=zorder,
            capsize=3, linestyle="",
            marker="_", label=label)

    ax.set_title(
        f"Precision: {precision} | Optimization: {optimization} | Precision Mode: {precision_mode_str(precision_mode)}")
    ax.set_xlabel('Threads')
    ax.set_xscale("log", base=2)
    ax.grid(axis="y", alpha=0.4)
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel('Time in ms')
    ax.legend()

    plt.savefig(output_file, dpi=DPI)
    plt.close(fig)

    logging.info(f'Created plot \'{output_file}\'')


def plot_speedup_efficiency_heatmaps_precision(
    data: pd.DataFrame,
    output_file: str,
    implementations: list[str] | dict[str, str],
    optimization: str = "O3",
    threads: int = 32,
    precision_mode: str | None = None
):
    """
    Filter the dataframe by the given parameters and compute speedup and efficiency.
    Then, plot heatmaps for both metrics with numeric annotations.
    """

    keys = list(implementations.keys()) if isinstance(implementations, dict) else implementations

    # Filter data
    df_filtered = data[
        (data['implementation'].isin(keys)) &
        (data['optimization'] == optimization) &
        (data['threads'] == threads) &
        (data['precision_mode'].isna() if precision_mode is None
         else data['precision_mode'] == precision_mode)
    ]

    pivot_mean = df_filtered.pivot(index="implementation", columns="precision", values="median")

    pivot_mean = pivot_mean[["dp", "sp"]]

    speedup = pivot_mean.rdiv(pivot_mean["dp"], axis=0)

    def plot_heatmap(data, title, path):
        n_rows = data.shape[0]
        colors = ["lightgray"] * 2
        nodes = [0.0, 1.0]
        cmap2 = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

        fig, axes = plt.subplots(n_rows, 1, figsize=(3+n_rows, 6), gridspec_kw={'hspace': 0.0})
        if n_rows == 1:
            axes = [axes]
        for i, (index, row) in enumerate(data.iterrows()):
            ax = axes[i]

            # Set color normalization to the column's own min and max
            vmin = row.min().min()
            vmax = row.max().max()
            norm = plt.Normalize(vmin, vmax)
            im = ax.imshow([row.values], aspect="auto", cmap=cmap2, vmin=vmin, vmax=vmax)
            if i == n_rows-1:
                ax.set_xticks(np.arange(data.shape[1]))
                ax.set_xticklabels(data.columns)
            else:
                ax.set_xticks([])

            ax.set_yticks([0])
            ax.set_yticklabels([implementations[index]])

            for spine in ax.spines.values():
                spine.set_visible(False)

            # Annotate each cell and set text color based on background luminance.
            for j in range(data.shape[1]):
                cell_value = row.values[j]
                # Get the RGBA value for the current cell's background
                rgba = im.cmap(norm(cell_value))
                if cell_value == vmax:
                    highlight_color = (0.0, 1.0, 0.0, 1.0)
                    ax.add_patch(Rectangle((j-0.5, -0.5), 1, 1, fill=True, facecolor=highlight_color))
                    rgba = highlight_color
                # Compute luminance using the Rec. 709 formula
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                # Choose black text for light backgrounds and white for dark backgrounds
                text_color = "black" if luminance > 0.5 else "white"
                ax.text(j, 0, f"{cell_value:.2f}", ha="center", va="center",
                        color=text_color, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(title)
        plt.savefig(path, dpi=DPI)
        plt.close()

    # Plot the heatmaps
    plot_heatmap(
        speedup,
        f"Speedup Heatmap | {optimization} | {precision_mode_str(precision_mode)} | {threads}",
        os.path.join(output_file, f"speedup_precision_{optimization}_{precision_mode}_{threads}"))


def plot_speedup_efficiency_heatmaps_optimization(
    data: pd.DataFrame,
    output_file: str,
    implementations: list[str] | dict[str, str],
    precision: str = "dp",
    threads: int = 32,
    precision_mode: str | None = None
):
    """
    Filter the dataframe by the given parameters and compute speedup and efficiency.
    Then, plot heatmaps for both metrics with numeric annotations.
    """

    keys = list(implementations.keys()) if isinstance(implementations, dict) else implementations

    # Filter data
    df_filtered = data[
        (data['implementation'].isin(keys)) &
        (data['precision'] == precision) &
        (data['threads'] == threads) &
        (data['precision_mode'].isna() if precision_mode is None
         else data['precision_mode'] == precision_mode)
    ]

    # Pivot the table: rows = unique file (configuration), columns = threads, values = mean time
    pivot_mean = df_filtered.pivot(index="implementation", columns="optimization", values="median")

    pivot_mean = pivot_mean[['O0', 'O1', 'O2', 'O3']]

    # Compute speedup: use the time for 1 thread as the baseline (each row should have a thread==1 value)
    speedup = pivot_mean.rdiv(pivot_mean["O0"], axis=0)

    def plot_heatmap(data, title, path):
        n_rows = data.shape[0]
        colors = ["lightgray"] * 2
        nodes = [0.0, 1.0]
        cmap2 = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

        fig, axes = plt.subplots(n_rows, 1, figsize=(3+n_rows, 6), gridspec_kw={'hspace': 0.0})
        if n_rows == 1:
            axes = [axes]
        for i, (index, row) in enumerate(data.iterrows()):
            ax = axes[i]

            # Set color normalization to the column's own min and max
            vmin = row.min().min()
            vmax = row.max().max()
            norm = plt.Normalize(vmin, vmax)
            im = ax.imshow([row.values], aspect="auto", cmap=cmap2, vmin=vmin, vmax=vmax)
            if i == n_rows-1:
                ax.set_xticks(np.arange(data.shape[1]))
                ax.set_xticklabels(data.columns)
            else:
                ax.set_xticks([])

            ax.set_yticks([0])
            ax.set_yticklabels([implementations[index]])

            for spine in ax.spines.values():
                spine.set_visible(False)

            # Annotate each cell and set text color based on background luminance.
            for j in range(data.shape[1]):
                cell_value = row.values[j]
                # Get the RGBA value for the current cell's background
                rgba = im.cmap(norm(cell_value))
                if cell_value == vmax:
                    highlight_color = (0.0, 1.0, 0.0, 1.0)
                    ax.add_patch(Rectangle((j-0.5, -0.5), 1, 1, fill=True, facecolor=highlight_color))
                    rgba = highlight_color
                # Compute luminance using the Rec. 709 formula
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                # Choose black text for light backgrounds and white for dark backgrounds
                text_color = "black" if luminance > 0.5 else "white"
                ax.text(j, 0, f"{cell_value:.2f}", ha="center", va="center",
                        color=text_color, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(title)
        plt.savefig(path, dpi=DPI)
        plt.close()

    # Plot the heatmaps
    plot_heatmap(
        speedup,
        f"Speedup Heatmap | {precision} | {precision_mode_str(precision_mode)} | {threads}",
        os.path.join(output_file, f"speedup_optimization_{precision}_{precision_mode}_{threads}"))


def plot_speedup_efficiency_heatmaps_precision_mode(
    data: pd.DataFrame,
    output_file: str,
    implementations: list[str] | dict[str, str],
    precision: str = "dp",
    threads: int = 32,
    optimization: str = "O3",
):
    """
    Filter the dataframe by the given parameters and compute speedup and efficiency.
    Then, plot heatmaps for both metrics with numeric annotations.
    """

    keys = list(implementations.keys()) if isinstance(implementations, dict) else implementations

    # Filter data
    df_filtered = data[
        (data['implementation'].isin(keys)) &
        (data['precision'] == precision) &
        (data['optimization'] == optimization) &
        (data['threads'] == threads)
    ]

    # Pivot the table: rows = unique file (configuration), columns = threads, values = mean time
    pivot_mean = df_filtered.pivot(index="implementation", columns="precision_mode", values="median")

    pivot_mean = pivot_mean[['STRICT', 'PRECISE', np.nan]]
    pivot_mean = pivot_mean.rename(columns={np.nan: "Default"})

    # Compute speedup: use the time for 1 thread as the baseline (each row should have a thread==1 value)
    speedup = pivot_mean.rdiv(pivot_mean["STRICT"], axis=0)

    def plot_heatmap(data, title, path):
        n_rows = data.shape[0]
        colors = ["lightgray"] * 2
        nodes = [0.0, 1.0]
        cmap2 = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

        fig, axes = plt.subplots(n_rows, 1, figsize=(3+n_rows, 6), gridspec_kw={'hspace': 0.0})
        if n_rows == 1:
            axes = [axes]
        for i, (index, row) in enumerate(data.iterrows()):
            ax = axes[i]

            # Set color normalization to the column's own min and max
            vmin = row.min().min()
            vmax = row.max().max()
            norm = plt.Normalize(vmin, vmax)
            im = ax.imshow([row.values], aspect="auto", cmap=cmap2, vmin=vmin, vmax=vmax)
            if i == n_rows-1:
                ax.set_xticks(np.arange(data.shape[1]))
                ax.set_xticklabels(data.columns)
            else:
                ax.set_xticks([])

            ax.set_yticks([0])
            ax.set_yticklabels([implementations[index]])

            for spine in ax.spines.values():
                spine.set_visible(False)

            # Annotate each cell and set text color based on background luminance.
            for j in range(data.shape[1]):
                cell_value = row.values[j]
                # Get the RGBA value for the current cell's background
                rgba = im.cmap(norm(cell_value))
                if cell_value == vmax:
                    highlight_color = (0.0, 1.0, 0.0, 1.0)
                    ax.add_patch(Rectangle((j-0.5, -0.5), 1, 1, fill=True, facecolor=highlight_color))
                    rgba = highlight_color
                # Compute luminance using the Rec. 709 formula
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                # Choose black text for light backgrounds and white for dark backgrounds
                text_color = "black" if luminance > 0.5 else "white"
                ax.text(j, 0, f"{cell_value:.2f}", ha="center", va="center",
                        color=text_color, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(title)
        plt.savefig(path, dpi=DPI)
        plt.close()

    # Plot the heatmaps
    plot_heatmap(
        speedup,
        f"Speedup Heatmap | {precision} | {optimization} | {threads}",
        os.path.join(output_file, f"speedup_precision_mode_{precision}_{optimization}_{threads}"))


def plot_speedup_efficiency_heatmaps(
    data: pd.DataFrame,
    output_file: str,
    implementations: list[str] | dict[str, str],
    precision: str = "dp",
    optimization: str = "O3",
    precision_mode: str | None = None
):
    """
    Filter the dataframe by the given parameters and compute speedup and efficiency.
    Then, plot heatmaps for both metrics with numeric annotations.
    """

    keys = list(implementations.keys()) if isinstance(implementations, dict) else implementations

    # Filter data
    df_filtered = data[
        (data['implementation'].isin(keys)) &
        (data['precision'] == precision) &
        (data['optimization'] == optimization) &
        (data['precision_mode'].isna() if precision_mode is None
         else data['precision_mode'] == precision_mode)
    ]

    baseline = data[
        (data['implementation'] == "baseline") &
        (data['precision'] == precision) &
        (data['optimization'] == optimization) &
        (data['precision_mode'].isna() if precision_mode is None
         else data['precision_mode'] == precision_mode)
    ]

    # Pivot the table: rows = unique file (configuration), columns = threads, values = mean time
    pivot_mean = df_filtered.pivot(index="file", columns="threads", values="median")

    pivot_baseline = baseline.pivot(index="file", columns="threads", values="median")

    # Make sure the pivot is sorted by threads
    pivot_mean = pivot_mean.sort_index(axis=1)

    pivot_baseline = pivot_baseline.sort_index(axis=1)

    divider = pd.DataFrame({1: pivot_baseline[1].repeat(len(pivot_mean[1])).values} | {2**i: pivot_mean[1].values for i in range(1, 6)} | {"total" : pivot_baseline[1].repeat(len(pivot_mean[1])).values})
    divider.index = pivot_mean.index

    pivot_mean["total"] = pivot_mean[32]

    # Compute speedup: use the time for 1 thread as the baseline (each row should have a thread==1 value)
    speedup = pivot_mean.rdiv(divider)

    # Compute efficiency: speedup divided by the number of threads
    efficiency = speedup.copy()
    for t in efficiency.columns:
        divider_scalar = t
        if t == "total":
            divider_scalar = 32
        efficiency[t] = efficiency[t] / divider_scalar

    def plot_heatmap(data, title, path, original_table):
        n_cols = data.shape[1]

        fig, axes = plt.subplots(1, n_cols, figsize=(3+n_cols, 6), gridspec_kw={'wspace': 0.0})
        if n_cols == 1:
            axes = [axes]
        for i, col in enumerate(data.columns):
            ax = axes[i]
            # Extract column data as a 2D array for imshow (n_rows x 1)
            col_data = data[[col]]
            # Set color normalization to the column's own min and max
            vmin = col_data.min().min()
            vmax = col_data.max().max()
            norm = plt.Normalize(vmin, vmax)
            im = ax.imshow(col_data.values, aspect="auto", cmap="plasma", vmin=vmin, vmax=vmax)
            ax.set_xticks([0])
            ax.set_xticklabels([data.columns[i]])

            # Remove x-axis ticks and set title as the column header

            # Only the first column gets the y-axis labels (configuration names)
            if i == 0:
                ax.set_yticks(np.arange(data.shape[0]))
                ax.set_yticklabels(
                    [implementations[original_table[original_table["file"] == key].iloc[0]["implementation"]]
                     for key in data.index])
            else:
                ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            # Annotate each cell with its value
            # Annotate each cell and set text color based on background luminance.
            for j in range(col_data.shape[0]):
                cell_value = col_data.iloc[j, 0]
                # Get the RGBA value for the current cell's background
                rgba = im.cmap(norm(cell_value))
                # Compute luminance using the Rec. 709 formula
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                # Choose black text for light backgrounds and white for dark backgrounds
                text_color = "black" if luminance > 0.5 else "white"
                ax.text(0, j, f"{cell_value:.2f}", ha="center", va="center",
                        color=text_color, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(title)
        plt.savefig(path, dpi=DPI)
        plt.close()

    # Plot the heatmaps
    plot_heatmap(
        speedup,
        f"Speedup Heatmap | {precision} | {optimization} | {precision_mode_str(precision_mode)}",
        os.path.join(output_file, f"speedup_{precision}_{optimization}_{precision_mode}"), data)
    plot_heatmap(
        efficiency,
        f"Efficiency Heatmap | {precision} | {optimization} | {precision_mode_str(precision_mode)}",
        os.path.join(output_file, f"efficiency_{precision}_{optimization}_{precision_mode}"), data)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # filename should be positional argument with default value
    parser.add_argument(
        "--filename",
        type=str,
        default="benchmark.csv")

    # output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots")

    # clean output directory
    parser.add_argument(
        "-c", "--clean",
        action="store_true",
        default=False,
        help="Clean the output directory before creating new plots")

    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
        help="DPI of the output plots")

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.clean:
        logging.warning(f"Flag --clean was passed. Cleaning output directory '{args.output_dir}'")
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)

    if args.dpi:
        DPI = args.dpi

    data = read_metrics(args.filename)

    # check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data[['algo', 'options']] = data['file'].str.rsplit('_', n=1, expand=True)
    # Report graphs
    data = fix_data(data)

    """
    scatter_algos(
        data,
        os.path.join(args.output_dir, "algos"))
    scatter_options(
        data,
        os.path.join(args.output_dir, "options"))
    boxplots(
        data,
        os.path.join(args.output_dir, "boxplot"))
    """

    # Plot baseline, static, dynamic
    scatter_implementation_with_fixed_params(
        data,
        os.path.join(args.output_dir, "multithreading"),
        implementations={
            'baseline': 'Baseline',
            'static_multithreading': 'Static multithreading',
            'dynamic_multithreading': 'Dynamic multithreading'
        },
        precision='dp',
        optimization='O3',
        precision_mode=None
    )

    # Plot dynamic, no_complex
    scatter_implementation_with_fixed_params(
        data,
        os.path.join(args.output_dir, "no-complex"),
        implementations={
            'dynamic_multithreading_simd': 'std::complex::abs',
            'dynamic_multithreading_simd_optimized': 'std::complex::norm',
            'no_complex': 'Custom complex'
        },
        precision='dp',
        optimization='O3',
        precision_mode=None
    )

    scatter_implementation_with_fixed_params(
        data,
        os.path.join(args.output_dir, "simd"),
        implementations={
            # 'avx2': 'Manual avx2',
            'avx2_optimized': 'Optimized manual avx2',
            # 'no_complex': 'Custom complex',
            'no_complex_simd': 'Custom complex using SIMD',
            # 'dynamic_multithreading_simd': 'Dynamic multithreading using SIMD',
            'dynamic_multithreading_simd_optimized': 'Dynamic multithreading using SIMD Optimized',
            # 'dynamic_multithreading': 'Dynamic multithreading'
        },
        precision='dp',
        optimization='O3',
        precision_mode=None
    )

    plot_speedup_efficiency_heatmaps_precision(
        data,
        args.output_dir,
        implementations={
            'baseline': "Baseline",
            'static_multithreading': "Static Multithreading",
            'dynamic_multithreading': "Dynamic Multitreading",
            'dynamic_multithreading_simd': 'Dynamic multithreading using SIMD',
            'dynamic_multithreading_simd_optimized': 'Dynamic multithreading using SIMD Optimized',
            'no_complex': 'Custom complex',
            'no_complex_simd': 'Custom complex using SIMD',
            'avx2': 'Manual avx2',
            'avx2_optimized': 'Optimized manual avx2',
        },
        optimization="O3",
        threads=4,
        precision_mode=None
    )

    plot_speedup_efficiency_heatmaps_optimization(
        data,
        args.output_dir,
        implementations={
            'baseline': "Baseline",
            'static_multithreading': "Static Multithreading",
            'dynamic_multithreading': "Dynamic Multitreading",
            'dynamic_multithreading_simd': 'Dynamic multithreading using SIMD',
            'dynamic_multithreading_simd_optimized': 'Dynamic multithreading using SIMD Optimized',
            'no_complex': 'Custom complex',
            'no_complex_simd': 'Custom complex using SIMD',
            'avx2': 'Manual avx2',
            'avx2_optimized': 'Optimized manual avx2',
        },
        precision='sp',
        threads=32,
        precision_mode=None
    )

    plot_speedup_efficiency_heatmaps_precision_mode(
        data,
        args.output_dir,
        implementations={
            'baseline': "Baseline",
            'static_multithreading': "Static Multithreading",
            'dynamic_multithreading': "Dynamic Multitreading",
            'dynamic_multithreading_simd': 'Dynamic multithreading using SIMD',
            'dynamic_multithreading_simd_optimized': 'Dynamic multithreading using SIMD Optimized',
            'no_complex': 'Custom complex',
            'no_complex_simd': 'Custom complex using SIMD',
            'avx2': 'Manual avx2',
            'avx2_optimized': 'Optimized manual avx2',
        },
        precision='sp',
        threads=1,
        optimization="O0",
    )

    for optimization in ["O0", "O1", "O2", "O3"]:
        plot_speedup_efficiency_heatmaps(
            data,
            args.output_dir,
            implementations={
                'baseline': "Baseline",
                'static_multithreading': "Static Multithreading",
                'dynamic_multithreading': "Dynamic Multitreading",
                'dynamic_multithreading_simd': 'Dynamic multithreading using SIMD',
                'dynamic_multithreading_simd_optimized': 'Dynamic multithreading using SIMD Optimized',
                'no_complex': 'Custom complex',
                'no_complex_simd': 'Custom complex using SIMD',
                'avx2': 'Manual avx2',
                'avx2_optimized': 'Optimized manual avx2',
            },
            precision='sp',
            optimization=optimization,
            precision_mode=None
        )
