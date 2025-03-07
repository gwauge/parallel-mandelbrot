#!/usr/bin/env python
import os
import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt

SHOW_PLOTS = False


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
    data = data.groupby(['file', 'threads'], as_index=False)['time'].median()

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
            ax.scatter(filtered_data['threads'], filtered_data['time'], label=f'{type_}')

            # Set labels and title
            ax.set_xlabel('Threads')
            ax.set_ylabel('Time')
            ax.set_xscale("log", base=2)
            ax.set_title(f'{category.capitalize()}')
            ax.grid(True)
            ax.legend()

        plt.savefig(os.path.join(output_file, f"{category}.png"))
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
            ax.scatter(filtered_data['threads'], filtered_data['time'], label=f'{category}')

            # Set labels and title
            ax.set_xlabel('Threads')
            ax.set_xscale("log", base=2)
            ax.set_ylabel('Time')
            ax.set_title(f'{type_}')
            ax.grid(True)
            ax.legend()

    plt.savefig(output_file)
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
        maximum = data[data["algo"] == algo].max()["time"]
        for y, option in enumerate(options):
            ax = axs[y]
            filtered_data = []
            for num_thread in num_threads:
                filtered_data.append(data[(data["algo"] == algo) & (data["options"] == option) & (data["threads"] == num_thread)]["time"])
            ax.boxplot(filtered_data)
            ax.set_xticks([y + 1 for y in range(len(filtered_data))],
                  labels=num_threads)
            ax.set_ylim(bottom=0, top=maximum)
            ax.set_xlabel('Number of threads')
            ax.set_ylabel('Time in ms')
            ax.set_title(f"{algo} - {option}")

        plt.savefig(os.path.join(output_file, f"{algo}.png"))
        plt.close(fig)


# Plot performance vs threads
def plot_performance(data, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(
        data['threads'],
        data['time'],
        marker='o',
        label='Execution Time')
    plt.title('Performance vs Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.legend()
    plt.savefig(output_file)

    if SHOW_PLOTS:
        plt.show()


# Plot speedup
def plot_speedup(data, output_file):
    baseline_time = data['time'].iloc[0]
    speedup = baseline_time / data['time']

    plt.figure(figsize=(10, 6))
    plt.plot(
        data['threads'],
        speedup, marker='o',
        color='orange',
        label='Speedup')
    plt.title('Speedup vs Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.grid()
    plt.legend()
    plt.savefig(output_file)

    if SHOW_PLOTS:
        plt.show()


# Plot efficiency
def plot_efficiency(data, output_file):
    baseline_time = data['time'].iloc[0]
    speedup = baseline_time / data['time']
    efficiency = speedup / data['threads']

    plt.figure(figsize=(10, 6))
    plt.plot(
        data['threads'],
        efficiency,
        marker='o',
        color='green',
        label='Efficiency')
    plt.title('Efficiency vs Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Efficiency')
    plt.grid()
    plt.legend()
    plt.savefig(output_file)

    if SHOW_PLOTS:
        plt.show()


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
        ax.scatter(subset['threads'], subset['time'], label=label)

    ax.set_title(f"Precision: {precision} | Optimization: {optimization} | Precision Mode: {'N/A' if not precision_mode else precision_mode}")
    ax.set_xlabel('Threads')
    ax.set_xscale("log", base=2)
    ax.set_ylabel('Time in ms')
    ax.grid(True)
    ax.legend()

    plt.savefig(output_file)
    plt.close(fig)


if __name__ == "__main__":

    # Read command line arguments

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

    args = parser.parse_args()

    data = read_metrics(args.filename)

    # check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data[['algo', 'options']] = data['file'].str.rsplit('_', n=1, expand=True)

    # Create graphs
    plot_performance(
        data,
        os.path.join(args.output_dir, "performance_vs_threads.png"))
    plot_speedup(
        data,
        os.path.join(args.output_dir, "speedup_vs_threads.png"))
    plot_efficiency(
        data,
        os.path.join(args.output_dir, "efficiency_vs_threads.png"))

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

    # Report graphs
    data = fix_data(data)

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
            'dynamic_multithreading': 'std::complex',
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
            'avx2': 'Manual avx2',
            'avx2_optimized': 'Optimized manual avx2',
            'no_complex': 'Custom complex',
            'no_complex_simd': 'Custom complex using SIMD',
            'dynamic_multithreading_simd': 'Dynamic multithreading using SIMD',
            'dynamic_multithreading': 'Dynamic multithreading'
        },
        precision='dp',
        optimization='O3',
        precision_mode=None
    )
