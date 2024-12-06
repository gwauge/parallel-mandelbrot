#!/usr/bin/env python
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

SHOW_PLOTS = False


# Read performance metrics from CSV
def read_metrics(filename):
    return pd.read_csv(filename)


def scatter_algos(data, output_file):
    # Create a figure for subplots
    fig, axs = plt.subplots(1, len(data["algo"].unique()), figsize=(15, 15))
    fig.tight_layout(pad=2.0)

    # Plotting each category and type combination in its respective subplot
    for i, category in enumerate(data["algo"].unique()):
        ax = axs[i]
        for type_ in data["options"].unique():

            # Filter the data based on category and type
            filtered_data = data[(data["options"] == type_) & (data["algo"] == category)]

            # Plot the scatter plot for this category and type
            ax.scatter(filtered_data['threads'], filtered_data['time'], label=f'{type_[1:]}')

            # Set labels and title
            ax.set_xlabel('Threads')
            ax.set_ylabel('Time')
            ax.set_title(f'{category.capitalize()}')
            ax.grid(True)
            ax.legend()

    plt.savefig(output_file)
    plt.close(fig)


def scatter_options(data, output_file):
    fig, axs = plt.subplots(1, len(data["options"].unique()), figsize=(15, 15))
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
            ax.set_ylabel('Time')
            ax.set_title(f'{type_}')
            ax.grid(True)
            ax.legend()

    plt.savefig(output_file)
    plt.close(fig)


def boxplots(data, output_file):
    num_threads = data["threads"].unique()
    algos = data["algo"].unique()
    options = data["options"].unique()
    fig, axs = plt.subplots(len(algos), len(options), figsize=(15, 15))
    fig.tight_layout(pad=5.0)
    for x, algo in enumerate(algos):
        for y, option in enumerate(options):
            ax = axs[x, y]
            filtered_data = []
            for num_thread in num_threads:
                filtered_data.append(data[(data["algo"] == algo) & (data["options"] == option) & (data["threads"] == num_thread)]["time"])
            ax.boxplot(filtered_data)
            ax.set_xticks([y + 1 for y in range(len(filtered_data))],
                  labels=num_threads)
            ax.set_xlabel('Number of threads')
            ax.set_ylabel('Time in ms')
            ax.set_title(f"{algo} - {option}")

    plt.savefig(output_file)
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
    scatter_algos(
        data,
        os.path.join(args.output_dir, "algos"))
    scatter_options(
        data,
        os.path.join(args.output_dir, "options"))
    boxplots(
        data,
        os.path.join(args.output_dir, "boxplot"))
