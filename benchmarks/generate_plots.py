#!/usr/bin/env python
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

SHOW_PLOTS = False


# Read performance metrics from CSV
def read_metrics(filename):
    return pd.read_csv(filename)


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
