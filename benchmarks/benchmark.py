#!/usr/bin/env python3
import os
import logging
import argparse
import subprocess
import time
from threading import Thread, Lock
from telegram_logging import TelegramFormatter, TelegramHandler
from tqdm.contrib.telegram import tqdm

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="../.env")

# Set up telegram logging
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("Telegram token and/or chat ID not found in environment variables")
    exit(1)

MAX_POTENTIAL_THREAD_COUNT = 32
NUM_CORES_AVAILABLE = os.cpu_count()
NUM_WARMUP_RUNS = 5
NUM_BENCHMARK_RUNS = 10
BENCHMARK_FILE = "benchmark.csv"


def get_logger(log_level: int = logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    formatter = TelegramFormatter(
        fmt="%(levelname)s %(message)s",
        datefmt=LOGGER_DATE_FORMAT,
        use_emoji=True
    )

    handler = TelegramHandler(
        bot_token=TELEGRAM_TOKEN,
        chat_id=TELEGRAM_CHAT_ID)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        fmt='[%(levelname)s - %(asctime)s] %(message)s',
        datefmt=LOGGER_DATE_FORMAT
    ))
    logger.addHandler(console_handler)

    return logger


logger = get_logger()

# Initialize core availability, all cores are free initially
cores = [False] * NUM_CORES_AVAILABLE
lock = Lock()  # Lock for thread-safe access to core list


def get_jobs(skip_baseline: bool, only_baseline: bool):
    """Get all possible jobs to run."""

    # get all binaries in the build directory
    binaries = []
    for root, dirs, files in os.walk("../build/bin/basic"):
        for file in files:

            # Skip the baseline binaries
            if skip_baseline and file.startswith("baseline"):
                continue

            # Only run baseline binaries
            if only_baseline and not file.startswith("baseline"):
                continue

            binaries.append(os.path.join(root, file))

    # get all divisors of the number of cores
    max_threads = min(MAX_POTENTIAL_THREAD_COUNT, NUM_CORES_AVAILABLE)
    thread_counts = [i for i in range(1, max_threads + 1) if max_threads % i == 0]

    # get all possible combinations of binaries and thread counts
    jobs = [(binary, thread_count) for binary in binaries for thread_count in thread_counts]

    # Sort jobs by thread_count in descending order
    jobs.sort(key=lambda x: x[1], reverse=True)

    return jobs


def run_job(
        binary,
        thread_count,
        core_indices,
        num_warmup_runs=NUM_WARMUP_RUNS,
        num_benchmark_runs=NUM_BENCHMARK_RUNS
):
    """Run a job with the specified binary and thread count on the given cores."""
    # env = {"OMP_NUM_THREADS": str(thread_count)}
    core_mask = ",".join(map(str, core_indices))

    # Parse filename from path
    filename = os.path.basename(binary)

    logger.debug("starting %s on cores %d-%d",
                 filename, core_indices[0], core_indices[-1])

    shell_script = f"""
    # set oneAPI environment
    source /opt/intel/oneapi/setvars.sh --include-intel-llvm

    # warm up
    for _ in $(seq 1 {num_warmup_runs}); do
        OMP_NUM_THREADS={thread_count} taskset -c {core_mask} {binary} result > /dev/null
    done

    # run the benchmark
    for _ in $(seq 1 {num_benchmark_runs}); do
        output=$(OMP_NUM_THREADS={thread_count} taskset -c {core_mask} {binary} result 2>/dev/null)
        # append output to csv
        echo "{filename},{thread_count},$output" >> {BENCHMARK_FILE}
    done
    """
    result = subprocess.run(
        shell_script,
        shell=True, executable='/bin/bash',
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        logger.error(
            "Job failed with return code %d: %s",
            result.returncode,
            result.stderr.decode())

    # Release the cores after the job completes
    with lock:
        for core in core_indices:
            cores[core] = False
    logger.debug("finished %s and released cores %d-%d",
                 filename, core_indices[0], core_indices[-1])


def allocate_cores(thread_count):
    """Allocate cores for a job if enough are available."""
    with lock:
        free_indices = [i for i, in_use in enumerate(cores) if not in_use]
        if len(free_indices) >= thread_count:
            allocated = free_indices[:thread_count]
            for core in allocated:
                cores[core] = True
            return allocated
    return None


def scheduler(args):
    """Main scheduler loop."""
    threads = []

    jobs = get_jobs(args.skip_baseline, args.only_baseline)
    logger.info("Running %d jobs with %d warmup and %d benchmark runs",
                len(jobs), args.warmup, args.benchmark)

    progress_bar = tqdm(
        token=TELEGRAM_TOKEN,
        chat_id=TELEGRAM_CHAT_ID,
        mininterval=5,
        total=len(jobs),
        desc="Progress",
        unit="job",
        disable=True if args.no_telegram else False)

    while jobs:
        binary, thread_count = jobs[0]  # Peek at the first job

        # Try to allocate cores
        allocated_cores = allocate_cores(thread_count)
        if allocated_cores:
            jobs.pop(0)  # Remove the job from the queue
            # Start the job in a separate thread
            thread = Thread(
                target=run_job,
                args=(
                    binary,
                    thread_count,
                    allocated_cores,
                    args.warmup,
                    args.benchmark
                ),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            progress_bar.update(1)
        else:
            # Wait and check again if cores are available
            time.sleep(1)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    progress_bar.close()

    logger.info("All jobs completed")


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmarking script for the parallel-mandelbrot project.")

    # output file
    parser.add_argument(
        "-o", "--output",
        type=str, default=BENCHMARK_FILE,
        help="File to write benchmark results to")

    # control number of runs
    parser.add_argument(
        "--warmup",
        type=int, default=NUM_WARMUP_RUNS,
        help="Number of warmup runs")
    parser.add_argument(
        "--benchmark",
        type=int, default=NUM_BENCHMARK_RUNS,
        help="Number of benchmark runs")

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging")

    # disable telegram logging
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable telegram logging"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline binaries"
    )
    group.add_argument(
        "--only-baseline",
        action="store_true",
        help="Only run baseline binaries"
    )

    yn_group = parser.add_mutually_exclusive_group()
    yn_group.add_argument(
        "-y",
        action="store_true",
        help="Automatically overwrite the benchmark file if it exists"
    )
    yn_group.add_argument(
        "-n",
        action="store_true",
        help="Automatically exit if the benchmark file exists"
    )

    args = parser.parse_args()

    BENCHMARK_FILE = args.output  # Set the benchmark file

    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.no_telegram:
        logger.removeHandler(logger.handlers[0])

    logger.info("Starting benchmarks")
    logger.info("CPU cores available: %d", NUM_CORES_AVAILABLE)

    if args.skip_baseline:
        logger.info("Skipping baseline binaries. Drop --skip-baseline flag to run all binaries.")
    elif args.only_baseline:
        logger.info("Only running baseline binaries. Drop --only-baseline flag to run all binaries.")

    # Check if the benchmark file already exists
    if os.path.exists(BENCHMARK_FILE):
        if args.y:
            logger.info(f"Overwriting file '{BENCHMARK_FILE}'")
            os.unlink(BENCHMARK_FILE)
        elif args.n:
            logger.warning(f"File '{BENCHMARK_FILE}' already exists. Exiting")
            exit(0)
        else:
            # ask the user if they want to overwrite the file
            response = input(f"File '{BENCHMARK_FILE}' already exists. Overwrite? (y/n): ")
            if response.lower() != "y":
                print("Exiting")
                exit(0)

            os.unlink(BENCHMARK_FILE)

    # Create the benchmark file
    with open(BENCHMARK_FILE, "w") as f:
        f.write("file,threads,time\n")

    scheduler(args)
