#!/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np


def main(input_file: str, output_file: str | None):
    image = np.loadtxt(input_file, delimiter=",", dtype=np.uint16)

    plt.imshow(image, interpolation="none")

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Input file, default is "result"
    parser.add_argument("input", help="Input file", type=str, default="result")

    # Optional output file, disabled by default
    parser.add_argument("-o", "--output", help="Output file", type=str)

    args = parser.parse_args()

    main(args.input, args.output)
