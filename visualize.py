#!/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np


def main(input_file: str, output_file: str | None, dpi: int = 2400) -> None:
    image = np.log(np.loadtxt(input_file, delimiter=",", dtype=np.uint16))

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.axis("off")
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(image, interpolation="none", cmap="BuGn", aspect="auto")

    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches=0, pad_inches=0)
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

    parser.add_argument("--dpi", help="DPI", type=int, default=2400)

    args = parser.parse_args()

    main(args.input, args.output, args.dpi)
