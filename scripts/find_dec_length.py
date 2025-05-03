import sys
import numpy as np
import scipy
import argparse
from functools import partial
from common import Cache, EnvDefault
from typing import Union
import matplotlib.pyplot as plt


def nll(mean_k, data):
    # print("testing at", mean_k)
    probs = np.clip(distribution(mean_k, data), 1e-10, 1)
    logs = np.log(probs)
    return -np.sum(logs)


def distribution(mean_k, x):
    mean_pi = 4188
    amount_pi = 0.84
    amount_k = 1 - amount_pi
    return amount_k * scipy.stats.expon.pdf(x, loc=0, scale=mean_k) + amount_pi * scipy.stats.expon.pdf(
        x, loc=0, scale=mean_pi
    )


def task():
    data = np.loadtxt(".\data\dec_lengths.txt")
    bracket = (100, 500, 5000)
    best = scipy.optimize.minimize_scalar(nll, bracket=bracket, args=(data,))
    return best


def main(args: argparse.Namespace) -> Union[int, tuple[int, Cache]]:
    if hasattr(args, "cache") and args.cache is not None:
        if isinstance(args.cache, Cache):
            cache = args.cache
        else:
            cache = Cache.from_b64(args.cache)
    else:
        cache = Cache(args.cache_file)
    decay_length = task()["x"]
    cache.average_decay_length = decay_length
    if args.no_write:
        return 0, cache
    else:
        cache.dump(args.cache_file)
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-c",
        "--cache-file",
        action=EnvDefault,
        type=str,
        envvar="VALUECACHE",
        default=r".\data\value_cache.json",
        help="File path of the value cache.",
    )
    group.add_argument(
        "--cache",
        help="A base64 representation of a UTF-8 JSON string containing the value cache data. Implies `--no-write`.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write out the updated value cache. Additionally, main will return the updated value cache instead.",
    )

    args = parser.parse_args()
    if hasattr(args, "cache") and args.cache is not None:
        args.no_write = True

    if args.no_write:
        sys.exit(main(args)[0])
    sys.exit(main(args))
