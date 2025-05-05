import sys
import argparse
import scipy
import numpy as np
import matplotlib.pyplot as plt

from common import Cache, EnvDefault, CONSTANTS as C, EXPERIMENTAL_CONSTANTS as E

from typing import Union


def nll(mean_k, data):
    # print("testing at", mean_k)
    probs = distribution(mean_k, data)
    logs = np.ma.log(probs)
    return -np.sum(logs)


def distribution(mean_k, x):
    mean_pi = C.PION_DECAY_LENGTH
    amount_pi = E.PERCENTAGE_PIONS_BEAM
    amount_k = E.PERCENTAGE_KAONS_BEAM
    return amount_k * scipy.stats.expon.pdf(x, loc=0, scale=mean_k) + amount_pi * scipy.stats.expon.pdf(
        x, loc=0, scale=mean_pi
    )

def find_interval(nll_function,best_mean_k,nll_min,data,direction):
    def func_that_should_be_zero(mean_k):
        return nll_function(mean_k,data)-(nll_min+0.5)
    lower=(0,best_mean_k)
    higher=(best_mean_k,best_mean_k*5)
    bracket = lower if direction=="lower" else higher
    solution = scipy.optimize.root_scalar(func_that_should_be_zero, bracket=bracket)
    return solution.root


def task():
    data = np.loadtxt(".\data\dec_lengths.txt")
    bracket = (100, 500, 5000)
    best = scipy.optimize.minimize_scalar(nll, bracket=bracket, args=(data,))
    lower=find_interval(nll,best.x,best.fun,data,direction="lower")
    higher = find_interval(nll, best.x, best.fun, data, direction="higher")
    print(best.x,lower,higher)
    return best.x, [best.x-lower,higher-best.x]

def plot():
    data = np.loadtxt("./data/dec_lengths.txt")
    plt.figure()
    plt.hist(data, bins=100)
    plt.xlabel("Decay Length [m]")
    plt.ylabel("Count")
    plt.title("Histogram of Decay Lengths")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def main(args: argparse.Namespace) -> Union[int, tuple[int, Cache]]:
    if hasattr(args, "cache") and args.cache is not None:
        if isinstance(args.cache, Cache):
            cache = args.cache
        else:
            cache = Cache.from_b64(args.cache)
    else:
        cache = Cache(args.cache_file)
    decay_length, dl_uncertainty = task()
    cache.average_decay_length = decay_length
    cache.dlength_uncertainty = dl_uncertainty
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
