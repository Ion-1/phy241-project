import os
import sys
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from find_dec_length import nll, distribution
from common import Cache


def plot_nll(cache: Cache):
    data=np.loadtxt("./data/dec_lengths.txt")

    fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=(1,1), figsize=(12.8, 7.2), dpi=150, layout="tight")

    xlim = (520, 600)
    values=np.linspace(*xlim,100)
    nlls=np.vectorize(nll, excluded={"data"})(mean_k=values,data=data)
    ax1.plot(values, nlls, "b-")
    ax1.set_xlim(*xlim)
    ax1.set_xlabel("Decay Length [m]")
    ax1.set_ylabel("NLL [unitless]")
    ax1.set_title("Negative log likelihood vs. Average Decay Length")
    # ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    ax1.grid(True, linestyle="--", alpha=0.5)
    nll_val = nll(cache.adl, data)
    ax1.annotate(
        f"({cache.adl:.2f}, {nll_val:.2f})", (cache.adl, nll_val), (-100, 70), textcoords="offset pixels"
    )
    ax1.plot(cache.adl, nll_val, "ko")
    lower = cache.adl - cache.dlength_uncertainty[0]
    nll_val = nll(lower, data)
    ax1.annotate(
        f"({lower:.2f}, {nll_val:.2f})", (lower, nll_val), (-250, 0), textcoords="offset pixels"
    )
    ax1.plot(lower, nll_val, "ko")
    higher = cache.adl + cache.dlength_uncertainty[1]
    nll_val = nll(higher, data)
    ax1.annotate(
        f"({higher:.2f}, {nll_val:.2f})", (higher, nll_val), (30, 0), textcoords="offset pixels"
    )
    ax1.plot(higher, nll_val, "ko")

    ax2.hist(data, bins=100, density=True)
    ax2.set_xlim(np.min(data), 30000)
    ax2.set_ylim(0, 0.0004)
    ax2.set_xlabel("Decay Length [m]")
    ax2.set_ylabel("Count")
    ax2.set_title("Histogram of Decay Lengths")
    x_vals = np.linspace(*ax2.get_xlim(), 1000)
    ax2.plot(x_vals, distribution(cache.adl, x_vals), label=f"The distribution with an a.d.l. of {cache.adl:.2f} meters")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.savefig("./graphs/task2_nll.png")

def main(*args: str) -> int:
    if not os.path.exists("./graphs/cat.png"):
        return 1
    cache = Cache(r"./data/value_cache.json")
    plot_nll(cache)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv))
