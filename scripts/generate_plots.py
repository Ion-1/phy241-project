import sys
import numpy as np
import matplotlib.pyplot as plt
from find_dec_length import nll
import os


def plot():
    data = np.loadtxt("./data/dec_lengths.txt")
    plt.figure()
    plt.hist(data, bins=100)
    plt.xlabel("Decay Length [m]")
    plt.ylabel("Count")
    plt.title("Histogram of Decay Lengths")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("./graphs/task2_decay.png")

def plot_nll():
    data=np.loadtxt("./data/dec_lengths.txt")
    values=np.linspace(540,580,100)
    nlls=np.vectorize(nll, excluded={"data"})(mean_k=values,data=data)
    fig,ax=plt.subplots()
    ax.plot(values,nlls)
    ax.set_xlabel("Decay Length [m]")
    ax.set_ylabel("NLL")
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("./graphs/task2_nll.png")

def main(*args: str) -> int:
    if not os.path.exists("./graphs/cat.png"):
        return 1
    plot()
    plot_nll()
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv))
