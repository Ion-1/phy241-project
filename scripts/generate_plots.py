import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

from vispy import app, scene
from vispy.visuals import MeshVisual
from vispy.scene.visuals import Arrow

from numpy.typing import NDArray

from find_dec_length import nll, distribution
from common import Cache, MAGIC as M, CONSTANTS as C, EXPERIMENTAL_CONSTANTS as E


logger = logging.getLogger(__name__)


def plot_nll(cache: Cache):
    logger.info("Plotting NLL")
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


def plot3d(a: NDArray, name: str, detector_z: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 7.2), dpi=150, subplot_kw={"projection":"3d"})
    fig.suptitle(name)
    a = a[::M.sample_size//100] # only take 100 element subset of sample
    a[:,1,:] = a[:,1,:] / (C.m_pp * C.MeVperCsq2kg)
    a[:,2,:] = a[:,2,:] / (C.m_np * C.MeVperCsq2kg)

    ax1.quiver(empty := np.zeros(a.shape[0]), empty, empty, a[:,0,0], a[:,0,1], a[:,0,2])

    r = np.linspace(0, E.DETECTOR2_RADIUS, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    x = np.outer(r, np.cos(phi))
    y = np.outer(r, np.sin(phi))
    z = np.full(x.shape, detector_z)
    ax2.plot_surface(x, y, z, label="Detector", shade=False, antialiased=False, color='orange')

    ax2.quiver(a[:,0,0], a[:,0,1], a[:,0,2], a[:,1,0], a[:,1,1], a[:,1,2], label="π⁺ velocities", normalize=True, length=1)
    ax2.quiver(a[:,0,0], a[:,0,1], a[:,0,2], a[:,2,0], a[:,2,1], a[:,2,2], label="π⁰ velocities", normalize=True, length=1)

    ax2.set_zlim(0, detector_z)
    ax1.set_title("Kaon decay vertices in $m$")
    ax2.set_title("Pion velocities in $ms^{-1}$")
    ax2.legend()
    ax1.view_init(210, 0, 90)
    ax2.view_init(210, 0, 90)
    fig.savefig(f"./graphs/task3_sample_{name.replace(' ', '_').lower()}.png")


def plot_samples(cache: Cache):
    logger.info("Plotting not-divergent beam")
    plot3d(cache.not_angled_sample, "Not divergent beam", cache.not_angled_ideal_z)
    logger.info("Plotting divergent beam")
    plot3d(cache.angled_sample, "Divergent sample", cache.angled_ideal_z)

def main(*args: str) -> int:
    if not os.path.exists("./graphs/cat.png"):
        return 1
    cache = Cache(r"./data/value_cache.json")
    plot_nll(cache)
    plot_samples(cache)
    return 0


if __name__ == "__main__":

    fmt = "[%(levelname)s|%(name)s] %(asctime)s: %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    sys.exit(main(*sys.argv))
