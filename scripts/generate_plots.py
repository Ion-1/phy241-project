import os
import sys
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
import vispy.io as io

from vispy import app, scene

from typing import overload, Literal as L, Union
from numpy.typing import NDArray

from find_dec_length import nll, distribution
from common import Cache, MAGIC as M, CONSTANTS as C, EXPERIMENTAL_CONSTANTS as E, EnvDefault


logger = logging.getLogger(__name__)


def plot_nll(cache: Cache, interactive: bool):
    fig1, fig2 = _plot_nll(cache)
    if interactive:
        fig1.show()
        fig2.show()
    else:
        fig1.savefig("./graphs/task2_nll.png")
        fig2.savefig("./graphs/task2_hist.png")


def _plot_nll(cache: Cache):
    logger.info("Plotting NLL")
    data = np.loadtxt("./data/dec_lengths.txt")

    fig1, ax1 = plt.subplots(figsize=(8, 4), dpi=200, layout="tight")
    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=200, layout="tight")

    xlim = (520, 600)
    values = np.linspace(*xlim, 100)
    nlls = np.vectorize(nll, excluded={"data"})(mean_k=values, data=data)
    ax1.plot(values, nlls, "b-")
    ax1.set_xlim(*xlim)
    ax1.set_xlabel("Decay Length [m]")
    ax1.set_ylabel("NLL [unitless]")
    ax1.set_title("Negative log likelihood vs. Average Decay Length")
    # ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    ax1.grid(True, linestyle="--", alpha=0.5)
    nll_val = nll(cache.adl, data)
    ax1.annotate(f"({cache.adl:.2f}, {nll_val:.2f})", (cache.adl, nll_val), (-150, 50), textcoords="offset pixels")
    ax1.plot(cache.adl, nll_val, "ko")
    lower = cache.adl - cache.dlength_uncertainty[0]
    nll_val = nll(lower, data)
    ax1.annotate(f"({lower:.2f}, {nll_val:.2f})", (lower, nll_val), (-350, -30), textcoords="offset pixels")
    ax1.plot(lower, nll_val, "ko")
    higher = cache.adl + cache.dlength_uncertainty[1]
    nll_val = nll(higher, data)
    ax1.annotate(f"({higher:.2f}, {nll_val:.2f})", (higher, nll_val), (100, -30), textcoords="offset pixels")
    ax1.plot(higher, nll_val, "ko")

    ax2.hist(data, bins=100, density=True)
    ax2.set_xlim(np.min(data), 30000)
    ax2.set_ylim(0, 0.0004)
    ax2.set_xlabel("Decay Length [m]")
    ax2.set_ylabel("Count")
    ax2.set_title("Histogram of Decay Lengths")
    x_vals = np.linspace(*ax2.get_xlim(), 1000)
    ax2.plot(
        x_vals, distribution(cache.adl, x_vals), label=f"The distribution with an a.d.l. of {cache.adl:.2f} meters"
    )
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    return fig1, fig2


def plot3d(a: NDArray, name: str, detector_z: float):
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 8), dpi=200, subplot_kw={"projection": "3d"}, layout="tight")
    # fig.suptitle(name)
    a = a[:: M.sample_size // 100]  # only take 100 element subset of sample
    # a[:, 1, :] = a[:, 1, :] / (C.m_pp * C.MeVperCsq2kg)
    # a[:, 2, :] = a[:, 2, :] / (C.m_np * C.MeVperCsq2kg)
    pos, mom, num, ipos, imom = extend_vectors(detector_z, a, True)

    # ax1.quiver(
    #     empty := np.zeros(pos.shape[0]),
    #     empty,
    #     empty,
    #     pos[:, 0],
    #     pos[:, 1],
    #     pos[:, 2],
    #     arrow_length_ratio=0,
    # )

    r = np.linspace(0, E.DETECTOR2_RADIUS, 50)
    phi = np.linspace(0, 2 * np.pi, 50)
    x = np.outer(r, np.cos(phi))
    y = np.outer(r, np.sin(phi))
    z = np.full(x.shape, detector_z)
    ax2.plot_surface(x, y, z, label="Detector", shade=False, antialiased=False, color="orange", alpha=0.3)
    ax2.plot(pos[:, 0], pos[:, 1], pos[:, 2], "kx", markersize=5, label="Decay Vertices")
    ax2.plot((ipos + imom)[:, 0], (ipos + imom)[:, 1], (ipos + imom)[:, 2], "rx", markersize=5, label="Intersection Points")

    ax2.quiver(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        mom[:, 0],
        mom[:, 1],
        mom[:, 2],
        arrow_length_ratio=0,
        label="$\pi^+$ and $\pi^0$ flight directions",
        alpha=0.5,
    )

    ax2.set_zlim(0, detector_z)
    # ax1.set_title("Kaon decay vertices in $m$")
    ax2.set_title(f"{name} with {num} intersecting particles")
    # ax1.view_init(200, 0, 90)
    ax2.view_init(200, 0, 90)
    ax2.legend(loc='upper left')

    return fig


def plot_samples(cache: Cache, interactive: bool = False):
    logger.info("Plotting not-divergent beam")
    fig1 = plot3d(cache.not_angled_sample, "Not divergent beam", cache.not_angled_ideal_z)
    logger.info("Plotting divergent beam")
    fig2 = plot3d(cache.angled_sample, "Divergent beam", cache.angled_ideal_z)
    if interactive:
        plt.show()
    else:
        fig1.savefig(name := f"./graphs/task3_sample_not_divergent_beam.png")
        logger.info(f"Plot saved to {name}")
        fig2.savefig(name := f"./graphs/task3_sample_divergent_beam.png")
        logger.info(f"Plot saved to {name}")


def plot_samples_vispy(cache: Cache, interactive: bool = False):
    logger.info("Creating vispy canvas for not divergent beam")
    a = create_canvas(cache.not_angled_sample, cache.not_angled_ideal_z)
    logger.info("Creating vispy canvas for divergent beam")
    b = create_canvas(cache.angled_sample, cache.angled_ideal_z)
    if interactive:
        app.run()
    else:
        io.write_png("./graphs/task3_sample_not_divergent.png", a.render())
        io.write_png("./graphs/task3_sample_divergent.png", b.render())
        logger.info("Plots saved to ./graphs")
        app.quit()

@overload
def extend_vectors(z: float, a: NDArray, num: L[False]) -> tuple[NDArray, NDArray]: ...
@overload
def extend_vectors(z: float, a: NDArray, num: L[True]) -> tuple[NDArray, NDArray, int, NDArray, NDArray]: ...
def extend_vectors(z: float, a: NDArray, num: bool = False) -> Union[tuple[NDArray, NDArray], tuple[NDArray, NDArray, int]]:
    """
    Extends the vectors in a sample to the detector.
    """
    z_travel = z - a[:, 0, 2]

    # Filter out kaons that decay on/behind the detector
    mask = z_travel > 0
    z_travel, a = z_travel[mask], a[mask]

    # We flatten our momentum sample into one big array, and repeat z_travel for every mom. vec.
    z_travel = np.repeat(z_travel, a.shape[1] - 1)
    momentum_vecs = a[:, 1:, :].reshape(-1, 3)
    decays = np.repeat(a[:, 0], a.shape[1] - 1, axis=0)

    with np.errstate(divide="ignore"):
        # We ignore division by 0, as that just means travel perpendicular to our detector, and
        # float('inf') is not going to be intersecting. (But imagine the probability of a div by 0)
        travel_time = z_travel / momentum_vecs[:, 2]

    # Filter out the ones going backwards in time
    mask = travel_time > 0
    travel_time, momentum_vecs, pos = travel_time[mask], momentum_vecs[mask], decays[mask]

    mask = np.sum((momentum_vecs[:, :2] * travel_time[:, np.newaxis] + pos[:, :2]) ** 2, axis=1) <= E.d2radsq
    if num:
        it, imom, ipos = travel_time[mask], momentum_vecs[mask], pos[mask]
        return pos, momentum_vecs * travel_time[:, np.newaxis], np.sum(mask), ipos, imom*it[:, np.newaxis]

    return pos, momentum_vecs * travel_time[:, np.newaxis]


def create_canvas(sample: NDArray, detector_z: float):
    canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="white")
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(fov=45, azimuth=30, elevation=30, distance=10)

    step = sample.shape[0] // M.plot_size
    sample[:, 1, :] = sample[:, 1, :] / (C.m_pp * C.MeVperCsq2kg)
    sample[:, 2, :] = sample[:, 2, :] / (C.m_np * C.MeVperCsq2kg)
    pos, d = extend_vectors(detector_z, sample[::step])

    arrows = np.hstack([pos, (end := pos + d)])
    logger.debug(f"{arrows.shape=}")
    vpos = np.empty((2 * pos.shape[0], 3), dtype=float)
    vpos[0::2] = pos
    vpos[1::2] = end

    arrow = scene.visuals.Arrow(
        pos=vpos,
        arrows=arrows,
        connect="segments",
        arrow_type="angle_30",
        arrow_size=50,
        color="red",
        arrow_color="red",
        parent=view.scene,
    )

    num_segments = 100
    radius = E.DETECTOR2_RADIUS
    z0 = detector_z
    # a) circle in XY at z=z0
    theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    z = np.full_like(x, z0)

    # b) vertices: first is center, then the rim
    verts = np.zeros((num_segments + 1, 3), dtype=float)
    verts[0] = (0.0, 0.0, z0)
    verts[1:] = np.column_stack((x, y, z))

    # c) faces: triangles [0, i, i+1]
    faces = []
    for i in range(1, num_segments):
        faces.append([0, i, i + 1])
    faces.append([0, num_segments, 1])
    faces = np.array(faces, dtype=np.uint32)

    # d) make the mesh
    disk = scene.visuals.Mesh(vertices=verts, faces=faces, color=(0.2, 0.2, 1.0, 0.3), parent=view.scene)

    axes = scene.visuals.XYZAxis(parent=view.scene)

    return canvas


def main(args: argparse.Namespace) -> int:
    if not os.path.exists("./graphs/cat.png"):
        logger.fatal("No cat (┬┬﹏┬┬)")
        return 1
    if hasattr(args, "cache") and args.cache is not None:
        if isinstance(args.cache, Cache):
            logger.debug("Cache provided as-is in namespace")
            cache = args.cache
        else:
            logger.info("Loading cache from b64 string")
            cache = Cache.from_b64(args.cache)
    else:
        logger.info("Loading cache from file")
        cache = Cache(args.cache_file)
    plot_nll(cache, args.interactive)
    plot_samples(cache, args.interactive)
    # plot_samples_vispy(cache, args.interactive)
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quiet", action="count", help="Decrease output verbosity.", default=0)
    parser.add_argument("-v", "--verbose", action="count", help="Increase output verbosity.", default=0)
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive graphs.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-c",
        "--cache-file",
        action=EnvDefault,
        type=str,
        envvar="VALUECACHE",
        default=r"./data/value_cache.json",
        help="File path of the value cache.",
    )
    group.add_argument(
        "--cache",
        help="A base64 representation of a UTF-8 JSON string containing the value cache data. Implies `--no-write`.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level={0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG, -1: logging.ERROR, -2: logging.CRITICAL}.get(
            min(max(args.verbose - args.quiet, -2), 2), logging.WARNING
        ),
        format=M.logger_fmt,
        datefmt=M.logger_datefmt,
    )
    logger.info(f"Parsed arguments: {args}")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    sys.exit(main(args))
