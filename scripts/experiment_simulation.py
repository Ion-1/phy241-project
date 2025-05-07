import sys
import logging
import argparse
import numpy as np
import scipy.stats as st

from numpy.random import Generator
from scipy.spatial.transform import Rotation as R
from common import EnvDefault, Cache, CONSTANTS as C, MAGIC as M, load_seedsequence, EXPERIMENTAL_CONSTANTS as E

from typing import Union
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def generate_sample_kaon_decay(avg_dlength: float, n: int, rng: Generator) -> NDArray:
    """
    Generates a (n, 3, 3) ndarray that represents a sample of K+ decay.
    The first index indexes the n samples.
    The second index indexes the (decay vertex, p_pos, p_neu) vector triplet,
    where p is the momentum vector representing the isotropically distributed
    (in the Kaon rest frame) momentum of the pions in the lab frame.
    Decay vertex represents the (0, 0, z) exponentially distributed position
    of the decay event in the z-direction, with a rate of 1/avg_dlength.

    The beam momentum is extrapolated from the decay length and the known mean life
    of the Kaon, and the momentum of the pions in the rest frame calculated
    at runtime and everything else from the constants in common.py.
    """
    # decay length \lambda = c * \beta * \gamma * \tau
    # momentum in MeV/c is \beta * \gamma * mass (in MeV/c^2)
    βγ = avg_dlength / C.t_k / C.c
    p_K = βγ * C.m_k
    E_K = np.sqrt(C.m_k**2 + p_K**2)
    # magnitude of pions' momentum stems from four-momentum conservation
    p_p = np.sqrt((C.m_k**2 - (C.m_pp + C.m_np) ** 2) * (C.m_k**2 - (C.m_pp - C.m_np) ** 2)) / 2 / C.m_k

    vertex_positions = np.concatenate(
        (
            np.zeros((n, 2)),
            np.reshape(rng.exponential(scale=avg_dlength, size=n), (n, 1)),
        ),
        axis=1,
    )
    isotrope_momenta = p_p * st.uniform_direction.rvs(dim=3, size=n, random_state=rng)

    E_positive_pion = np.sqrt(C.m_pp**2 + p_p**2)
    # Giving the positive pion the positive array makes no difference to giving it the neutral one
    pos_pi_4_momenta = np.concatenate((np.full((n, 1), E_positive_pion), isotrope_momenta), axis=1)
    E_neutral_pion = np.sqrt(C.m_np**2 + p_p**2)
    neu_pi_4_momenta = np.concatenate((np.full((n, 1), E_neutral_pion), -isotrope_momenta), axis=1)

    γ = E_K / C.m_k
    lorentz_boost = np.array([[γ, 0, 0, βγ], [0, 1, 0, 0], [0, 0, 1, 0], [βγ, 0, 0, γ]])
    boosted_pos_p_4m = C.MeV2mps * pos_pi_4_momenta @ lorentz_boost.T  # If only I could use np.matvec
    boosted_neu_p_4m = C.MeV2mps * neu_pi_4_momenta @ lorentz_boost.T  # but alas, python 3.9
    return np.stack((vertex_positions, boosted_pos_p_4m[:, 1:], boosted_neu_p_4m[:, 1:]), axis=1)


def rotate_sample(sample: NDArray, angle_std: float, rng: Generator) -> NDArray:
    """
    Takes a (n, 3, 3) NDArray sample of Kaon decay, and rotates the decay vertex and momentum
    vectors according to a Gaussian for the polar angle and a uniform distribution for the azimuthal angle.
    Due to rotational symmetry, we can simply reuse our non-angled sample.
    """
    n = sample.shape[0]

    azimuthal = rng.normal(loc=0, scale=angle_std, size=n)
    polar = rng.uniform(low=0, high=2 * np.pi, size=n)

    rotation = R.from_euler("zy", np.vstack((polar, azimuthal)).T)

    return np.einsum('ipc,icq->ipq', sample, rotation.as_matrix())


def main(args: argparse.Namespace) -> Union[int, tuple[int, Cache]]:
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
    rng = np.random.default_rng(load_seedsequence(args.seed, args.seed_file, args.write_out_seed)[0])
    sample = generate_sample_kaon_decay(cache.average_decay_length, M.sample_size, rng)
    logger.info("Finished generating straight-beam sample")
    cache.not_angled_sample = sample
    angled_sample = rotate_sample(sample, E.ANGULAR_DIVERGENCE_STD, rng)
    logger.info("Finished rotating straight-beam sample")
    cache.angled_sample = angled_sample
    if args.no_write:
        return 0, cache
    else:
        cache.dump(args.cache_file)
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quiet", action="count", help="Decrease output verbosity.", default=0)
    parser.add_argument("-v", "--verbose", action="count", help="Increase output verbosity.", default=0)
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
    parser.add_argument(
        "--seed",
        help="Seed for the default_rng. To see what is accepted, check `load_seedsequence` in `common.py`",
        default=None,
    )
    parser.add_argument(
        "--seed-file",
        help="File at which seed should be stored. Check `load_seedsequence` in `common.py` for more.",
        default=None,
    )
    parser.add_argument(
        "--write-out-seed",
        action="store_false",
        help="Whether to write out the seed used. See `load_seedsequence` in `common.py`.",
    )

    args = parser.parse_args()
    if hasattr(args, "cache") and args.cache is not None:
        args.no_write = True

    fmt = "[%(levelname)s|%(name)s] %(asctime)s: %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level={0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG, -1: logging.ERROR, -2: logging.CRITICAL}.get(
            min(max(args.verbose - args.quiet, -2), 2), logging.WARNING
        ),
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    logger.info(f"Parsed arguments: {args}")

    if args.no_write:
        sys.exit(main(args)[0])
    sys.exit(main(args))
