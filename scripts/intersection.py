import sys
import logging
import argparse
import numpy as np
import scipy.optimize as opt
from common import Cache, EXPERIMENTAL_CONSTANTS as E, EnvDefault, MAGIC as M
from typing import Union
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def calculate_num_intersects(z: float, a: NDArray) -> int:
    """
    Takes an NDArray of shape (n, m, 3), where n are the number of samples,
    (m, 3) where m >= 2 is the array of vectors with the position vector
    in index 0. Any following vector is the momentum vector of a particle
    originating from the position.
    This function will calculate the number of particles i, where
    0 <= i <= n*(m-1), that will intersect with a disk at position (0, 0, z).
    """
    z_travel = z - a[:, 0, 2]

    # Filter out kaons that decay on/behind the detector
    mask = z_travel > 0
    z_travel, a = z_travel[mask], a[mask]

    # We flatten our momentum vectors into one big array, and repeat z_travel for every mom. vec.
    z_travel = np.repeat(z_travel, a.shape[1] - 1)
    momentum_vecs = a[:, 1:, :].reshape(-1, 3)
    xy_offsets = np.repeat(a[:, 0, :2], a.shape[1] - 1, axis=0)

    with np.errstate(divide="ignore"):
        # We ignore division by 0, as that just means travel perpendicular to our detector, and
        # float('inf') is not going to be intersecting. (But imagine the probability of a div by 0)
        travel_time = z_travel / momentum_vecs[:, 2]

    # Filter out the ones going backwards in time
    mask = travel_time > 0
    travel_time, momentum_vecs, xy_offsets = travel_time[mask], momentum_vecs[mask], xy_offsets[mask]

    radius_sq_travelled = np.sum(
        (momentum_vecs[:, :2] * travel_time[:, np.newaxis] + xy_offsets) ** 2,
        axis=1,
    )

    logger.debug(f"Returning {(ret_val := np.sum(radius_sq_travelled <= E.d2radsq))} intersections for {z=}")
    return ret_val


def maximize(fun, bracket, *args) -> float:
    result = opt.minimize_scalar(lambda *a, **k: -fun(*a, **k), bracket=bracket, args=args)
    return result["x"]


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
    logger.debug("Calculating not angled intersections")
    optimal_not_angled_z = maximize(
        calculate_num_intersects,
        (1, cache.average_decay_length, 2 * cache.average_decay_length),
        cache.not_angled_sample,
    )
    cache.not_angled_ideal_z = optimal_not_angled_z
    logger.debug("Calculating angled intersections")
    optimal_angled_z = maximize(
        calculate_num_intersects,
        (1, cache.average_decay_length, 2 * cache.average_decay_length),
        cache.angled_sample,
    )
    cache.angled_ideal_z = optimal_angled_z
    logger.info(
        f"Found the optimal z for straight Kaon beam: {cache.not_angled_ideal_z}; for angled Kaon beam: {cache.angled_ideal_z}"
    )
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

    args = parser.parse_args()
    if hasattr(args, "cache") and args.cache is not None:
        args.no_write = True

    logging.basicConfig(
        stream=sys.stdout,
        level={0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG, -1: logging.ERROR, -2: logging.CRITICAL}.get(
            min(max(args.verbose - args.quiet, -2), 2), logging.WARNING
        ),
        format=M.logger_fmt,
        datefmt=M.logger_datefmt,
    )
    logger.info(f"Parsed arguments: {args}")

    if args.no_write:
        sys.exit(main(args)[0])
    sys.exit(main(args))
