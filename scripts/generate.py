import sys
import logging
import argparse
import numpy as np

import find_dec_length
import experiment_simulation
import intersection

from argparse import Namespace
from common import load_seedsequence, Cache, EnvDefault

from typing import Union

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> Union[int, tuple[int, Cache]]:
    if hasattr(args, "cache") and args.cache is not None:
        if isinstance(args.cache, Cache):
            cache = args.cache
        else:
            cache = Cache.from_b64(args.cache)
    else:
        cache = Cache(args.cache_file)
    ssequence = load_seedsequence(args.seed, args.seed_file, args.no_write_out_seed)[0]
    logger.info("Calculating average decay length...")
    _, cache = find_dec_length.main(Namespace(cache=cache, no_write=True))
    logger.info("Generating experiment sample")
    _, cache = experiment_simulation.main(
        Namespace(cache=cache, seed=ssequence, no_write=True, seed_file=None, write_out_seed=False)
    )  # Spawning a child from ssequence is done by `load_seed_sequence` within main
    logger.info("Finished generating samples")
    logger.info("Calculating intersections and optimizing z")
    _, cache = intersection.main(Namespace(cache=cache, no_write=True))
    cache.dump_readable_summary(args.summary)
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
        default=r"./data/value_cache.json",
        help="File path of the value cache.",
    )
    group.add_argument(
        "-s",
        "--summary",
        action=EnvDefault,
        type=str,
        envvar="SUMMARYFILE",
        default=r"./data/summary.txt",
        help="File path for the human readable data summary"
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
        help="Seed for the default_rng. To see what is accepted, check `load_seedsequence` in `common.py`.",
        default=None,
    )
    parser.add_argument(
        "--seed-file",
        help="File at which seed should be stored. Check `load_seedsequence` in `common.py` for more.",
        default="./data/entropy",
    )
    parser.add_argument(
        "--no-write-out-seed",
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
