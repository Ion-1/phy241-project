import sys
import numpy as np
import argparse
from common import EnvDefault, Cache
from numpy.typing import NDArray


def generate_sample(avg_dlength: float) -> NDArray:
    pass


def main(args: argparse.Namespace) -> int:
    if "cache" in args:
        cache = Cache(bytes.fromhex(args.cache))
    else:
        cache = Cache(args.cache_file)
    sample = generate_sample(cache.average_decay_length)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--cache-file", action=EnvDefault, envvar='VALUECACHE', default=r".\data\value_cache.json", help="File path of the value cache.")
    group.add_argument("--cache", help="A hexadecimal bytes representation of a JSON string containing the value cache data")
    args = parser.parse_args()
    sys.exit(main(args))
