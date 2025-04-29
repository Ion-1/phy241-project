import sys
import numpy as np
import argparse
from common import EnvDefault, Cache
from numpy.typing import NDArray
from typing import Union


def generate_sample(avg_dlength: float) -> NDArray:
    pass


def main(args: argparse.Namespace) -> Union[int, tuple[int, Cache]]:
    if hasattr(args, "cache"):
        cache = Cache(bytes.fromhex(args.cache))
    else:
        cache = Cache(args.cache_file)
    sample = generate_sample(cache.average_decay_length)
    cache.not_angled_sample = sample
    if args.no_write:
        return 0, cache
    else:
        cache.dump(args.cache_file)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--cache-file", action=EnvDefault, envvar='VALUECACHE', default=r".\data\value_cache.json", help="File path of the value cache.")
    group.add_argument("--cache", help="A hexadecimal bytes representation of a UTF-8 JSON string containing the value cache data. Makes `--no-write` true by default.")
    parser.add_argument("--no-write", type="store_true", help="Do not write out to the value cache. Additionally, main will return the updated value cache instead.")
    args = parser.parse_args()
    if hasattr(args, "cache"):
        args.no_write = True
    if args.no_write:
        sys.exit(main(args)[0])
    sys.exit(main(args))
