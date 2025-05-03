from __future__ import annotations

import os
import io
import json
import base64
import pickle
import logging
import argparse

import numpy as np

from dataclasses import dataclass
from numpy.random import SeedSequence

from typing import Any, Union, TypeVar, Optional
from typing_extensions import Self
from numpy.typing import NDArray


global logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MAGIC:
    sample_size = 100000


# Physical constants
@dataclass(frozen=True)
class CONSTANTS:
    EV_TO_JOULES = 1.602176634e-19
    SPEED_OF_LIGHT = 299792458  # in ms^-1
    PION_POS_MASS = 139.57039  # in MeV/c^2
    PION_NEU_MASS = 134.9768  # in MeV/c^2
    KAON_MASS = 493.677  # in MeV/c^2
    KAON_MEANLIFE = 1.2380e-8  # in s
    PION_DECAY_LENGTH = 4188 # in m
    # Aliases
    c = SPEED_OF_LIGHT
    m_k = KAON_MASS
    m_pp = PION_POS_MASS
    m_np = PION_NEU_MASS
    t_k = KAON_MEANLIFE
    e = EV_TO_JOULES  # elementary charge
    MeV2mps = 10**6 * EV_TO_JOULES / SPEED_OF_LIGHT


@dataclass(frozen=True)
class EXPERIMENTAL_CONSTANTS:
    DETECTOR2_RADIUS = 2  # in m
    DETECTOR2_RADIUS_SQUARED = 4
    PERCENTAGE_PIONS_BEAM = 0.84
    PERCENTAGE_KAONS_BEAM = 1 - PERCENTAGE_PIONS_BEAM
    # Aliases
    d2radsq = DETECTOR2_RADIUS_SQUARED


# https://stackoverflow.com/questions/10551117/setting-options-from-environment-variables-when-using-argparse
class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs):
        if envvar:
            if envvar in os.environ:
                default = os.environ[envvar]
        if required and default:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class Cache:
    def __init__(self, file_or_json: Union[str, bytes]):
        if isinstance(file_or_json, bytes):
            self.data = vc_is_dict(json.loads(file_or_json))
        elif isinstance(file_or_json, str):
            if not os.path.exists(file_or_json):
                raise EnvironmentError("Value cache file does not seem to exist / has too restrictive permissions")
            with open(file_or_json, "r") as f:
                self.data = vc_is_dict(json.load(f))
        else:
            raise TypeError(f"`file_or_json` has invalid type: {type(file_or_json)}. Expected: str | bytes.")

    @classmethod
    def from_b64(cls, data: str) -> Self:
        return cls(base64.b64decode(data))

    def to_b64(self) -> str:
        return base64.b64encode(bytes(json.dumps(self.data), "UTF-8")).decode()

    def dump(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=2)

    # I wish PEP 638 is a thing. So much boilerplate.
    @property
    def average_decay_length(self) -> float:
        return print_if_None_and_return(
            self.data.get(
                "average_decay_length",
            ),
            "No value for `average_decay_length` was retrieved",
            560,
        )

    @average_decay_length.setter
    def average_decay_length(self, value: float):
        self.data["average_decay_length"] = value

    @property
    def not_angled_sample(self) -> NDArray:
        return np.array(
            print_if_None_and_return(
                self.data.get("not_angled_sample"), "No value for `not_angled_sample` was retrieved", [[[]]]
            )
        )

    @not_angled_sample.setter
    def not_angled_sample(self, value: NDArray):
        self.data["not_angled_sample"] = value.tolist()

    @property
    def not_angled_ideal_z(self) -> float:
        return print_if_None_and_return(
            self.data.get(
                "not_angled_ideal_z",
            ),
            "No value for `not_angled_ideal_z` was retrieved",
            440,
        )

    @not_angled_ideal_z.setter
    def not_angled_ideal_z(self, value: float):
        self.data["not_angled_ideal_z"] = value

    @property
    def angled_sample(self) -> NDArray:
        return np.array(
            print_if_None_and_return(
                self.data.get("angled_sample"), "No value for `angled_sample` was retrieved", [[[]]]
            )
        )

    @angled_sample.setter
    def angled_sample(self, value: NDArray):
        self.data["angled_sample"] = value.tolist()

    @property
    def angled_ideal_z(self) -> float:
        return print_if_None_and_return(
            self.data.get(
                "angled_ideal_z",
            ),
            "No value for `angled_ideal_z` was retrieved",
            400,
        )

    @angled_ideal_z.setter
    def angled_ideal_z(self, value: float):
        self.data["angled_ideal_z"] = value


T = TypeVar("T")


def print_if_None_and_return(val: Optional[T], warning: str, default: T) -> T:
    if val is None:
        print(warning)
        return default
    return val


def vc_is_dict(a: Any) -> dict:
    if not isinstance(a, dict):
        raise EnvironmentError("Bad value cache file_or_json")
    return a


def load_seedsequence(
    seed: Optional[Union[str, SeedSequence]] = None, filename: Optional[str] = None, writeout: bool = True
) -> tuple[SeedSequence, int]:
    """
    Creates a SeedSequence by attempting to convert seed to an `int`, or unpickle an `int`,
    `tuple[int]`, `list[int]`, `None`, or a `SeedSequence`, where seed is base64 encoded bytes.
    If seed is a `SeedSequence`, it will spawn a child off it.
    If not provided, it will try unpickling the file at `filename` in binary mode before using
    `None` for the initial entropy.
    It will try to write out the entropy at filename if filename was provided,
    but `seed` or `None` were used instead (can be disabled with `writeout=False`).
    Returns a `SeedSequence` and an `int`, indicating whether seed (1), filename (2), or None (3)
    was used.
    """
    if isinstance(seed, np.random.bit_generator.SeedSequence):
        logger.debug("Spawning SeedSequence from existing")
        return_data = seed.spawn(1)[0], 1
    else:
        return_data = load_seedsequence_(seed, filename)
    if filename is not None and writeout and return_data[1] != 2:
        try:
            with open(filename, "wb") as fp:
                pickle.dump(return_data[0].entropy, fp, protocol=0)
                logger.info("Wrote SeedSequence entropy out to file")
        except Exception:
            pass
    return return_data


def load_seedsequence_(seed, filename) -> tuple[SeedSequence, int]:
    # Once upon a time I though exception-based handling wasn't that bad...
    if seed is not None:
        try:
            return_data = SeedSequence(int(seed)), 1
            logger.debug("Created SeedSequence from `seed` integer")
            return return_data
        except Exception:
            pass
        try:
            data = restricted_loads(base64.b64decode(seed))
            if isinstance(data, SeedSequence):
                logger.debug("Created SeedSequence from pickled SeedSequence")
                return data, 1
            return_data = SeedSequence(data), 1
            logger.debug("Created SeedSequence from pickled int or sequence[int]")
            return return_data
        except Exception:
            pass
    if filename is not None:
        try:
            with open(filename, "rb") as fp:
                data = RestrictedUnpickler(fp).load()
                if isinstance(data, SeedSequence):
                    logger.debug("Created SeedSequence from pickled SeedSequence from file")
                    return data, 2
                return_data = SeedSequence(data), 2
                logger.debug("Created SeedSequence from pickled int from file")
                return return_data
        except Exception:
            pass
    logger.debug("Created SeedSequence from None")
    return SeedSequence(None), 3


def b64_seedsequence(data: SeedSequence) -> str:
    """Returns base64-encoded pickled SeedSequence string."""
    return base64.b64encode(pickle.dumps(data)).decode()


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Only allow SeedSequence.
        if module == "numpy" and name == "SeedSequence":
            return getattr(np, name)
        # Forbid everything else.
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" % (module, name))


def restricted_loads(s):
    """Helper function analogous to pickle.loads()."""
    return RestrictedUnpickler(io.BytesIO(s)).load()
