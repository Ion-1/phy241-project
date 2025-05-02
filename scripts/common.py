from __future__ import annotations

import os
import json
import base64
import argparse
import numpy as np

from functools import cached_property
from typing import Any, Union, TypeVar, Optional
from numpy.typing import NDArray
from dataclasses import dataclass


# Physical constants
@dataclass
class CONSTANTS:
    EV_TO_JOULES = 1.602176634e-19
    SPEED_OF_LIGHT = 299792458  # in ms^-1
    PION_POS_MASS = 139.57039  # in MeV/c^2
    PION_NEU_MASS = 134.9768  # in MeV/c^2
    KAON_MASS = 493.677  # in MeV/c^2
    KAON_MEANLIFE = 1.2380e-8  # in s
    # Aliases
    c = SPEED_OF_LIGHT
    m_k = KAON_MASS
    m_pp = PION_POS_MASS
    m_np = PION_NEU_MASS
    t_k = KAON_MEANLIFE
    e = EV_TO_JOULES  # elementary charge
    MeV2mps = 10**6 * EV_TO_JOULES / SPEED_OF_LIGHT


@dataclass
class EXPERIMENTAL_CONSTANTS:
    DETECTOR2_RADIUS = 2  # in m
    DETECTOR2_RADIUS_SQUARED = 4
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
