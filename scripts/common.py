from __future__ import annotations

import argparse
import os
import json
import numpy as np
from typing import Any, Union
from numpy.typing import NDArray

# https://stackoverflow.com/questions/10551117/setting-options-from-environment-variables-when-using-argparse
class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs):
        if envvar:
            if envvar in os.environ:
                default = os.environ[envvar]
        if required and default:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required,
                                         **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class Cache:
    def __init__(self, file_or_json: Union[str, bytes]):
        if isinstance(file_or_json, bytes):
            self.data = vc_is_dict(json.loads(file_or_json))
        if isinstance(file_or_json, str):
            if not os.path.exists(file_or_json):
                raise EnvironmentError("Value cache file does not seem to exist / has too restrictive permissions")
            with open(file_or_json, 'r') as f:
                self.data = vc_is_dict(json.load(f))
        raise TypeError(f"`file_or_json` has invalid type: {type(file_or_json)}. Expected: str | bytes.")
    def to_hex(self) -> str:
        return bytes(json.dumps(self.data), "UTF-8").hex()
    def dump(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.data, f)
    @property
    def average_decay_length(self) -> float:
        return self.data.get("average_decay_length", 4188)
    @average_decay_length.setter
    def average_decay_length(self, value: float):
        self.data["average_decay_length"] = value
    @property
    def not_angled_sample(self) -> NDArray:
        return np.array(self.data.get("not_angled_sample"))
    @not_angled_sample.setter
    def not_angled_sample(self, value: NDArray):
        self.data["not_angled_sample"] = value.tolist()

def vc_is_dict(a: Any) -> dict:
    if not isinstance(a, dict):
        raise EnvironmentError("Bad value cache file_or_json")
    return a
