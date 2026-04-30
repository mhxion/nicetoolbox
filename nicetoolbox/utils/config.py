"""
Functions for handling configuration files.
"""

import json
from pathlib import Path

import numpy as np
import toml
import yaml


def default(obj):
    # serialize numpy array for saving to json
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj.item()
    raise TypeError("Unknown type:", type(obj))


def save_config(configs: dict, config_file: str) -> None:
    """
    Save the given configuration data to the specified file.

    Args:
        configs (dict): The configuration data to be saved.
        config_file (str): The path to the file where the configuration data
        will be saved.

    Raises:
        NotImplementedError: If the file type is not supported.
            Supported types are yaml/yml and toml.

    Note:
        If the file type is Windows, it will convert the paths to Windows format.
    """
    config_file = Path(config_file)

    if config_file.suffix in [".yml", ".yaml"]:
        with open(config_file, "w") as file:
            yaml.dump_all(configs, file, default_flow_style=False, indent=4, sort_keys=False)
    elif config_file.suffix == ".toml":
        with open(config_file, "w") as file:
            toml.dump(configs, file, encoder=toml.TomlNumpyEncoder())
    elif config_file.suffix == ".json":
        with open(config_file, "w") as file:
            json.dump(configs, file, default=default)
    else:
        raise NotImplementedError(
            f"config_file type {config_file} is not supported currently. " f"Implemented are yaml/yml and toml."
        )
