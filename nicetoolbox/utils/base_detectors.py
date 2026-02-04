"""
Helper functions for method and feature detector base classes.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

from pydantic import BaseModel

# (1) BASE METHOD and FEATURE utils
# ---------------------


def flatten_inference_config(static_config: BaseModel, runtime_config: BaseModel) -> Dict[str, Any]:
    """
    Build flattened inference config dictionary (Static + Runtime).

    Args:
        static_config: The loaded static properties (e.g. from TOML).
        runtime_config: The computed runtime properties.

    Returns:
        Flattened dictionary ready for serialization to run_config.toml
    """
    # 1. Dump static config
    inference_config = static_config.model_dump(by_alias=True)

    # 2. Remove the nested RuntimeConfig class definition if present in the dump
    inference_config.pop("RuntimeConfig", None)

    # 3. Dump runtime and merge (Runtime > Static)
    inference_config.update(runtime_config.model_dump())

    return inference_config


# (2) BASE FEATURE utils
# ---------------------


InputMapTuple = Dict[Tuple[str, str], Path]  # Type alias for input map with tuple keys (internal use)
InputMapStr = Dict[str, str]  # # Type alias for input map with string keys (serialization)


def tuple_key_to_string(key: Tuple[str, str]) -> str:
    """Convert tuple key to string for TOML serialization."""
    return f"{key[0]}__{key[1]}"


def string_key_to_tuple(key: str) -> Tuple[str, str]:
    """Convert string key back to tuple."""
    parts = key.split("__")
    return (parts[0], parts[1])


def input_map_to_string_keys(input_map: InputMapTuple) -> InputMapStr:
    """Convert input_map with tuple keys to string keys and paths for serialization."""
    return {tuple_key_to_string(k): str(v) for k, v in input_map.items()}


def input_map_to_tuple_keys(input_map: InputMapStr) -> InputMapTuple:
    """Convert input_map with string keys back to tuple keys with Path objects."""
    return {string_key_to_tuple(k): Path(v) for k, v in input_map.items()}
