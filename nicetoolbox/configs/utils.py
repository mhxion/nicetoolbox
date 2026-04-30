# shared config utils functions

import getpass
import os
import time
from pathlib import Path
from pprint import pformat
from typing import Optional, Type, TypeVar

import pydantic
import toml
from pydantic import BaseModel

from ..utils.git_utils import try_get_toolbox_git_metadata

# general type for all pydantic models
ModelT = TypeVar("ModelT", bound=BaseModel)


class ConfigValidationError(Exception):
    """
    Custom exception for configuration validation errors.
    Provides better formatting for Pydantic validation errors.
    """

    def __init__(self, error: pydantic.ValidationError, filepath: Optional[Path] = None):
        message = f"Config validation error in {filepath.name}\n" if filepath else ""
        for err in error.errors():
            message += "=" * 40 + "\n"
            error_path = ".".join(str(loc) for loc in err["loc"])
            message += f"{err['msg']}: '{error_path}'.\n"
            message += pformat(err["input"], depth=2) + "\n"

        super().__init__(message)


def load_raw_config(config_file: Path) -> dict:
    """
    Load a configuration file in TOML format.

    Args:
        config_file (Path): The path to the configuration file.

    Returns:
        dict: The configuration data loaded from the file.

    Raises:
        IOError: When an array with no valid (existing).
        FileNotFoundError: If the file does not exist.
        NotImplementedError: If the file type is not supported.
        TomlDecodeError: Error while decoding toml.

    Note:
        If the operating system is Windows, the paths in the configuration data will
        be converted to Windows format.
    """

    if config_file.suffix == ".toml":
        config = toml.load(config_file)
    else:
        raise NotImplementedError(
            f"config_file type {config_file} is not supported. " f"Only toml config files are supported."
        )
    return config


def model_to_dict(model: BaseModel) -> dict:
    """Converts pydantic model into the primitives dict"""
    # pydantic tries to keep Paths, UUID and other fields as python types
    # that confuses our resolver. json mode force to convert them to primitives
    # we also serialize_as_any to force nested structured to be serialized
    # and some of our fields has aliases, so we force it use original names
    return model.model_dump(mode="json", serialize_as_any=True, by_alias=True)


def dict_to_model(config_raw: dict, schema: type[ModelT]) -> ModelT:
    """Converts and validate dict to pydantic model"""
    try:
        config = schema.model_validate(config_raw)
    except pydantic.ValidationError as e:
        raise ConfigValidationError(e) from None
    return config


def keys_collision_dict(a: dict, b: dict) -> set:
    """Returns key collision between two dictionaries"""
    return set(a) & set(b)


def merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dictionaries, raising an error if any keys overlap."""
    collision = keys_collision_dict(a, b)
    if collision:
        raise KeyError(f"Duplicate keys detected: {collision}")
    return {**a, **b}


def get_latest_experiment_config_path(exp_folder: Path) -> Path:
    # check if folder exist
    if not exp_folder.is_dir():
        raise FileNotFoundError(f"Experiment folder does not exist or is not a directory: {exp_folder}")
    # check if it's not empty
    config_files = sorted(list(exp_folder.glob("config_*.toml")))
    if not config_files:
        raise RuntimeError(f"No 'config_*.toml' files found in experiment folder: {exp_folder}")
    return config_files[-1]


def default_auto_placeholders(working_directory=None):
    if working_directory is None:
        working_directory = os.getcwd()

    git_metadata = try_get_toolbox_git_metadata(working_directory)
    if git_metadata is not None:
        git_hash, commit_message = git_metadata
        git_hash = git_hash[:7]
    else:
        git_hash, commit_message = "unknown", "unknown"

    placeholder_dict = dict(
        git_hash=git_hash,
        commit_message=commit_message,
        user_name=getpass.getuser(),
        yyyymmdd=time.strftime("%Y%m%d", time.localtime()),
        time=time.strftime("%H_%M", time.localtime()),
        pwd=working_directory,
    )

    return placeholder_dict


def default_runtime_placeholders():
    return {
        "cur_video_length",
        "cur_video_start",
        "cur_sequence_ID",
        "cur_dataset_name",
        "cur_session_ID",
        "cur_cam_face1",
        "cur_cam_face2",
        "cur_cam_top",
        "cur_cam_front",
        "cur_algorithm_name",
        "cur_component_name",
        "cur_camera_name",
        "cur_metric_name",
    }


# TODO: Deprecated and should be removed after updating rerun visualization
def load_validated_config_raw(config_filepath: str, schema: Type[ModelT]) -> dict:
    """
    Load a configuration file, validate it using a Pydantic model
    and return the raw dictionary.

    Args:
        config_filepath (str): Path to the config file.
        schema (Type[ModelT]): Pydantic model class used to validate.

    Returns:
        dict: Raw config dictionary.

    Raises:
        IOError: When an array with no valid (existing)
        FileNotFoundError: If the file does not exist.
        NotImplementedError: If the file type is not supported.
        TomlDecodeError: Error while decoding toml
        ConfigValidationError: If validation fails.
    """
    config_raw = load_raw_config(config_filepath)
    try:
        schema.model_validate(config_raw, extra="forbid")
    except pydantic.ValidationError as e:
        raise ConfigValidationError(e, Path(config_filepath)) from None
    return config_raw
