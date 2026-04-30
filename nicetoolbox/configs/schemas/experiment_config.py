from pathlib import Path

from pydantic import BaseModel

from .dataset_properties import DatasetProperties
from .detectors_config import DetectorsConfig
from .detectors_run_file import DetectorsRunFile
from .evaluation_config import EvaluationConfig
from .machine_specific_paths import MachineSpecificConfig
from .predictions_mapping import PredictionsMappingConfig
from .project_config import ProjectConfig


class CodeConfig(BaseModel):
    """
    Auto-placeholders saved during detectors run.
    """

    user_name: str
    git_hash: str
    commit_message: str
    yyyymmdd: str
    time: str
    pwd: Path


class DetectorsExperimentConfig(BaseModel):
    """
    Combined schema for detectors experiment output.
    Describe all relevant configuration state during detectors run.
    Used as an input for visualizer and evaluation.
    """

    # args from command line
    project_folder: Path
    project_config_path: Path
    machine_specific_path: Path
    run_config_file_path: Path

    # calculated during runtime
    code_config: CodeConfig

    # configs loaded and resolved
    machine_specific_config: MachineSpecificConfig
    project_config: ProjectConfig
    run_config: DetectorsRunFile
    dataset_config: DatasetProperties
    detector_config: DetectorsConfig
    predictions_mapping: PredictionsMappingConfig


class EvaluationExperimentConfig(BaseModel):
    """
    Combined schema for evaluation experiment output.
    Describe all relevant configuration state during evaluation run.
    Used for logging purposes.
    """

    # args from command line
    project_folder: Path
    project_config_path: Path
    machine_specific_path: Path
    evaluation_config_path: Path

    # calculated during runtime
    code_config: CodeConfig

    # configs loaded and resolved
    machine_specific_config: MachineSpecificConfig
    project_config: ProjectConfig
    evaluation_config: EvaluationConfig
    predictions_mapping: PredictionsMappingConfig
