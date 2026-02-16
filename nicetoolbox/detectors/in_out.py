"""
IO module for the NICE toolbox.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from ..configs.placeholders import resolve_placeholders
from ..configs.video_runtime_config import VideoRuntimeConfig
from ..utils import check_and_exception as exc
from ..utils import system as oslab_sys


class VideoIO:
    """
    IO for per-video operations.

    Handles:
    - Video-specific folder creation
    - Detector output paths
    - Data source paths
    - Calibration file access
    """

    # All attributes declared for type checking
    algorithm_names: List[str]
    out_folder: Path
    out_sub_folder: Path
    csv_folder: Path
    code_folder: Path
    nice_input_folder: Path
    _data_source_folder: Path
    calibration_file: Optional[Path]
    conda_path: Path
    _video_context: VideoRuntimeConfig

    def __init__(
        self,
        video_context: VideoRuntimeConfig,
        algorithm_names: List[str],
    ):
        """
        Initialize for video processing.

        Args:
            video_context: Frozen video runtime configuration
            algorithm_names: List of all algorithm names for folder creation
        """
        self.algorithm_names = algorithm_names
        self._video_context = video_context

        # All paths from resolved IO config
        io = video_context.io
        self.out_folder = io.out_folder
        self.out_sub_folder = io.out_sub_folder
        self.csv_folder = io.csv_out_folder
        self.code_folder = io.code_folder
        self.nice_input_folder = io.nicetoolbox_input_folder

        # Dataset properties
        self._data_source_folder = video_context.data_source_folder
        self.calibration_file = video_context.calibration_path

        # Machine config
        self.conda_path = video_context.machine.conda_path

        # Create folders
        self._create_folders()

    def _create_folders(self) -> None:
        """Create necessary output and data folders."""
        self.out_sub_folder.mkdir(parents=True, exist_ok=True)
        self.csv_folder.mkdir(parents=True, exist_ok=True)
        self.nice_input_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Path Getters
    # -------------------------------------------------------------------------

    def get_data_source_folder(self, camera_name: str) -> Path:
        """
        Returns the folder path to the original dataset source data. (E.g. storing mp4/avi files)

        Args:
            camera_name (str): Specific camera name, used for path resolut

        Returns:
            Path: The path to the source data folder.
        """
        resolved_path = resolve_placeholders(self._data_source_folder, {"cur_camera_name": camera_name})
        return resolved_path

    def get_nice_input_folder(self) -> Path:
        """
        Returns the data folder associated with the current instance.

        Returns:
            str: The path to the data folder.
        """
        return self.nice_input_folder

    def get_calibration_file(self):
        """
        Returns the calibration file path.

        Returns:
            str: The path of the calibration file.
        """
        return self.calibration_file

    def get_conda_path(self):
        """
        Returns the path to the Conda installation directory.

        Returns:
            str: The path to the Conda installation directory.
        """
        return self.conda_path

    def get_inference_path(self, component_name, detector_name):
        """
        Get the file path for the inference script of a given detector.

        Args:
            detector_name (str): The name of the detector.

        Returns:
            str: The file path for the inference script.

        Raises:
            FileNotFoundError: If the inference script file does not exist.
        """
        filepath = os.path.join(
            self.code_folder,
            "nicetoolbox",
            "detectors",
            "method_detectors",
            component_name,
            f"{detector_name}_inference.py",
        )
        try:
            exc.file_exists(filepath)
        except FileNotFoundError:
            logging.exception(f"Detector inference file {filepath} does not exist!")
            raise
        return filepath

    def get_venv_path(self, detector_name, env_name):
        """
        Get the file path of the virtual environment for the given detector and
        environment name.

        Args:
            detector_name (str): The name of the detector.
            env_name (str): The name of the environment.

        Returns:
            str: The file path of the virtual environment.

        Raises:
            FileNotFoundError: If the virtual environment does not exist.
        """
        os_type = oslab_sys.detect_os_type()
        if os_type == "linux":
            filepath = os.path.join(self.code_folder, "envs", env_name, "bin/activate")
        elif os_type == "windows":
            filepath = os.path.join(self.code_folder, "envs", env_name, "Scripts", "activate")
        try:
            exc.file_exists(filepath)
        except FileNotFoundError:
            logging.exception(
                f"Virtual environment file {filepath} for detector = " f"'{detector_name}' does not exist!"
            )
            raise
        return filepath

    def get_output_folder(self, token: str) -> Path:
        """
        Get output folder by token.

        Args:
            token: One of 'output', 'main', 'csv'
        """
        if token == "output":
            return self.out_sub_folder
        if token == "main":
            return self.out_folder
        if token == "csv":
            os.makedirs(self.csv_folder, exist_ok=True)
            return self.csv_folder
        raise NotImplementedError(f"Unknown token '{token}'")

    def get_detector_output_folder(self, component: str, algorithm: str, token: str) -> Path:
        """
        Get detector-specific output folder.

        Args:
            component: Component name (e.g., 'body_joints')
            algorithm: Algorithm name (e.g., 'hrnetw48')
            token: Folder type - 'output', 'visualization', 'additional', 'run_config', 'result'
        """
        path = self._video_context.get_detector_folder(component, algorithm, token)
        os.makedirs(path, exist_ok=True)
        return path
