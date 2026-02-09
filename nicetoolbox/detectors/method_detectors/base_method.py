"""
Base class for Method Detectors.
Method detectors run external inference scripts in separate virtual environments.
"""

import json
import logging
import os
import subprocess
from abc import abstractmethod
from pathlib import Path
from typing import final

from nicetoolbox_core.entrypoint import SubprocessError

from ...configs.schemas.detectors_algos_configs import MethodDetectorRuntime
from ...configs.video_runtime_config import VideoRuntimeConfig
from ...utils.base_detectors import flatten_inference_config
from ...utils.config import save_config
from ...utils.system import detect_os_type
from ..base_detector import BaseDetector
from ..data import VideoData
from ..in_out import VideoIO


class BaseMethod(BaseDetector):
    """
    Abstract base class for method detectors.

    Method detectors run inference in external virtual environments via subprocess.
    """

    runtime: MethodDetectorRuntime

    # subprocess settings
    os_type: str
    conda_path: Path
    venv: str
    env_name: str
    script_path: str

    # method specific settings
    visualize: bool
    requires_out_folder: bool
    out_folder: str
    result_folders: dict[str, str]
    viz_folder: str
    config_path: Path

    @final
    def __init__(self, io: VideoIO, data: VideoData, video_context: VideoRuntimeConfig) -> None:
        """
        Initialize base method detector with references.
        """
        # (1) Call BaseDetector __init__()
        super().__init__(io, data, video_context)

        logging.info(
            f"Initializing method detector {self.__class__.__name__} with algorithm '{self.algorithm}' "
            f"and component(s) {self.components}."
        )

        # (2) Setup subprocess execution settings
        self._setup_subprocess_settings()

        # (3) Setup method detector (builds runtime, validates, saves config)
        self.visualize = getattr(self.detector_config, "visualize", False)
        self.requires_out_folder = getattr(self.detector_config, "visualize", False)
        self.out_folder = self.compute_output_folder(self.requires_out_folder)
        self.result_folders = self.compute_result_folders()
        self.viz_folder = self.compute_viz_folder(self.visualize)

        self.runtime = self._initialize_detector()  # This will be overloaded by child detectors!

        # (4) Flatten config for subprocess
        inference_config = flatten_inference_config(self.detector_config, self.runtime)

        # (5) Save config for subprocess
        folder = self.io.get_detector_output_folder(self.components[0], self.algorithm, "run_config")
        self.config_path = folder / "run_config.toml"
        save_config(inference_config, self.config_path)

        logging.info("Inference preparation completed.\n")

    def _initialize_detector(self) -> MethodDetectorRuntime:
        """
        Create runtime configuration.

        Override in subclasses that need extended runtime fields.
        The override should:
        1. Call super()._create_runtime() to get base runtime
        2. Return an instance of the detector's RuntimeConfig class with additional fields

        Returns:
            MethodDetectorRuntime or subclass with detector-specific fields
        """
        return MethodDetectorRuntime(
            result_folders=self.result_folders,
            out_folder=self.out_folder,
            viz_folder=self.viz_folder,
            algorithm=self.algorithm,
            visualize=self.visualize,
            subjects_descr=self.data.subjects_descr,
            log_file=str(self.video_context.log_file),
            log_level=self.video_context.log_level,
            calibration=self.data.calibration,
            cam_sees_subjects=self.data.cam_sees_subjects,
            input_recipe=self.data.get_input_recipe()["input_recipe"],
        )

    def _setup_subprocess_settings(self) -> None:
        """Setup OS-specific subprocess execution settings."""
        self.os_type = detect_os_type()
        self.conda_path = self.io.get_conda_path()

        env_name = getattr(self.detector_config, "env_name", "venv:nicetoolbox")  # Default to nicetoolbox env
        self.venv, self.env_name = env_name.split(":")

        framework = getattr(self.detector_config, "framework", self.algorithm)
        self.script_path = self.io.get_inference_path(self.components[0], framework)

        if self.venv == "venv":
            self.venv_path = self.io.get_venv_path(framework, self.env_name)

    # -------------------------------------------------------------------------
    # BaseDetector Interface Implementation
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute method detector: run subprocess inference + post_inference.

        Returns None - visualization uses external data.
        """
        self._run_inference()

    def _run_inference(self) -> None:
        """Run the inference subprocess."""
        logging.info(f"INFERENCE: Launching {self.algorithm} subprocess...")

        command = self._create_command()

        if self.os_type == "windows":
            cmd_result = subprocess.run(command, capture_output=True, text=True, shell=True, check=False)
        else:
            cmd_result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                shell=True,
                executable="/bin/bash",
                check=False,
            )

        if cmd_result.returncode == 0:
            logging.info(f"INFERENCE: {self.algorithm} finished successfully (Exit 0).")
            self.post_inference()
            return

        logging.error(f"INFERENCE: {self.algorithm} subprocess failed (Exit 1).")
        self._handle_subprocess_error(cmd_result)

    def _create_command(self) -> str:
        """Create the shell command to run inference."""
        if self.venv == "conda":
            if self.os_type == "windows":
                command = (
                    f"deactivate && "
                    f'cmd "/c conda activate {self.env_name} && '
                    f'python {self.script_path} {self.config_path}"'
                )
            else:
                conda_path = os.path.join(self.conda_path, "bin/activate")
                python_path = os.path.join(self.conda_path, "envs", self.env_name, "bin/python")
                command = (
                    f"conda init bash && source ~/.bashrc && "
                    f"{conda_path} {self.env_name} && "
                    f"{python_path} {self.script_path} {self.config_path}"
                )
        elif self.venv == "venv":
            if self.os_type == "windows":
                command = f'cmd "/c {self.venv_path} && ' f'python {self.script_path} {self.config_path}"'
            else:
                command = f"source {self.venv_path} && " f"python {self.script_path} {self.config_path}"
        else:
            raise ValueError(f"Unknown venv type '{self.venv}'. Expected 'conda' or 'venv'.")
        return command

    def _handle_subprocess_error(self, cmd_result) -> None:
        """Handle subprocess failure by checking for error.json."""
        error_file = self.config_path.parent / "error.json"

        if error_file.exists():
            try:
                with open(error_file) as file:
                    remote_exc_json = json.load(file)
                remote_exc = SubprocessError(**remote_exc_json)
                raise RuntimeError(
                    f"Subprocess raised {remote_exc.exception_type}: {remote_exc.message}\n\n"
                    f"--- Remote Traceback ---\n\n{remote_exc.traceback}"
                )
            except (json.JSONDecodeError, KeyError) as err:
                raise RuntimeError(
                    f"Subprocess failed and error report is corrupt: {err}\n" f"Stderr: {cmd_result.stderr}"
                ) from err
        else:
            raise RuntimeError(f"Subprocess Hard Crash (No Error Report):\n{cmd_result.stderr}")

    @abstractmethod
    def post_inference(self) -> None:
        """Post-processing after inference completes."""
        pass
