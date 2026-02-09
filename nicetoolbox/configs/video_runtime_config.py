"""
Runtime configuration for processing a single video.
Created by Configuration factory, discarded after video processing.
"""

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from .schemas.dataset_properties import DatasetConfig
from .schemas.detectors_config import DetectorsConfig
from .schemas.detectors_run_file import DetectorsRunIO, LoggingLevelEnum, RunConfigVideo
from .schemas.machine_specific_paths import MachineSpecificConfig
from .schemas.predictions_mapping import PredictionsMappingConfig


class VideoRuntimeConfig(BaseModel):
    """
    Immutable configuration context for processing a single video.

    Created by Configuration.iter_video_contexts(), holds all resolved
    configuration needed for one video. Discarded after processing.

    All placeholders (except <cur_component_name> and <cur_algorithm_name>
    in IO paths) are fully resolved at construction time.

    Note: frozen=True only prevents reassignment of top-level attributes.
    Nested models are mutable but should be treated as immutable by convention.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # -------------------------------------------------------------------------
    # Core Configuration (all resolved)
    # -------------------------------------------------------------------------
    log_level: LoggingLevelEnum
    log_file: Path

    dataset_name: str

    video_config: RunConfigVideo
    dataset_properties: DatasetConfig
    io: DetectorsRunIO  # Resolved (except component/algorithm placeholders)
    machine: MachineSpecificConfig
    detectors_config: DetectorsConfig
    predictions_mapping: PredictionsMappingConfig

    # Component/algorithm selection for this video
    components: List[str]
    component_mapping: Dict[str, List[str]]

    # Pre-referenced cameras required for current video (based on upstream dependencies too)
    all_camera_names: List[str]

    # -------------------------------------------------------------------------
    # Convenience Properties
    # -------------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self.video_config.session_ID

    @property
    def sequence_id(self) -> str:
        return self.video_config.sequence_ID

    @property
    def video_start(self) -> int:
        return self.video_config.video_start

    @property
    def video_length(self) -> int:
        return self.video_config.video_length

    @property
    def fps(self) -> int:
        return self.dataset_properties.fps

    @property
    def subjects_descr(self) -> List[str]:
        return self.dataset_properties.subjects_descr

    @property
    def calibration_path(self) -> Optional[Path]:
        path = self.dataset_properties.path_to_calibrations
        return Path(path) if path else None

    @property
    def data_source_folder(self) -> Path:
        return Path(self.dataset_properties.data_input_folder)

    @property
    def all_selected_algorithms(self) -> List[str]:
        """Flat list of all algorithms to run for this video."""
        algorithms = []
        for algo_list in self.component_mapping.values():
            algorithms.extend(algo_list)
        return list(set(algorithms))

    # -------------------------------------------------------------------------
    # Config Access (no resolution needed - already resolved)
    # -------------------------------------------------------------------------

    def get_detector_config(self, algorithm_name: str) -> BaseModel:
        """
        Get the pre-resolved configuration for a detector.

        Args:
            algorithm_name: Name of the algorithm (e.g., 'hrnetw48', 'velocity_body')

        Returns:
            Resolved configuration dict ready for detector initialization.
            Includes the injected 'visualize' flag.

        Raises:
            KeyError: If algorithm was not in the selected algorithms for this video
        """
        if algorithm_name not in self.detectors_config.algorithms:
            raise KeyError(
                f"Algorithm '{algorithm_name}' not found. "
                f"Available: {list(self.detectors_config.algorithms.keys())}"
            )
        return self.detectors_config.algorithms[algorithm_name]

    # -------------------------------------------------------------------------
    # IO Path Helpers (handle remaining component/algorithm placeholders)
    # -------------------------------------------------------------------------

    def get_detector_folder(self, component: str, algorithm: str, folder_type: str) -> Path:
        """
        Get resolved detector-specific folder path.

        The IO paths contain <cur_component_name> and <cur_algorithm_name>
        placeholders that are resolved here based on the specific detector.

        Args:
            component: Component name (e.g., 'body_joints')
            algorithm: Algorithm name (e.g., 'hrnetw48')
            folder_type: One of 'output', 'visualization', 'additional', 'run_config', 'result'

        Returns:
            Resolved Path to the requested folder
        """
        template_map = {
            "output": self.io.detector_out_folder,
            "visualization": self.io.detector_visualization_folder,
            "additional": self.io.detector_additional_output_folder,
            "run_config": self.io.detector_run_config_path,
            "result": self.io.detector_final_result_folder,
        }

        if folder_type not in template_map:
            raise ValueError(f"Unknown folder_type '{folder_type}'. Valid: {list(template_map.keys())}")

        path_str = str(template_map[folder_type])
        resolved = path_str.replace("<cur_component_name>", component).replace("<cur_algorithm_name>", algorithm)

        return Path(resolved)
