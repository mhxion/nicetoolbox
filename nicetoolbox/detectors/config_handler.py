""" """

from pathlib import Path
from typing import Any, Dict, Generator, List

from ..configs.config_loader import ConfigLoader
from ..configs.placeholders import PLACEHOLDERS_TYPE
from ..configs.schemas.dataset_properties import DatasetProperties
from ..configs.schemas.detectors_config import DetectorsConfig
from ..configs.schemas.detectors_run_file import DetectorsRunFile, LoggingLevelEnum
from ..configs.schemas.experiment_config import CodeConfig, DetectorsExperimentConfig
from ..configs.schemas.machine_specific_paths import MachineSpecificConfig
from ..configs.schemas.predictions_mapping import PredictionsMappingConfig
from ..configs.utils import default_auto_placeholders, default_runtime_placeholders, model_to_dict
from ..configs.video_runtime_config import VideoRuntimeConfig
from ..utils.config import save_config


def flatten_list(input_list) -> list[Any]:
    if isinstance(input_list, str):
        return [input_list]
    if isinstance(input_list, int):
        return [input_list]
    if isinstance(input_list, list):
        output_list = []
        for item in input_list:
            output_list += flatten_list(item)
        return output_list
    raise NotImplementedError


class Configuration:
    """
    Handles loading and resolving all configurations required for detectors pipeline. This includes:
    - machine specifics
    - run configuration file
    - detectors configuration
    - dataset properties

    Further provides a config factory that produces frozen and resolved runtime configs per video context
    """

    # (1) Config Loader with Placeholder Resolution
    cfg_loader: ConfigLoader
    auto_placeholders: dict[str, PLACEHOLDERS_TYPE]
    runtime_placeholders: set[str]

    # (2) Loaded Pydantic Configurations
    machine_specific_config: MachineSpecificConfig
    run_config: DetectorsRunFile
    detectors_config: DetectorsConfig
    dataset_properties: DatasetProperties

    def __init__(self, run_config_file, machine_specifics_file):
        """
        Load all static configuration files.

        Args:
            run_config_file (str): Path to detectors_run_file.toml
            machine_specifics_file (str): Path to machine_specific_paths.toml
        """
        # init config loader
        self.auto_placeholders = default_auto_placeholders()
        self.runtime_placeholders = default_runtime_placeholders()
        self.cfg_loader = ConfigLoader(self.auto_placeholders, self.runtime_placeholders)
        # code config
        self.code_config = CodeConfig(**self.auto_placeholders)
        # machine specific
        self.machine_specific_config = self.cfg_loader.load_config(machine_specifics_file, MachineSpecificConfig)
        self.cfg_loader.extend_global_ctx(self.machine_specific_config)
        # run file
        self.run_config = self.cfg_loader.load_config(run_config_file, DetectorsRunFile)
        self.cfg_loader.extend_global_ctx(self.run_config.io)
        # detectors config
        detectors_config_file = str(self.run_config.io.detectors_config)
        self.detectors_config = self.cfg_loader.load_config(detectors_config_file, DetectorsConfig)
        # dataset config
        dataset_properties_file = str(self.run_config.io.dataset_properties)
        self.dataset_properties = self.cfg_loader.load_config(dataset_properties_file, DatasetProperties)
        # predictions mapping
        predictions_mapping_file = str(self.run_config.io.predictions_mapping)
        self.predictions_mapping = self.cfg_loader.load_config(predictions_mapping_file, PredictionsMappingConfig)

    # -------------------------------------------------------------------------
    # Factory Method for Video Runtime Configurations
    # -------------------------------------------------------------------------

    def iter_video_contexts(self) -> Generator[VideoRuntimeConfig, None, None]:
        """
        Iterate over all videos and yield frozen runtime configurations.

        Each yielded VideoRuntimeConfig is fully resolved and immutable.
        It should be discarded after the video is processed.

        Yields:
            VideoRuntimeConfig for each video defined in the run configuration
        """
        for dataset_name, videos_run_config in self.run_config.run.items():
            # Build component -> algorithms mapping (Filtered via selection in run_config file)
            component_mapping = {
                comp: self.run_config.component_algorithm_mapping[comp] for comp in videos_run_config.components
            }

            # Get all selected algorithms for this dataset
            selected_algorithms: list[str] = list(set(flatten_list(list(component_mapping.values()))))

            # Get dataset properties
            dataset_props = self.dataset_properties[dataset_name]

            for video in videos_run_config.videos:
                yield self._create_video_runtime_config(
                    dataset_name=str(dataset_name),
                    video=video,
                    dataset_props=dataset_props,
                    components=videos_run_config.components,
                    component_mapping=component_mapping,
                    selected_algorithms=selected_algorithms,
                )

    def _create_video_runtime_config(
        self,
        dataset_name: str,
        video,
        dataset_props,
        components: List[str],
        component_mapping: Dict[str, List[str]],
        selected_algorithms: List[str],
    ) -> VideoRuntimeConfig:
        """
        Create a fully resolved, frozen VideoRuntimeConfig.

        All placeholders are resolved before constructing the frozen model.
        """
        # 1. Expand dependencies to include upstream method detectors (run features without method detectors)
        expanded_algorithms = self._expand_dependencies(selected_algorithms)

        # 2. Compute camera names from all resolved detector configs required for this run
        all_camera_names = self._compute_camera_names(expanded_algorithms)

        # 3. Construct frozen model with all resolved values
        runtime_config = VideoRuntimeConfig(
            log_level=self.log_level,
            log_file=self.log_file,
            dataset_name=dataset_name,
            video_config=video,
            dataset_properties=dataset_props,
            io=self.run_config.io,
            machine=self.machine_specific_config,
            detectors_config=self.detectors_config,
            predictions_mapping=self.predictions_mapping,
            components=components,
            component_mapping=component_mapping,
            all_camera_names=all_camera_names,
        )
        # 4. Build runtime context for this video
        runtime_ctx = {
            "cur_dataset_name": dataset_name,
            "cur_session_ID": video.session_ID,
            "cur_sequence_ID": video.sequence_ID,
            "cur_video_start": video.video_start,
            "cur_video_length": video.video_length,
            "cur_cam_face1": dataset_props.cam_face1,
            "cur_cam_face2": dataset_props.cam_face2,
            "cur_cam_top": dataset_props.cam_top,
            "cur_cam_front": dataset_props.cam_front,
        }
        # 5. RESOLVE
        resolved_runtime = self.cfg_loader.resolve(runtime_config, runtime_ctx, ignore_auto_and_global=True)
        return resolved_runtime

    def _expand_dependencies(self, algorithm_names: List[str]) -> List[str]:
        """
        Recursively expand algorithm list to include upstream method detectors.

        Feature detectors depend on method detectors (e.g., velocity_body needs hrnetw48).
        This ensures we include cameras from all required algorithms.

        Args:
            algorithm_names: Initially selected algorithms

        Returns:
            Expanded list including all dependencies
        """
        expanded = set(algorithm_names)

        for algo_name in algorithm_names:
            config = self.detectors_config.algorithms.get(algo_name)
            if config is None:
                continue

            # Check if this is a feature detector with dependencies
            input_detector_names = getattr(config, "input_detector_names", None)
            if input_detector_names:
                # Format: [['component', 'algorithm'], ...]
                dependencies = [pair[1] for pair in input_detector_names]
                # Recursively expand
                expanded.update(self._expand_dependencies(dependencies))

        return list(expanded)

    def _compute_camera_names(self, algorithm_names: List[str]) -> List[str]:
        """
        Extract all unique camera names from resolved detector configs.

        Args:
            detectors_configs: Pre-resolved detector configurations (includes dependencies)

        Returns:
            Sorted list of unique camera names
        """
        cameras = set()
        for algorithm in algorithm_names:
            config = self.detectors_config.algorithms[algorithm]
            camera_names = getattr(config, "camera_names", None)
            if camera_names:
                # Filter out empty strings that might result from placeholder resolution
                cameras.update(c for c in camera_names if c)

        cameras -= set([""])  # Remove empty strings when camera parsing fails

        return sorted(list(cameras))

    # -------------------------------------------------------------------------
    # Static Queries (don't depend on runtime context)
    # -------------------------------------------------------------------------

    def get_all_detector_names(self) -> list[str]:
        """
        Returns all detector names defined in the detectors configuration.
        """
        return list(self.detectors_config.algorithms.keys())

    def save_experiment_config(self, output_folder) -> None:
        # save all experiment configurations
        config = DetectorsExperimentConfig(
            run_config=self.run_config,
            dataset_config=self.dataset_properties,
            detector_config=self.detectors_config,
            machine_specific_config=self.machine_specific_config,
            code_config=self.code_config,
        )
        save_config(model_to_dict(config), output_folder / f"config_{self.code_config.time}.toml")

    @property
    def visualize(self) -> bool:
        return self.run_config.visualize

    @property
    def save_csv(self) -> bool:
        return self.run_config.save_csv

    @property
    def error_level(self) -> str:
        return self.run_config.error_level

    @property
    def log_level(self) -> LoggingLevelEnum:
        return self.run_config.log_level

    @property
    def log_file(self) -> Path:
        return self.run_config.io.out_folder / "nicetoolbox.log"
