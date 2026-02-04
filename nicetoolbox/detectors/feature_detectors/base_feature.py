"""
Base class for Feature Detectors.
Feature detectors run computations in-process using method detector outputs.
"""

import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

from ...configs.schemas.detectors_algos_configs import FeatureDetectorRuntime
from ...utils.base_detectors import flatten_inference_config, input_map_to_string_keys
from ...utils.config import save_config
from ..base_detector import BaseDetector


class BaseFeature(BaseDetector):
    """
    Abstract base class for feature detectors.

    Feature detectors run in-process, computing derived features from
    method detector outputs.
    """

    def _setup_feature_detector(self, requires_out_folder: bool = True) -> None:
        """
        Setup feature detector configuration.
        """
        logging.info(
            f"Initializing feature detector {self.__class__.__name__} for '{self.algorithm}' "
            f"and components {self.components}."
        )

        # Build runtime config
        self.runtime = self._create_runtime(requires_out_folder)

        # Build and validate composed inference config
        self.inference_config = flatten_inference_config(self.static_config, self.runtime)

        # Save config for reproducibility
        folder = self.io.get_detector_output_folder(self.components[0], self.algorithm, "run_config")
        config_path = os.path.join(str(folder), "run_config.toml")
        save_config(self.inference_config, config_path)

        logging.info(f"Feature detector for component {self.components} and algorithm {self.algorithm} initialized.\n")

    def _create_runtime(self, requires_out_folder: bool) -> FeatureDetectorRuntime:
        """
        Create standard feature detector runtime configuration.

        Subclasses MUST override this if they have a specific RuntimeConfig
        that requires additional extension fields.
        """
        visualize = getattr(self.static_config, "visualize", False)

        # Resolve input paths from upstream detectors (tuple keys for internal usage)
        self.input_map = self._resolve_input_paths()
        # Convert to string keys for runtime config (TOML serialization)
        input_map_str = input_map_to_string_keys(self.input_map)

        return FeatureDetectorRuntime(
            result_folders=self.compute_result_folders(),
            out_folder=self.compute_output_folder(requires_out_folder),
            viz_folder=self.compute_viz_folder(visualize),
            algorithm=self.algorithm,
            visualize=visualize,
            subjects_descr=self.data.subjects_descr,
            input_map=input_map_str,
        )

    def _build_inference_config(self) -> Dict[str, Any]:
        """
        Build flattened config dictionary (Static + Runtime).
        """
        config = self.static_config.model_dump(by_alias=True)
        config.pop("RuntimeConfig", None)
        # Runtime fields take precedence
        config.update(self.runtime.model_dump())
        return config

    def _resolve_input_paths(self) -> Dict[Tuple[str, str], Path]:
        """
        Resolve input paths from upstream method detectors.

        Uses input_detector_names from static config to find upstream outputs.
        """
        input_map = {}
        input_detector_names = getattr(self.static_config, "input_detector_names", [])

        for component, algorithm in input_detector_names:
            input_path = self.io.get_detector_output_folder(component, algorithm, "result")
            input_map[(component, algorithm)] = input_path / f"{algorithm}.npz"

        return input_map

    def get_input_file(self, component: str, algorithm: str) -> Path:
        """
        Get the input file path for a specific upstream detector.

        Args:
            component: Component name (e.g., 'body_joints')
            algorithm: Algorithm name (e.g., 'hrnetw48')

        Returns:
            Path to the .npz result file
        """
        return self.input_map[(component, algorithm)]

    # -------------------------------------------------------------------------
    # BaseDetector Interface Implementation
    # -------------------------------------------------------------------------

    def run(self) -> Any:
        """
        Execute feature detector: compute() + post_compute().

        Returns computed data for visualization.
        """
        data = self.compute()
        return data

    @abstractmethod
    def compute(self) -> Any:
        """
        Compute the feature from method detector outputs.

        Returns:
            Computed feature data (passed to visualization and post_compute)
        """
        pass
