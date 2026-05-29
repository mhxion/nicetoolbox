"""
Base class for Feature Detectors.
Feature detectors run computations in-process using method detector outputs.
"""

import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple, final

from ...configs.schemas.detectors_instances_configs import FeatureDetectorRuntime
from ...utils.base_detectors import flatten_inference_config, input_map_to_string_keys
from ...utils.config import save_config
from ..base_detector import BaseDetector


class BaseFeature(BaseDetector):
    """
    Abstract base class for feature detectors.

    Feature detectors run in-process, computing derived features from
    method detector outputs.
    """

    requires_out_folder: bool = False

    @final
    def __init__(self, io, data, sequence_context, algorithm_instance: str):
        super().__init__(io, data, sequence_context, algorithm_instance)
        logging.info(
            f"Initializing feature detector {self.__class__.__name__} for instance '{self.algorithm_instance}' "
            f"and components {self.components}."
        )

        # Some common fields
        self.subjects_descr = self.data.subjects_descr
        self.input_map = self._resolve_input_paths()
        self.viz_folders = self.compute_viz_folders(self.visualize)
        self.out_folders = self.compute_output_folders(self.requires_out_folder)
        self.result_folders = self.compute_result_folders()

        # This hook is used to allow detector initialize custom fields
        self._initialize_detector()

        # Prepare infernce config
        self.runtime = self._build_runtime()
        self.inference_config = flatten_inference_config(self.detector_config, self.runtime)

        # Pre-map legacy single component out_folder and viz_folder for backward compatibility
        if len(self.components) == 1:
            comp = self.components[0]
            self.out_folder = self.out_folders.get(comp)
            self.viz_folder = self.viz_folders.get(comp)

        # Save config for reproducibility
        for comp in self.components:
            folder = self.io.get_detector_output_folder(comp, self.algorithm_instance, "run_config")
            config_path = os.path.join(str(folder), "run_config.toml")
            save_config(self.inference_config, config_path)

        logging.info(
            f"Feature detector for component {self.components} and instance {self.algorithm_instance} initialized.\n"
        )

    def _build_runtime(self) -> FeatureDetectorRuntime:
        """
        Create standard feature detector runtime configuration.

        Subclasses MUST override this if they have a specific RuntimeConfig
        that requires additional extension fields. Currently, this used purely for logging.
        """
        return FeatureDetectorRuntime(
            result_folders=self.result_folders,
            out_folders=self.out_folders,
            viz_folders=self.viz_folders,
            algorithm=self.algorithm_instance,
            visualize=self.visualize,
            subjects_descr=self.subjects_descr,
            input_map=input_map_to_string_keys(self.input_map),
        )

    def _build_inference_config(self) -> Dict[str, Any]:
        """
        Build flattened config dictionary (Static + Runtime).
        """
        config = self.detector_config.model_dump(by_alias=True)
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
        input_detector_names = getattr(self.detector_config, "input_detector_names", [])

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

    def _initialize_detector(self) -> None:
        pass

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
