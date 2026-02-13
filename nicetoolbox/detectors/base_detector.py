"""
Base class for all detectors (method and feature).
Provides common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..configs.video_runtime_config import VideoRuntimeConfig
from .data import VideoData
from .in_out import VideoIO


class BaseDetector(ABC):
    """
    Abstract base class for ALL detectors.

    Defines the common interface that both method and feature detectors implement.
    This enables a unified detector loop in main.py.
    """

    # Instance attributes set during initialization
    data: VideoData
    io: VideoIO
    video_context: VideoRuntimeConfig
    detector_config: BaseModel

    # Class attributes to be defined by subclasses
    inference_config: Any
    components: List[str]
    algorithm: str

    # Additional attributes
    visualize: bool

    def __init__(self, io: VideoIO, data: VideoData, video_context: VideoRuntimeConfig) -> None:
        """
        Initialize base detector with references.

        Subclasses should call super().__init__() and set inference_config.
        """
        self.io = io
        self.data = data
        self.video_context = video_context
        self.detector_config = video_context.get_detector_config(self.algorithm)
        self.visualize = getattr(self.detector_config, "visualize", False)

    @abstractmethod
    def run(self) -> Optional[Any]:
        """
        Execute the detector's main computation.

        For method detectors: runs subprocess inference + post_inference()
        For feature detectors: runs compute()

        Returns:
            Optional data for visualization (feature detectors currently return computed data)
        """
        pass

    @abstractmethod
    def visualization(self, data: Any) -> None:
        """
        Visualize detector output.

        Args:
            data: Output from run() or external data source
        """
        pass

    @property
    @abstractmethod
    def components(self) -> List[str]:
        """Components this detector produces."""
        raise NotImplementedError

    @property
    @abstractmethod
    def algorithm(self) -> str:
        """Algorithm name for this detector."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Shared Helper Methods
    # -------------------------------------------------------------------------

    @property
    def predictions_mapping(self):
        """Access predictions mapping from runtime config."""
        return self.video_context.predictions_mapping

    def compute_result_folders(self) -> Dict[str, str]:
        """Compute result folders for all components."""
        return {
            comp: str(self.io.get_detector_output_folder(comp, self.algorithm, "result")) for comp in self.components
        }

    def compute_output_folder(self, requires_out_folder: bool) -> Optional[str]:
        """Compute main output folder."""
        if requires_out_folder:
            return str(self.io.get_detector_output_folder(self.components[0], self.algorithm, "output"))
        return None

    def compute_viz_folder(self, visualize: bool) -> Optional[str]:
        """Compute visualization folder."""
        if visualize:
            return str(self.io.get_detector_output_folder(self.components[0], self.algorithm, "visualization"))
        return None

    def __str__(self) -> str:
        config_name = type(self.inference_config).__name__
        return f"Detector: {self.algorithm}\n  Components: {self.components}\n  Config: {config_name}\n"
