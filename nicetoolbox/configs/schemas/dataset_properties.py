from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, PrivateAttr, model_validator

from ..models.dict_model import DictModel


class AudioTrackConfig(BaseModel):
    """
    Configuration for a single audio track.

    A track is either:
    - Embedded: extracted from a camera's video file (has `camera` field)
    - Standalone: loaded from a separate audio file (has `path` field)

    Exactly one of `camera` or `path` must be set.
    """

    # Source: one of these must be set
    camera: Optional[str] = None  # Camera id to extract audio from. Mutually exclusive with path.
    path: Optional[Path] = None  # Path to standalone audio file. Mutually exclusive with camera.

    # Audio stream index in the source file (0-based). Relevant for multi-stream video files.
    stream: NonNegativeInt = 0
    # Audio channel index in the source file (0-based). Relevant for stereo, surround sound channel layouts
    # None will pass all channels to the audio detectors and let them to decide how to process multiple channels
    channel: Optional[NonNegativeInt] = None

    # Which subjects this track can hear. Indices into `subjects_descr`. Must be non-empty.
    hears_subjects: List[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source(self):
        """Ensure exactly one of camera or path is set."""
        has_camera = self.camera is not None and self.camera != ""
        has_path = self.path is not None

        if has_camera and has_path:
            raise ValueError("Audio track must have either 'camera' or 'path', not both.")
        if not has_camera and not has_path:
            raise ValueError("Audio track must have either 'camera' or 'path'.")
        return self

    @property
    def is_embedded(self) -> bool:
        """True if this track is extracted from a video file."""
        return self.camera is not None and self.camera != ""

    @property
    def is_standalone(self) -> bool:
        """True if this track is a standalone audio file."""
        return self.path is not None


class DatasetAudio(BaseModel):
    """
    Configuration for dataset audio modality.
    """

    tracks: Optional[Dict[str, AudioTrackConfig]] = Field(default_factory=dict)


class AnnotationComponentConfig(BaseModel):
    """
    Annotation source configuration for a single component.
    """

    path: Path


class DatasetAnnotation(BaseModel):
    """
    Optional per-component annotation paths used by evaluation input blocks.
    """

    components: Dict[str, AnnotationComponentConfig] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    """
    Configuration schema for a single dataset.
    Contains metadata and paths required for processing and evaluation.
    """

    session_IDs: List[str]
    sequence_IDs: List[str]

    cam_front: str = ""
    cam_top: str = ""
    cam_face1: str = ""
    cam_face2: str = ""

    subjects_descr: List[str]
    cam_sees_subjects: Optional[Dict[str, List[int]]] = Field(default_factory=dict)

    data_input_folder: Path
    path_to_calibrations: Optional[Path] = None

    start_frame_index: NonNegativeInt
    fps: PositiveInt

    annotation: DatasetAnnotation = Field(default_factory=DatasetAnnotation)
    audio: DatasetAudio = Field(default_factory=DatasetAudio)

    # Runtime fields
    _dataset_name: str = PrivateAttr()


# Top-level config in dataset_properties.toml
class DatasetProperties(DictModel[str, DatasetConfig]):
    """
    Dictionary of dataset configurations, keyed by unique dataset name.
    Users can define any custom datasets.
    """

    def model_post_init(self, _):
        # injecting key into each DatasetConfig
        for name, ds in self.root.items():
            ds._dataset_name = name
