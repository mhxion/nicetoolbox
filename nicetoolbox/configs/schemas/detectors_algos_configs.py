# collection of all method and feature detectors configurations
# new detectors should be added and registered here

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field

from ..models.models_registry import ModelsRegistry

# registries for detectors and frameworks
DETECTORS_REGISTRY = ModelsRegistry()
FRAMEWORKS_REGISTRY = ModelsRegistry()
detector_config = DETECTORS_REGISTRY.register
framework_config = FRAMEWORKS_REGISTRY.register


# =============================================================================
# Common Runtime Fields
# =============================================================================


class BaseDetectorRuntime(BaseModel):
    """
    Common runtime fields for ALL detectors (method and feature).

    These fields are computed during detector initialization from IO, Data,
    and VideoRuntimeConfig - NOT from static config files.
    """

    model_config = ConfigDict(extra="forbid")

    # Output paths
    result_folders: Dict[str, str]
    out_folder: Optional[str] = None
    viz_folder: Optional[str] = None

    # Algorithm identity
    algorithm: str

    # Run config flag
    visualize: bool

    # Data context
    subjects_descr: List[str]


class MethodDetectorRuntime(BaseDetectorRuntime):
    """
    Runtime fields specific to method detectors.
    Extends BaseDetectorRuntime with subprocess and inference requirements.
    """

    # Logging (needed for subprocess)
    log_file: str
    log_level: str

    # Data context for inference
    calibration: Optional[Dict[str, Any]] = None
    cam_sees_subjects: Dict[str, List[int]]

    # Input recipe for dataloader in subprocess
    input_recipe: dict[str, Any]


class FeatureDetectorRuntime(BaseDetectorRuntime):
    """
    Runtime fields specific to feature detectors.
    Extends BaseDetectorRuntime with input path requirements.
    """

    # Input paths (from upstream method detectors)
    input_map: dict[str, str]  # {(component, algorithm): path}
    # We can add common feature runtime fields later (if we need them (e.g. for audio pipeline))


# ================================================
#                 METHOD DETECTORS
# ================================================


@framework_config("mmpose")
class FrameworksMMPoseConfig(BaseModel):
    input_data_format: str
    camera_names: List[str]
    env_name: str
    save_detector_images: bool
    save_detector_predictions: bool
    device: str
    filtered: bool
    window_length: int
    polyorder: int
    # python identifier cannot start with a number (using alias)
    results_3d: bool = Field(alias="3d_results")
    visualize: bool


@detector_config("hrnetw48")
@detector_config("vitpose")
class MMPoseAlgorithmConfig(FrameworksMMPoseConfig):
    framework: str
    pose_config: str
    pose_checkpoint: str
    detection_config: str
    detection_checkpoint: str
    keypoint_mapping: str
    min_detection_confidence: float

    # Nested runtime config class - extends base with MMPose-specific fields
    class RuntimeConfig(MethodDetectorRuntime):
        """MMPose-specific runtime fields."""

        prediction_folders: Dict[str, str]
        image_folders: Dict[str, str]
        keypoints_indices: Dict[str, List[int]]
        keypoints_description: Dict[str, List[str]]


@detector_config("spiga")
class SpigaConfig(BaseModel):
    input_data_format: str
    camera_names: List[str]
    env_name: str
    log_frame_idx_interval: int
    batch_size: int
    visualize: bool
    # model_config is reserved variable in pydantic model
    model_configuration: str = Field(alias="model_config")

    class RuntimeConfig(MethodDetectorRuntime):
        """SPIGA-specific runtime fields."""

        face_landmarks_description: List[str]


@detector_config("py_feat")
class PyFeatConfig(BaseModel):
    input_data_format: str
    camera_names: List[str]
    env_name: str
    log_frame_idx_interval: int
    batch_size: int
    visualize: bool


@detector_config("multiview_eth_xgaze")
class MultiViewETHXGazeConfig(BaseModel):
    input_data_format: str
    camera_names: List[str]
    env_name: str
    shape_predictor_filename: str
    face_model_filename: str
    pretrained_model_filename: str
    face_detector_filename: str
    log_frame_idx_interval: int
    filtered: bool
    window_length: int
    polyorder: int
    visualize: bool


# === Add Method detectors HERE ===


# ================================================
#                 FEATURE DETECTORS
# ================================================


@detector_config("gaze_distance")
class GazeDistanceConfig(BaseModel):
    input_detector_names: List[List[str]]
    keypoint_mapping: str
    threshold_look_at: float
    visualize: bool


@detector_config("velocity_body")
class VelocityConfig(BaseModel):
    input_detector_names: List[List[str]]
    visualize: bool


@detector_config("body_angle")
class BodyAngleConfig(BaseModel):
    input_detector_names: List[List[str]]
    used_keypoints: List[List[str]]
    visualize: bool


@detector_config("body_distance")
class BodyDistanceConfig(BaseModel):
    input_detector_names: List[List[str]]
    used_keypoints: List[str]
    visualize: bool


@detector_config("gaze_fusion")
class GazeFusionConfig(BaseModel):
    input_detector_names: List[List[str]]
    fusion_method: str
    filtered: bool
    window_length: int
    polyorder: int
    ensemble_enabled: bool
    visualize: bool


# === Add Feature detectors HERE ===


# =============================================================================
# Helper Functions
# =============================================================================


def get_runtime_config_class(static_config: BaseModel) -> Type[MethodDetectorRuntime]:
    """
    Get the RuntimeConfig class for a detector's static config.

    Returns the nested RuntimeConfig if defined, otherwise MethodDetectorRuntime.
    """
    return getattr(static_config.__class__, "RuntimeConfig", MethodDetectorRuntime)


def has_extended_runtime(static_config: BaseModel) -> bool:
    """Check if detector has extended runtime fields."""
    return hasattr(static_config.__class__, "RuntimeConfig")
