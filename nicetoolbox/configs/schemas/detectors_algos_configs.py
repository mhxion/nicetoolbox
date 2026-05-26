# collection of all method and feature detectors configurations
# new detectors should be added and registered here

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nicetoolbox_core.input_recipes import InputRecipes

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
    and SequenceRuntimeConfig - NOT from static config files.
    """

    model_config = ConfigDict(extra="forbid")

    # Output paths
    result_folders: Dict[str, str]
    out_folders: Dict[str, str] = Field(default_factory=dict)
    viz_folders: Dict[str, str] = Field(default_factory=dict)

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

    # Input recipes for dataloaders in subprocess
    input_recipes: InputRecipes


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
    camera_names: List[str]
    env_name: str
    save_detector_images: bool
    save_detector_predictions: bool
    device: str
    filtered: bool
    window_length: int
    polyorder: int
    # python identifier cannot start with a number (using alias)
    visualize: bool


@detector_config("hrnetw48")
@detector_config("vitpose")
@detector_config("vitpose_huge")
@detector_config("rtmpose_l_aic")
@detector_config("rtmpose_l_wholebody")
@detector_config("rtmpose_m_mpii")
class MMPoseAlgorithmConfig(FrameworksMMPoseConfig):
    framework: str
    pose_config: str
    keypoint_mapping: str
    min_detection_confidence: float
    required_assets: Dict[str, str] = Field(default_factory=dict)
    # Optional dependency edges for topological sort (same shape as feature detectors).
    input_detector_names: Optional[List[List[str]]] = None

    # Nested runtime config class - extends base with MMPose-specific fields
    class RuntimeConfig(MethodDetectorRuntime):
        """MMPose-specific runtime fields."""

        prediction_folders: Dict[str, str]
        image_folders: Dict[str, str]
        keypoints_indices: Dict[str, List[int]]
        keypoints_description: Dict[str, List[str]]
        # Video / subprocess (explicit in run_config.toml — no guessed defaults in inference scripts).
        fps: int


@detector_config("motionbert")
class MotionbertAlgorithmConfig(FrameworksMMPoseConfig):
    """Static config for MotionBERT 3D lifting (2D NPZ input); kept separate from 2D MMPose algorithms."""

    framework: str
    keypoint_mapping: str
    min_detection_confidence: float
    # 3D lifter weights only; 2D detector assets are merged from input_detector_names at init.
    required_assets: Dict[str, str] = Field(default_factory=dict)
    # Must include exactly one upstream body_joints producer (NPZ path, 2D pose assets, etc.).
    input_detector_names: List[List[str]]
    # Optional MMPose 3D frame-export layout (serialized to run_config; read with strict keys in subprocess).
    mmpose_3d_nice_multi_layout: bool = True
    pelvis_sep_scale: float = 1.0

    @model_validator(mode="after")
    def _validate_body_joints_upstream(self) -> "MotionbertAlgorithmConfig":
        algs = [e[1] for e in self.input_detector_names if len(e) == 2 and e[0] == "body_joints"]
        if not algs:
            raise ValueError(
                "motionbert requires input_detector_names to include "
                "['body_joints', '<upstream_2d_algorithm>'], e.g. [['body_joints', 'vitpose_huge']]."
            )
        if len(set(algs)) != 1:
            raise ValueError(f"motionbert expects a single body_joints upstream algorithm; got {sorted(set(algs))!r}.")
        for key in ("pose3d_config_file", "pose3d_checkpoint"):
            if key not in self.required_assets:
                raise ValueError(f"motionbert.required_assets must include {key!r}")
        return self

    class RuntimeConfig(MMPoseAlgorithmConfig.RuntimeConfig):
        """Runtime fields for MotionBERT (NPZ path, merged 2D+3D assets, derived 2D lifter inputs)."""

        pose_config: str
        motionbert_2d_pose_det_dataset: str
        motionbert_2d_coco_body_indices: List[int]
        motionbert_2d_keypoints_npz: str
        required_assets: Dict[str, str]


@detector_config("spiga")
class SpigaConfig(BaseModel):
    camera_names: List[str]
    env_name: str
    log_frame_idx_interval: int
    batch_size: int
    visualize: bool
    dataset_name: str

    required_assets: Dict[str, str] = Field(default_factory=dict)

    class RuntimeConfig(MethodDetectorRuntime):
        """SPIGA-specific runtime fields."""

        face_landmarks_description: List[str]


@detector_config("py_feat")
class PyFeatConfig(BaseModel):
    camera_names: List[str]
    env_name: str
    log_frame_idx_interval: int
    batch_size: int
    visualize: bool
    required_assets: Dict[str, str] = Field(default_factory=dict)


@detector_config("multiview_eth_xgaze")
class MultiViewETHXGazeConfig(BaseModel):
    camera_names: List[str]
    env_name: str
    log_frame_idx_interval: int
    filtered: bool
    window_length: int
    polyorder: int
    visualize: bool
    required_assets: Dict[str, str] = Field(default_factory=dict)


@detector_config("whisperx")
class WhisperXConfig(BaseModel):
    env_name: str
    visualize: bool
    track_names: List[str]

    model_size: str
    compute_type: str
    batch_size: int
    language: Optional[str]
    vad_onset: float
    vad_offset: float
    alignment_model_name: str
    hf_weights_cache_dir: str = Field(
        default="<assets>/whisperx",
        description="Directory for WhisperX / Hugging Face weight caches (created at inference if missing).",
    )

    class RuntimeConfig(MethodDetectorRuntime):
        """Runtime including HF token for pyannote / Hub (resolved in detector init)."""

        hf_token: str


@detector_config("sam_3d_body")
class Sam3dBodyConfig(BaseModel):
    """SAM 3D Body (Hugging Face weights; set hugging_face_token in machine_specific_paths.toml)."""

    camera_names: List[str]
    env_name: str
    device: str
    visualize: bool

    hf_repo_id: str = "facebook/sam-3d-body-dinov3"
    sam3d_repo_path: str = ""
    detector_name: str = "vitdet"
    segmentor_name: str = "sam2"
    fov_name: str = "moge2"
    inference_type: str = "body"
    save_vertices: bool = True
    visualize_mesh: bool = True
    visualize_mesh_interactive: bool = False
    interactive_mesh_frame_stride: int = 1
    interactive_mesh_camera_index: int = 0
    interactive_mesh_prefer_world: bool = True
    interactive_mesh_subject_spacing: Optional[float] = None
    interactive_mesh_line_width: int = 1
    # Keep interactive HTML browser-sized (uncapped wire × frames can exceed ~500 MB).
    interactive_mesh_max_edges: int = 12_000
    interactive_mesh_max_frames: int = 48
    interactive_mesh_plotlyjs_cdn: bool = False
    bbox_sort_left_to_right: bool = True
    temporal_smooth: bool = True
    smooth_window_length: int = 7
    smooth_polyorder: int = 2
    world_align_keypoints_3d: bool = True
    cross_view_consistency: bool = True
    stereo_triangulation_body_joints: bool = True
    triangulation_min_detection_confidence: float = 0.6
    keypoint_mapping: str = "sam_3d_body_mhr"
    required_assets: Dict[str, str] = Field(default_factory=dict)


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
