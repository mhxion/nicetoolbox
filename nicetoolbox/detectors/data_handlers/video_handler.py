"""
Video/frame data handler for the NICE Toolbox.

Handles frame extraction from video files and preparation of image sequences.
Also owns camera calibration loading since calibration is video-specific.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from nicetoolbox_core.input_recipes import VideoInputRecipe

from ...configs.models.video_timestamp import timestamp_to_frame_index
from ...configs.video_runtime_config import SequenceRuntimeConfig
from ...utils import video as vid
from ...utils.logging_utils import log_with_underscore
from ..in_out import SequenceIO
from .handler import BaseModalityHandler

FILENAME_TEMPLATE = "{idx:09d}.png"


class VideoDataHandler(BaseModalityHandler):
    """
    Handles video/frame data preparation.

    Responsibilities:
    - Validate camera names and locate one video file per camera
    - Validate that all cameras share the same FPS and frame count
    - Extract frames from video files (mp4, mov)
    - Validate existing frame sequences
    - Generate input recipes for frame loaders
    - Load camera calibration data
    """

    def __init__(self, io: SequenceIO, sequence_context: SequenceRuntimeConfig):
        # Shared fields
        super().__init__(io, sequence_context)

        # Video-specific state
        self.start_frame_index = self.dataset_properties.start_frame_index

        # Resolved during prepare()
        self.camera_video_paths: Optional[Dict[str, Path]] = None
        self.calibration: Optional[Dict[str, Any]] = None

    @property
    def modality_name(self) -> str:
        return "video"

    def prepare(self) -> None:
        log_with_underscore("Preparing Video Modality...")

        # Validate camera names not empty
        if not self.all_camera_names:
            raise ValueError("No camera names provided.")
        for name in self.all_camera_names:
            if not name or not name.strip():
                raise ValueError(f"Invalid camera name {name!r}")

        # Find exactly one video per camera (recursive, exact name match)
        self.camera_video_paths = self._find_video_paths()

        # Probe all videos, check cross-camera consistency, then validate against config
        self.fps, self.length_frames = self._resolve_fps_and_length()
        self.start_frame = timestamp_to_frame_index(self.sequence_context.video_start, self.fps)

        # Check and create input data if necessary
        self._input_data_creation()

        # Load camera calibration if available
        self.calibration = self._load_calibration()

        self._available = True
        logging.info("Video DATA CREATION completed.")

    def get_recipe(self) -> VideoInputRecipe:
        """
        Generates the Recipe config to be injected into the subprocess TOML.
        """
        return VideoInputRecipe(
            root_path=str(self.nice_input_folder),
            camera_names=sorted(list(self.all_camera_names)),
            filename_template="{camera}/frames/" + FILENAME_TEMPLATE,
            range_start=self.start_frame,
            range_end=self.start_frame + self.length_frames,
            step=1,
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _find_video_paths(self) -> Dict[str, Path]:
        """
        For each camera, recursively search its source folder for exactly one
        video file whose stem or any ancestor directory name equals the camera
        name exactly.

        Returns:
            dict mapping camera name -> Path of the matched video file.

        Raises:
            ValueError: If a camera matches zero or more than one file.
        """
        result: Dict[str, Path] = {}

        for cam in self.all_camera_names:
            source_folder = self.io.get_data_source_folder(cam)

            candidates = []
            for p in source_folder.rglob("*"):
                # is this a video extension?
                if p.suffix.lower() not in vid.VIDEO_EXTENSIONS:
                    continue
                # does it contains exact match of camera name in path?
                if not _contains_camera_name(p, cam):
                    continue
                candidates.append(p)

            if len(candidates) == 0:
                raise ValueError(
                    f"No video file found for camera '{cam}' in '{source_folder}'. "
                    f"Expected a file or directory named exactly '{cam}'."
                )
            if len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous: found {len(candidates)} video files for camera '{cam}' "
                    f"in '{source_folder}': {[str(p) for p in candidates]}"
                )

            result[cam] = candidates[0]

        return result

    def _resolve_fps_and_length(self) -> tuple[int, int]:
        """
        Probe all camera videos, assert cross-camera consistency of FPS and
        frame count, then validate FPS against config.

        Returns:
            (fps, length_frames) — fps detected from videos, length in frames
            from config or auto-detected.

        Raises:
            ValueError: If cameras disagree on FPS or frame count, or if
                video_start is beyond the end of the video.
        """
        infos = {}
        for cam, path in self.camera_video_paths.items():
            raw = vid.probe_video(str(path))
            infos[cam] = vid.json_to_video_info(raw)

        # Cross-camera consistency
        fps_values = {cam: int(info.fps) for cam, info in infos.items() if info.fps is not None}
        frame_values = {cam: info.frames for cam, info in infos.items() if info.frames is not None}

        if len(set(fps_values.values())) > 1:
            raise ValueError(f"Cameras have inconsistent FPS: {fps_values}")

        if len(set(frame_values.values())) > 1:
            raise ValueError(f"Cameras have inconsistent frame counts: {frame_values}")

        # Resolve FPS
        fps = next(iter(fps_values.values())) if fps_values else None
        if fps is None:
            raise ValueError("Could not determine FPS from any camera video.")

        if fps != self.sequence_context.fps:
            logging.warning(f"Detected fps={fps} does not match config fps={self.sequence_context.fps}!")

        # Resolve length
        video_length_frame = timestamp_to_frame_index(self.sequence_context.video_length, fps)
        if video_length_frame > 0:
            return fps, video_length_frame

        # Auto-detect from frame count
        total_frames = next(iter(frame_values.values())) if frame_values else None
        if total_frames is None:
            raise ValueError("Could not determine frame count from any camera video.")

        start_frame = timestamp_to_frame_index(self.sequence_context.video_start, fps)
        available = total_frames - start_frame
        if available <= 0:
            raise ValueError(f"video_start ({start_frame}) is beyond the end of the video " f"({total_frames} frames).")

        logging.info(f"Auto-detected length: {available} frames " f"(Total: {total_frames}, Start: {start_frame})")
        return fps, available

    def _input_data_creation(self) -> None:
        """
        Initializes the data required for running NICE toolbox.
        """
        if self._check_frames_exist():
            logging.info("Frames FOUND in nicetoolbox input folder")
        else:
            logging.info("EXTRACTING frames from video...")
            self._extract_frames_from_video()

    def _check_frames_exist(self) -> bool:
        """
        Check if frames exist in the nicetoolbox input folder ("Source of truth").

        Returns:
            bool: True if frames exist for all cameras, False otherwise.
        """
        start_idx = self.start_frame
        end_idx = self.start_frame + self.length_frames - 1

        for cam in self.all_camera_names:
            cam_folder = self.nice_input_folder / cam / "frames"

            start_name = FILENAME_TEMPLATE.format(idx=start_idx)
            end_name = FILENAME_TEMPLATE.format(idx=end_idx)

            if not ((cam_folder / start_name).exists() and (cam_folder / end_name).exists()):
                logging.info(f"No input frames found for camera '{cam}': " f"Files will be created in '{cam_folder}'.")
                return False

        return True

    def _extract_frames_from_video(self) -> None:
        """
        Extract frames from each camera's resolved video file into the
        nicetoolbox_input folder.
        """
        for cam, video_path in self.camera_video_paths.items():
            logging.info(f"Extracting frames for camera '{cam}' from '{video_path}'...")

            raw_video_info = vid.probe_video(str(video_path))
            video_info_path = self.nice_input_folder / f"{cam}_meta.json"
            with open(video_info_path, "w") as f:
                json.dump(raw_video_info, f, indent=4)

            video_info = vid.json_to_video_info(raw_video_info)

            cam_folder = self.nice_input_folder / cam
            frames_folder = cam_folder / "frames"
            frames_folder.mkdir(parents=True, exist_ok=True)

            vid.split_into_frames(
                str(video_path),
                str(frames_folder) + "/",
                video_info.frames,
                start_frame=self.start_frame_index,
                keep_indices=True,
            )

    def _load_calibration(self) -> dict | None:
        """
        Load camera calibration from a file for a specific dataset.

        Returns:
            dict: A dictionary containing the loaded camera calibration.

        Raises:
            KeyError: If loading camera calibration for the specified
            dataset is not implemented.
        """
        calib_path = self.io.get_calibration_file()
        if not calib_path or not os.path.isfile(calib_path):
            logging.warning("Calibration file not found, skipping calibration.")
            return None

        calib_details = "__".join([word for word in [self.session_id, self.sequence_id] if word])
        try:
            loaded_calib = np.load(calib_path, allow_pickle=True)[calib_details].item()
        except KeyError as err:
            logging.exception(
                f"Calibration for session '{self.session_id}' and sequence "
                f"'{self.sequence_id}' not found for calibration file at "
                f"'{calib_path}'."
            )
            raise err
        try:
            calib = {key: value for key, value in loaded_calib.items() if key in self.all_camera_names}
        except Exception as err:
            logging.exception(f"An error occurred while creating calibration dictionary: {err}")
            raise err

        return calib


# -------------------------------------------------------------------------
# Module-level helpers
# -------------------------------------------------------------------------


def _contains_camera_name(video_path: Path, camera_name: str) -> bool:
    """
    Return True if the camera name matches the video file's stem or any
    directory component in its path exactly (case-insensitive).

    'cam_front' matches:
      - cam_front.mp4          (stem == camera_name)
      - cam_front/video.mp4    (parent dir == camera_name)
      - root/cam_front/sub/v.mp4

    'cam_front' does NOT match:
      - cam_front_test.mp4     (stem != camera_name)
      - cam_front_test/v.mp4   (dir != camera_name)
    """
    cam_lower = camera_name.lower()

    # Check file stem
    if video_path.stem.lower() == cam_lower:
        return True

    # Check every directory component in the path
    return any(part.lower() == cam_lower for part in video_path.parts[:-1])
