"""
Video/frame data handler for the NICE Toolbox.

Handles frame extraction from video files and preparation of image sequences.
Also owns camera calibration loading since calibration is video-specific.
"""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from nicetoolbox_core.input_recipes import VideoInputRecipe

from ...configs.models.video_timestamp import timestamp_to_frame_index
from ...configs.schemas import dataset_properties
from ...configs.video_runtime_config import SequenceRuntimeConfig
from ...utils import video as vid
from ...utils.logging_utils import log_with_underscore
from ..in_out import SequenceIO
from .handler import BaseModalityHandler


class VideoDataHandler(BaseModalityHandler):
    """
    Handles video/frame data preparation.

    Responsibilities:
    - Detect input format (video files vs image sequences)
    - Extract frames from video files (mp4, avi)
    - Validate existing frame sequences
    - Generate input recipes for frame loaders
    - Load camera calibration data
    """

    def __init__(
        self,
        # Shared fields (passed to base)
        io: SequenceIO,
        sequence_context: SequenceRuntimeConfig,
    ):
        super().__init__(io=io, sequence_context=sequence_context)

        # Video-specific state
        self.start_frame_index = self.dataset_properties.start_frame_index
        self.calibration_path = getattr(dataset_properties, "path_to_calibrations", None)
        self.filename_template = getattr(dataset_properties, "filename_template", "{idx:09d}.png")

        self.input_folder = io.get_nice_input_folder()

        # Resolved during prepare()
        self.input_format: Optional[str] = None
        self.video_sample_path: Optional[Path] = None
        self.calibration: Optional[Dict[str, Any]] = None

    @property
    def modality_name(self) -> str:
        return "video"

    def prepare(self) -> None:
        log_with_underscore("Preparing Video Modality...")

        # 1. Detect input format and search for an example video file
        self.input_format: str = self._get_input_format()
        self.video_sample_path: Path = self._get_example_video_path()

        # 2. Validate fps, video start and video length given an example video file
        self.fps: int = self._get_fps_validated(self.fps)
        # TODO: do here actual validation for start frame
        self.start_frame = timestamp_to_frame_index(self.sequence_context.video_start, self.fps)
        self.length_frames = self._resolve_video_length(self.sequence_context.video_length, self.fps)

        # 3. Check and create input data if necessary
        self._input_data_creation()

        # 4. Load camera calibration if available
        self.calibration: dict | None = self._load_calibration()

        self._available = True
        logging.info("Video DATA CREATION completed.")

    def get_recipe(self) -> VideoInputRecipe:
        """
        Generates the Recipe config to be injected into the subprocess TOML.
        """
        if self.input_format in [".avi", ".mp4"]:
            root = self.input_folder  # Extracted data lives in nicetoolbox_input
            template = "{camera}/frames/" + self.filename_template  # Structure: {cam}/frames/{idx:09d}.png
        else:
            raise NotImplementedError(f"Input recipe generation for '{self.input_format}' not implemented.")

        return VideoInputRecipe(
            root_path=str(root),
            camera_names=sorted(list(self.all_camera_names)),
            filename_template=template,
            range_start=self.start_frame,
            range_end=self.start_frame + self.length_frames,
            step=1,
        )

    # -------------------------------------------------------------------------
    # Helper methods for video processing
    # -------------------------------------------------------------------------

    def _get_input_format(self) -> str:
        """
        Get the input format for the given camera names.

        Returns:
            str: The input format for the given camera names.

        Raises:
            ValueError: If multiple or no valid input format is found in the data
                input folder.
        """
        possible_formats = [".mp4", ".avi", ".png", ".jpg", ".jpeg"]
        cam_0 = self.all_camera_names[0]
        example_input_folder = self.io.get_data_source_folder(cam_0)

        found_formats = [name in "_".join(sorted(os.listdir(example_input_folder))) for name in possible_formats]
        if sum(found_formats) != 1:
            raise ValueError(
                f"Multiple/no valid input format found in '{example_input_folder}'. "
                f"Found '{found_formats}', valid formats are ['mp4', 'avi'].",
            )

        return possible_formats[found_formats.index(True)]

    def _get_example_video_path(self) -> Path:
        """
        Finds a video file for metadata extraction Reused by FPS check and Length resolution.

        Returns:
            Path: The path to an example video file.
        """
        cam_0 = self.all_camera_names[0]
        example_input_folder = self.io.get_data_source_folder(cam_0)

        files = sorted(example_input_folder.glob(f"*{self.input_format}"))

        if not files:
            raise FileNotFoundError(
                f"No video files ({self.input_format}) found in {example_input_folder} " f"for camera {cam_0}."
            )
        return files[0]

    def _get_fps_validated(self, target_fps: int) -> int:
        """
        Validates the frames per second (fps) of the input video files against the
        target fps specified in the configuration.

        Args:
            target_fps (int): The desired fps specified in the configuration.

        Returns:
            int: The fps of the input video files. Raises if target fps does not match detected fps.
        """
        if self.input_format in [".mp4", ".avi"]:
            fps = vid.get_fps(str(self.video_sample_path))
            if fps != target_fps:
                logging.warning(f"Detected fps = {fps} does not match fps given in the " f"config = {target_fps}!")
            return fps

        raise NotImplementedError(f"FPS validation for input format '{self.input_format}' is not implemented.")

    def _resolve_video_length(self, video_length: str | int, fps: int) -> int:
        """
        Resolves the video length in frames. If the video_length is specified in the
        configuration, it is returned directly. If it is (-1), the length is determined
        from the example video file.

        Returns:
            int: The resolved video length in frames.
        """
        video_length_frame = timestamp_to_frame_index(video_length, fps)
        if video_length_frame > 0:  # TODO: check if frame in range
            return video_length_frame

        # If length is -1, we need to figure out the length from the video file

        if self.input_format in [".mp4", ".avi"]:
            total_frames = vid.get_number_of_frames(str(self.video_sample_path))
            available_length = total_frames - self.start_frame

            if available_length <= 0:
                raise ValueError(
                    f"video_start ({self.start_frame}) is beyond the end of the video "
                    f"({total_frames} frames) in {self.video_sample_path.name}"
                )

            logging.info(
                f"Auto-detected length: {available_length} frames "
                f"(Total: {total_frames}, Start: {self.start_frame})"
            )
            return available_length

        raise NotImplementedError(f"Video length resolution for '{self.input_format}' not implemented.")

    def _input_data_creation(self) -> None:
        """
        Initializes the data required for running NICE toolbox.
        """
        if self.input_format in [".avi", ".mp4"]:
            if self._check_frames_exist():
                logging.info("Frames FOUND in nicetoolbox input folder")
            else:
                logging.info("EXTRACTING frames from video...")
                self._extract_frames_from_video()
            self.filename_template = "{idx:09d}.png"

        elif self.input_format in [".png", ".jpg", ".jpeg"]:
            # TODO: implement source frame checking
            # This requires knowing the original folder structure and
            # filename template from config, including session_id, sequence_id, etc.
            # cam_folder = root / self.session_id / self.sequence_id / cam /
            raise NotImplementedError("Checking source frames is not implemented yet.")
        else:
            raise NotImplementedError(f"Input format '{self.input_format}' not supported for data creation.")

    def _check_frames_exist(self) -> bool:
        """
        Check if frames exist in the nicetoolbox input folder ("Source of truth").

        Returns:
            bool: True if frames exist for all cameras, False otherwise.
        """
        template = self.filename_template

        start_idx = self.start_frame
        end_idx = self.start_frame + self.length_frames - 1

        for cam in self.all_camera_names:
            cam_folder = self.input_folder / cam / "frames"

            start_name = template.format(idx=start_idx)
            end_name = template.format(idx=end_idx)

            start_path = cam_folder / start_name
            end_path = cam_folder / end_name

            # Check existence
            if not (start_path.exists() and end_path.exists()):
                logging.info(f"No input frames found for camera '{cam}': " f"Files will be created in '{cam_folder}'.")
                return False

        return True

    def _extract_frames_from_video(self):
        """
        Create input frames from video files inside the nicetoolbox_input folder.

        This method detects video input files, splits them into frames, and organizes
        the frames into different data formats which are frames, segments, and snippets.

        Raises:
            AssertionError: If the length of the frame indices list does not match
                the specified video length.
            AssertionError: If the frame indices of different cameras do not match.
        """
        # detect all video input files
        # Build glob pattern for all camera input folders and list video files
        data_input_pattern = self.io.get_data_source_folder("*")
        pattern = data_input_pattern / "*"
        video_files = sorted(glob.glob(str(pattern)))

        for video_file in video_files:
            camera_name_indices = [name.lower() in video_file.lower() for name in list(self.all_camera_names)]
            if not any(camera_name_indices):
                continue
            camera_name = list(self.all_camera_names)[camera_name_indices.index(True)]

            logging.info("Extracting Video Specifications...")
            raw_video_info = vid.probe_video(video_file)
            video_info_path = os.path.join(self.input_folder, camera_name + "_meta.json")
            with open(video_info_path, "w") as f:
                json.dump(raw_video_info, f, indent=4)

            logging.info("Parsing Video Specifications...")
            video_info = vid.json_to_video_info(raw_video_info)

            # split video into frames
            input_folder = os.path.join(self.input_folder, camera_name)
            os.makedirs(input_folder, exist_ok=True)

            os.makedirs(os.path.join(input_folder, "frames"), exist_ok=True)
            vid.split_into_frames(
                video_file,
                os.path.join(input_folder, "frames/"),
                video_info.frames,
                start_frame=self.start_frame_index,
                keep_indices=True,
            )

    def _load_calibration(self) -> dict | None:
        """
        Load camera calibration from a file for a specific dataset.

        Currently implemented for the datasets 'dyadic_communication' and
        'mpi_inf_3dhp'.

        Returns:
            dict: A dictionary containing the loaded camera calibration.

        Raises:
            KeyError: If loading camera calibration for the specified
            dataset is not implemented.
        """
        # 1. Get calibration file path
        calib_path = self.io.get_calibration_file()
        if not calib_path or not os.path.isfile(calib_path):
            logging.warning("Calibration file not found, skipping calibration.")
            return None

        # 2. Load calibration file
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
            calib = dict((key, value) for key, value in loaded_calib.items() if key in self.all_camera_names)
        except Exception as err:
            logging.exception(f"An error occurred while creating calibration dictionary: {err}")
            raise err

        return calib
