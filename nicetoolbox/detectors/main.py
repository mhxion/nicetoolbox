"""
Run the NICE toolbox inference pipeline. The main script imports various modules and
classes to run method detectors and feature detectors on the provided datasets.
"""

import argparse
import logging
import time

from nicetoolbox_core.errors import ErrorLevel

from ..configs.schemas.detectors_run_file import LoggingLevelEnum
from ..utils import logging_utils as log_ut
from ..utils import to_csv as csv
from ..utils.error_handling import manage_error_scope
from . import config_handler as confh
from .data import VideoData
from .feature_detectors.base_feature import BaseFeature
from .feature_detectors.gaze_interaction.gaze_distance import GazeDistance
from .feature_detectors.gaze_multiview.gaze_fusion import GazeFusion
from .feature_detectors.kinematics.velocity_body import VelocityBody
from .feature_detectors.proximity.body_distance import BodyDistance
from .in_out import VideoIO
from .method_detectors.base_method import BaseMethod
from .method_detectors.body_joints.mmpose_framework import HRNetw48, VitPose
from .method_detectors.emotion_individual.py_feat import PyFeat
from .method_detectors.gaze_individual.Multiview_Eth_XGaze import MultiviewEthXgaze
from .method_detectors.head_orientation.spiga_detector import Spiga

ALL_DETECTORS = dict(
    # method detectors
    multiview_eth_xgaze=MultiviewEthXgaze,
    hrnetw48=HRNetw48,
    vitpose=VitPose,
    py_feat=PyFeat,
    spiga=Spiga,
    # feature detectors
    velocity_body=VelocityBody,
    body_distance=BodyDistance,
    gaze_distance=GazeDistance,
    gaze_fusion=GazeFusion,
)


def main(run_config_file, machine_specifics_file):
    """
    Main entry point for the NICE Toolbox detectors pipeline.

    Args:
        run_config_file (str): The path to the run configuration file.
        detector_config_file (str): The path to the detector configuration file.
        machine_specifics_file (str): The path to the machine specifics file.
    """
    # ==================================
    # PHASE 1: Load Static Configuration
    # ==================================
    config = confh.Configuration(run_config_file, machine_specifics_file)

    error_level = ErrorLevel(config.error_level)
    log_level = LoggingLevelEnum(config.log_level)

    main_output_folder = config.run_config.io.out_folder
    main_output_folder.mkdir(parents=True, exist_ok=True)

    log_file = main_output_folder / "nicetoolbox.log"
    log_ut.setup_logging(log_file, log_level.name)
    log_ut.log_main_banner(f"NICE TOOLBOX STARTED. Saving results to '{main_output_folder}'.")
    config.save_experiment_config(main_output_folder)

    all_algorithms = config.get_all_detector_names()

    # ========================
    # PHASE 2: Process Videos
    # ========================
    for video_context in config.iter_video_contexts():  # for each video
        # get video meta information for logging
        dataset_name = video_context.dataset_name
        session_id = video_context.video_config.session_ID
        sequence_id = video_context.video_config.sequence_ID
        sequence_name = f"{dataset_name}:{session_id}:{sequence_id}"

        log_ut.log_banner(f"RUNNING dataset: '{dataset_name}', session: '{session_id}', sequence: '{sequence_id}'")
        with manage_error_scope(error_level, ErrorLevel.VIDEO, sequence_name):
            # Create IO and Data from runtime config for the current video
            io = VideoIO(video_context, all_algorithms)
            data = VideoData(video_context, io)

            # Algorithms based on user-selected components
            selected_algorithms = video_context.all_selected_algorithms
            method_names = [a for a in selected_algorithms if issubclass(ALL_DETECTORS[a], BaseMethod)]
            feature_names = [a for a in selected_algorithms if issubclass(ALL_DETECTORS[a], BaseFeature)]
            ordered_detectors = method_names + feature_names

            # ======================
            # PHASE 3: RUN DETECTORS
            # ======================
            for detector_name in ordered_detectors:  # for each detector
                with manage_error_scope(error_level, ErrorLevel.DETECTOR, detector_name):
                    log_ut.log_with_underscore(f"STARTING '{detector_name}'.")
                    start_time = time.time()

                    detector_class = ALL_DETECTORS[detector_name]
                    detector = detector_class(io, data, video_context)

                    result_data = detector.run()

                    if config.visualize and detector.visualize:
                        detector.visualization(result_data)

                    logging.info(f"FINISHED '{detector_name}' in {time.time() - start_time}s.\n\n")

            # Convert results to .csv
            if config.save_csv:
                csv.results_to_csv(io.out_sub_folder, io.csv_folder)
                logging.info("Converting current video results to CSV successful.")

    log_ut.log_with_underscore("Detectors finished.")


def entry_point():
    """Entry point for running NICE toolbox detectors."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_config",
        default="configs/detectors_run_file.toml",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--machine_specifics",
        default="machine_specific_paths.toml",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    main(args.run_config, args.machine_specifics)


if __name__ == "__main__":
    entry_point()
