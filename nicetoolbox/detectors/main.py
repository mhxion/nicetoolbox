"""
Run the NICE toolbox inference pipeline. The main script imports various modules and
classes to run method detectors and feature detectors on the provided datasets.
"""

import argparse
import logging
import time
from pathlib import Path

from nicetoolbox_core.errors import ErrorLevel

from ..configs.schemas.detectors_run_file import LoggingLevelEnum
from ..download_manager.manager import AssetManager
from ..utils import logging_utils as log_ut
from ..utils import to_csv as csv
from ..utils.dependency_sort import sort_detectors_order
from ..utils.error_handling import manage_error_scope
from ..utils.system import check_long_path_support
from . import config_handler as confh
from .data import SequenceData
from .feature_detectors.gaze_interaction.gaze_distance import GazeDistance
from .feature_detectors.gaze_multiview.gaze_fusion import GazeFusion
from .feature_detectors.kinematics.velocity_body import VelocityBody
from .feature_detectors.proximity.body_distance import BodyDistance
from .in_out import SequenceIO
from .method_detectors.audio_transcription.whisper_detector import Whisper
from .method_detectors.body_joints.mmpose_framework_2d import (
    HRNetw48,
    RTMPoseLAIC,
    RTMPoseMPII,
    RTMPoseWHolebody,
    VitPose,
    VitPoseHuge,
)
from .method_detectors.body_joints.mmpose_framework_3d import MotionBERT
from .method_detectors.emotion_individual.py_feat import PyFeat
from .method_detectors.gaze_individual.Multiview_Eth_XGaze import MultiviewEthXgaze
from .method_detectors.head_orientation.spiga_detector import Spiga

ALL_DETECTORS = dict(
    # method detectors
    multiview_eth_xgaze=MultiviewEthXgaze,
    hrnetw48=HRNetw48,
    vitpose=VitPose,
    vitpose_huge=VitPoseHuge,
    rtmpose_l_aic=RTMPoseLAIC,
    rtmpose_l_wholebody=RTMPoseWHolebody,
    rtmpose_m_mpii=RTMPoseMPII,
    motionbert=MotionBERT,
    py_feat=PyFeat,
    spiga=Spiga,
    whisper=Whisper,
    # feature detectors
    velocity_body=VelocityBody,
    body_distance=BodyDistance,
    gaze_distance=GazeDistance,
    gaze_fusion=GazeFusion,
)


def main(project_folder_path: Path, machine_specifics_file: Path, run_config_file: Path):
    """
    Main entry point for the NICE Toolbox detectors pipeline.

    Args:
        project_folder_path (Path): Path to the project folder containing nice_project.toml.
        machine_specifics_file (Path): The path to the machine specifics file.
        run_config_file (Path): The path to the run configuration file.
    """
    # ==================================
    # PHASE 1: Load Static Configuration
    # ==================================
    config = confh.Configuration(project_folder_path, machine_specifics_file, run_config_file)

    error_level = ErrorLevel(config.error_level)
    log_level = LoggingLevelEnum(config.log_level)

    main_output_folder = config.run_config.io.out_folder
    main_output_folder.mkdir(parents=True, exist_ok=True)

    log_file = main_output_folder / "nicetoolbox.log"
    log_ut.setup_logging(log_file, log_level.name)
    check_long_path_support()
    log_ut.log_main_banner(f"NICE TOOLBOX STARTED. Saving results to '{main_output_folder}'.")

    # asset download manager
    manager = AssetManager(config)
    manager.ensure_assets_for_config(config)

    config.save_experiment_config(main_output_folder)

    all_algorithms = config.get_all_detector_names()

    # ==========================
    # PHASE 2: Process Sequences
    # ==========================
    for sequence_context in config.iter_sequence_contexts():  # for each sequence
        # get sequence meta information for logging
        sequence_str = str(sequence_context.video_config)
        log_ut.log_banner(f"RUNNING {sequence_str}")
        with manage_error_scope(error_level, ErrorLevel.SEQUENCE, sequence_str):
            # Create IO and Data from runtime config for the current sequence
            io = SequenceIO(sequence_context, all_algorithms)
            data = SequenceData(sequence_context, io)

            # Save video config
            config.save_video_config(sequence_context.video_config, io.get_output_folder("output"))

            # Algorithms based on user-selected components, topologically sorted
            selected_algorithms = sequence_context.all_selected_algorithms
            ordered_detectors = sort_detectors_order(
                sequence_context.detectors_config, selected_algorithms, config.check_missing_detectors_dependencies
            )

            # ======================
            # PHASE 3: RUN DETECTORS
            # ======================
            for detector_name in ordered_detectors:  # for each detector
                with manage_error_scope(error_level, ErrorLevel.DETECTOR, detector_name):
                    log_ut.log_with_underscore(f"STARTING '{detector_name}'.")
                    start_time = time.time()

                    detector_class = ALL_DETECTORS[detector_name]
                    detector = detector_class(io, data, sequence_context)

                    result_data = detector.run()

                    if config.visualize and detector.visualize:
                        detector.visualization(result_data)

                    logging.info(f"FINISHED '{detector_name}' in {time.time() - start_time}s.\n\n")

            # Convert results to .csv
            if config.save_csv:
                csv.results_to_csv(io.out_sub_folder, io.csv_folder)
                logging.info("Converting current sequence results to CSV successful.")

    log_ut.log_with_underscore("Detectors finished.")


def entry_point():
    """Entry point for running NICE toolbox detectors."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_folder_path",
        default=Path("."),
        type=Path,
        required=False,
        help="Path to the NICE Toolbox project folder containing nice_project.toml config",
    )
    parser.add_argument(
        "--machine_specifics",
        default=Path("machine_specific_paths.toml"),
        type=Path,
        required=False,
        help="Path to machine_specific_paths.toml config",
    )
    parser.add_argument(
        "--run_config",
        default="<configs_folder_path>/detectors_run_file.toml",
        type=Path,
        required=False,
        help="Path to detectors_run_file.toml, supports placeholders",
    )
    args = parser.parse_args()

    main(args.project_folder_path, args.machine_specifics, args.run_config)


if __name__ == "__main__":
    entry_point()
