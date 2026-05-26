import logging
from pathlib import Path

import numpy as np

from ...configs.models.video_timestamp import timestamp_to_frame_index
from ...utils import logging_utils as log_ut
from ...utils.to_csv import convert_npz_to_csv_files
from ...utils.video import json_to_video_info, probe_video
from ..config_handler import ConnectorConfigHandler
from .elan_configs import ElanImportGazeConfig
from .elan_parser import parse_elan_file
from .elan_processing import VideoMeta, trim_tiers, validate_video_alignment
from .gaze_parser import eyes_npz_to_gaze
from .labeling_data import rename_subjects
from .labeling_from_elan import elan_data_to_hierarchical
from .npz_schema import schema_from_data
from .toolbox_writer import hierarchical_to_npz_dict


def import_gaze(
    project_folder_path: Path,
    machine_specifics: Path,
    connector_config: Path,
) -> None:
    handler = ConnectorConfigHandler(
        project_folder_path,
        machine_specifics,
        connector_config,
        ElanImportGazeConfig,
    )
    cfg = handler.connector_config

    log_ut.log_main_banner("ELAN CONNECTOR: import_gaze")
    logging.info(f"Project path: '{handler.project_folder}'")
    logging.info(f"Run config: '{connector_config}'")
    logging.info(f"Sequences: {list(cfg.run)}")

    for sequence_id, sequence in cfg.run.items():
        log_ut.log_banner(f"Sequence: {sequence_id}")
        logging.info(f"Input:  {sequence.input}")
        logging.info(f"Output: {sequence.output}")

        video_info = json_to_video_info(probe_video(str(sequence.video)))
        video_meta = VideoMeta(fps=video_info.fps, duration_sec=video_info.duration_in_sec)
        logging.info(f"Video: fps={video_meta.fps}, duration={video_meta.duration_sec:.3f}s")

        fps = video_meta.fps
        start_sec = timestamp_to_frame_index(sequence.start, fps) / fps
        end_sec = video_meta.duration_sec if sequence.end == -1 else timestamp_to_frame_index(sequence.end, fps) / fps

        elan_data = parse_elan_file(sequence.input)
        validate_video_alignment(elan_data, video_meta)

        logging.info(f"Trimming to [{start_sec:.3f}s, {end_sec:.3f}s]...")
        trimmed = trim_tiers(elan_data, start_sec, end_sec)

        hier_data = elan_data_to_hierarchical(trimmed)
        logging.info(f"Converted: {hier_data}")

        if cfg.subjects:
            hier_data = rename_subjects(hier_data, cfg.subjects)

        schema = schema_from_data(hier_data)
        logging.info(f"Schema: {schema}")

        eyes_npz = hierarchical_to_npz_dict(
            hier_data,
            schema,
            fps,
            start_sec,
            end_sec,
            serialize="boolean",
            category_gap_fills={},
            reset_frames=sequence.reset_frames,
        )

        gaze_npz = eyes_npz_to_gaze(eyes_npz)

        sequence.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(sequence.output, **gaze_npz)
        logging.info(f"Saved gaze NPZ to: {sequence.output}")

        if cfg.export_csv:
            convert_npz_to_csv_files(sequence.output, sequence.output.parent)
