"""Parse napari body-joint annotations into Toolbox body_joints NPZ.

The napari (DeepLabCut-style) annotation file uses a 4-level column index
(scorer, individuals, bodyparts, coords) and a 3-level row index
(root, camera, image_filename).

Output shape: 2d (subjects, cameras, frames, joints, 3) — x/y/confidence.
Missing frames within the kept window are written as NaN.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ...configs.models.video_timestamp import timestamp_to_frame_index
from ...utils.video import get_fps
from .napari_configs import NapariImportBodyJointsConfig, NapariSequenceConfig

REQUIRED_COORDS = ("x", "y", "likelihood")


def load_napari_dataframe(input_path: Path) -> pd.DataFrame:
    """Load a napari CSV or HDF5 annotation file as a MultiIndex DataFrame."""
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path, header=[0, 1, 2, 3], index_col=[0, 1, 2])
    elif suffix in {".h5", ".hdf", ".hdf5"}:
        df = pd.read_hdf(input_path)
    else:
        raise ValueError(f"Unsupported napari input extension '{suffix}'. Use .csv or .h5")

    df.columns = pd.MultiIndex.from_tuples(
        [tuple(str(lvl) for lvl in col) for col in df.columns],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    return df


def _frame_number(image_name: str) -> int:
    """Extract integer frame number from a filename like '000003600.png'."""
    return int(Path(image_name).stem)


def napari_to_body_joints_npz(
    sequence: NapariSequenceConfig,
    cfg: NapariImportBodyJointsConfig,
) -> None:
    """Convert one napari annotation file to a Toolbox body_joints NPZ."""
    df = load_napari_dataframe(sequence.input)
    logging.info(f"Loaded napari data from: {sequence.input} (rows={len(df)})")

    individuals = list(df.columns.get_level_values("individuals").unique())
    bodyparts = list(df.columns.get_level_values("bodyparts").unique())
    coords = list(df.columns.get_level_values("coords").unique())
    logging.info(f"Napari columns: individuals={individuals}, bodyparts={len(bodyparts)}, coords={coords}")

    for required in REQUIRED_COORDS:
        if required not in coords:
            raise ValueError(f"Expected coord '{required}' in napari columns, got: {coords}")

    if cfg.subjects:
        unknown = [v for v in individuals if v not in cfg.subjects]
        if unknown:
            raise ValueError(f"Napari individuals {unknown} not in [subjects] mapping {sorted(cfg.subjects)}")
        subjects = [cfg.subjects[v] for v in individuals]
    else:
        subjects = list(individuals)
    logging.info(f"Subject order: {individuals} -> {subjects}")

    napari_cameras = list(dict.fromkeys(df.index.get_level_values(1)))
    if cfg.cameras:
        unknown = [v for v in napari_cameras if v not in cfg.cameras]
        if unknown:
            raise ValueError(f"Napari cameras {unknown} not in [cameras] mapping {sorted(cfg.cameras)}")
        cameras = [cfg.cameras[v] for v in napari_cameras]
    else:
        cameras = list(napari_cameras)
    logging.info(f"Camera order: {napari_cameras} -> {cameras}")

    all_frame_numbers = sorted({_frame_number(name) for _, _, name in df.index})
    first_frame, last_frame = all_frame_numbers[0], all_frame_numbers[-1]

    fps = get_fps(str(sequence.video))
    logging.info(f"Video fps={fps}")
    start_frame = max(first_frame, timestamp_to_frame_index(sequence.start, fps))
    if sequence.end != -1:
        end_frame = timestamp_to_frame_index(sequence.end, fps)
    else:
        end_frame = last_frame
    n_frames = end_frame - start_frame

    if sequence.reset_frames:
        frame_labels = [f"{i:09d}" for i in range(n_frames)]
    else:
        frame_labels = [f"{start_frame + i:09d}" for i in range(n_frames)]
    logging.info(f"Frame range: {start_frame}..{end_frame} ({n_frames} frames, reset_frames={sequence.reset_frames})")

    arr = np.full(
        (len(subjects), len(cameras), n_frames, len(bodyparts), 3),
        fill_value=np.nan,
        dtype=np.float64,
    )

    for cam_idx, napari_cam in enumerate(napari_cameras):
        cam_df = df.xs(napari_cam, level=1)
        raw_frames = np.array([_frame_number(name) for _, name in cam_df.index])
        keep = (raw_frames >= start_frame) & (raw_frames < end_frame)
        cam_df: pd.DataFrame = cam_df.iloc[keep]
        frame_indices = raw_frames[keep] - start_frame
        for ind_idx, individual in enumerate(individuals):
            for joint_idx, joint in enumerate(bodyparts):
                for coord_idx, coord in enumerate(REQUIRED_COORDS):
                    col_values = cam_df.xs((individual, joint, coord), level=(1, 2, 3), axis=1).to_numpy().ravel()
                    arr[ind_idx, cam_idx, frame_indices, joint_idx, coord_idx] = col_values

    logging.info(f"Filled array of shape {arr.shape}")

    data_description = {
        "2d": {
            "axis0": subjects,
            "axis1": cameras,
            "axis2": frame_labels,
            "axis3": list(bodyparts),
            "axis4": ["coordinate_x", "coordinate_y", "confidence_score"],
        },
    }
    npz_dict = {
        "2d": arr,
        "data_description": np.array(data_description, dtype=object),
    }

    sequence.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(sequence.output, **npz_dict)
    logging.info(f"Saved body_joints NPZ to: {sequence.output}")
