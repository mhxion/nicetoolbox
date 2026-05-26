"""Derive gaze_look_at and gaze_mutual arrays from an eyes boolean NPZ dict."""

import logging

import numpy as np


def eyes_npz_to_gaze(eyes_npz_dict: dict) -> dict:
    """Derive gaze_look_at_3d and gaze_mutual_3d from the eyes boolean NPZ dict.

    Expects eyes_npz_dict to have an 'eyes' array with 'eyfx' in axis3:
      eyes shape: (subjects, cameras, frames, labels)
      eyfx = looking at the other person
    """
    eyes = eyes_npz_dict["eyes"]
    desc = eyes_npz_dict["data_description"].item()
    eyes_desc = desc["eyes"]

    subjects: list[str] = eyes_desc["axis0"]
    cameras: list[str] = eyes_desc["axis1"]
    frame_indices: list[str] = eyes_desc["axis2"]
    labels: list[str] = eyes_desc["axis3"]

    if "eyfx" not in labels:
        raise ValueError(f"Expected 'eyfx' label in eyes array, got: {labels}")

    eyfx_idx = labels.index("eyfx")
    n_subjects = len(subjects)
    n_frames = eyes.shape[2]

    logging.info(f"Loaded eyes array: subjects={subjects}, frames={n_frames}, labels={labels}")

    look_at = np.full((n_subjects, len(cameras), n_frames, n_subjects), fill_value=np.nan)
    mutual = np.full((n_subjects, len(cameras), n_frames, n_subjects), fill_value=np.nan)

    for s_idx in range(n_subjects):
        eyfx_frames = eyes[s_idx, 0, :, eyfx_idx] == 1.0
        for other_idx in range(n_subjects):
            if other_idx == s_idx:
                continue
            look_at[s_idx, 0, :, other_idx] = eyfx_frames.astype(np.float64)

    for s_idx in range(n_subjects):
        for other_idx in range(n_subjects):
            if other_idx == s_idx:
                continue
            mutual[s_idx, 0, :, other_idx] = (
                (look_at[s_idx, 0, :, other_idx] == 1.0) & (look_at[other_idx, 0, :, s_idx] == 1.0)
            ).astype(np.float64)

    logging.info(f"Derived gaze_look_at_3d and gaze_mutual_3d for {n_subjects} subjects")

    data_description = {
        "gaze_look_at_3d": {"axis0": subjects, "axis1": cameras, "axis2": frame_indices, "axis3": list(subjects)},
        "gaze_mutual_3d": {"axis0": subjects, "axis1": cameras, "axis2": frame_indices, "axis3": list(subjects)},
    }
    return {
        "gaze_look_at_3d": look_at,
        "gaze_mutual_3d": mutual,
        "data_description": np.array(data_description, dtype=object),
    }
