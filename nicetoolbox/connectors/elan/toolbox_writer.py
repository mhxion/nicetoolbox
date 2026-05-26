"""Build Toolbox npz arrays from ELAN annotation data."""

import logging

import numpy as np

from .labeling_data import MultipersonHierarchicalData
from .npz_schema import NpzSchema


def _seconds_to_frame(sec: float, fps: float) -> int:
    return round(sec * fps)


def hierarchical_to_npz_dict(
    hier_data: MultipersonHierarchicalData,
    schema: NpzSchema,
    fps: float,
    start_sec: float,
    end_sec: float,
    serialize: str,
    category_gap_fills: dict[str, str],
    reset_frames: bool,
) -> dict:
    start_frame = _seconds_to_frame(start_sec, fps)
    end_frame = _seconds_to_frame(end_sec, fps)
    n_frames = end_frame - start_frame

    subjects = schema.subjects
    cameras = ["3d"]
    frame_offset = 0 if reset_frames else start_frame
    frame_indices = [f"{frame_offset + i:09d}" for i in range(n_frames)]
    categories = schema.categories

    logging.info(
        f"NPZ dimensions: subjects={len(subjects)}, cameras={len(cameras)}, "
        f"categories={len(categories)}, frames={n_frames} [{start_frame}:{end_frame}]"
    )

    if serialize == "text":
        return _build_text_dict(
            hier_data, subjects, cameras, frame_indices, categories, n_frames, start_frame, fps, category_gap_fills
        )
    if serialize == "boolean":
        return _build_boolean_dict(
            hier_data,
            schema,
            subjects,
            cameras,
            frame_indices,
            categories,
            n_frames,
            start_frame,
            fps,
            category_gap_fills,
        )
    raise ValueError(f"serialize must be 'text' or 'boolean', got '{serialize}'")


def _build_text_dict(
    hier_data, subjects, cameras, frame_indices, categories, n_frames, start_frame, fps, category_gap_fills
) -> dict:
    ret_data = np.full((len(subjects), len(cameras), n_frames, len(categories)), fill_value="", dtype="<U64")

    for s_idx, subject in enumerate(subjects):
        for c_idx, category in enumerate(categories):
            intervals = hier_data.data.get(subject, {}).get(category, [])
            for iv in intervals:
                iv_start = _seconds_to_frame(iv.start_sec, fps) - start_frame
                iv_end = _seconds_to_frame(iv.end_sec, fps) - start_frame
                label_str = ", ".join(sorted(iv.labels))
                for f_idx in range(iv_start, iv_end):
                    ret_data[s_idx, 0, f_idx, c_idx] = label_str

            gap_label = category_gap_fills.get(category)
            if gap_label is not None:
                gap_mask = ret_data[s_idx, 0, :, c_idx] == ""
                ret_data[s_idx, 0, :, c_idx][gap_mask] = gap_label

    key = "labels"
    data_description = {key: {"axis0": subjects, "axis1": cameras, "axis2": frame_indices, "axis3": categories}}
    return {key: ret_data, "data_description": np.array(data_description, dtype=object)}


def _build_boolean_dict(
    hier_data, schema, subjects, cameras, frame_indices, categories, n_frames, start_frame, fps, category_gap_fills
) -> dict:
    out_dict: dict = {}
    data_description: dict = {}

    for category in categories:
        cat_labels = schema.labels_per_category.get(category, [])
        if not cat_labels:
            logging.warning(f"Category '{category}' has no labels, skipping")
            continue

        label_index = {label: idx for idx, label in enumerate(cat_labels)}
        arr = np.full((len(subjects), len(cameras), n_frames, len(cat_labels)), fill_value=np.nan, dtype=np.float64)

        for s_idx, subject in enumerate(subjects):
            intervals = hier_data.data.get(subject, {}).get(category, [])
            for iv in intervals:
                iv_start = _seconds_to_frame(iv.start_sec, fps) - start_frame
                iv_end = _seconds_to_frame(iv.end_sec, fps) - start_frame
                for f_idx in range(iv_start, iv_end):
                    arr[s_idx, 0, f_idx, :] = 0.0
                    for label in iv.labels:
                        arr[s_idx, 0, f_idx, label_index[label]] = 1.0

        gap_label = category_gap_fills.get(category)
        if gap_label is not None:
            gap_label_idx = label_index[gap_label]
            nan_mask = np.isnan(arr[:, 0, :, 0])
            arr[:, 0, :, :][nan_mask] = 0.0
            arr[:, 0, :, gap_label_idx][nan_mask] = 1.0
            logging.info(f"Gap-filled {int(nan_mask.sum())} frames in '{category}' with '{gap_label}'")

        out_dict[category] = arr
        data_description[category] = {"axis0": subjects, "axis1": cameras, "axis2": frame_indices, "axis3": cat_labels}
        logging.info(f"Boolean array for '{category}': shape={arr.shape}, labels={cat_labels}")

    out_dict["data_description"] = np.array(data_description, dtype=object)
    return out_dict
