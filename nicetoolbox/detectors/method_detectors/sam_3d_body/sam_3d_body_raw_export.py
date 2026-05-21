"""
Serialize SAM 3D Body inference raw pack (_sam_3d_body_inference_raw.npz) to JSON/CSV
under body_joints_local/<algorithm>/detector_output/.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_to_jsonable(x) for x in obj.tolist()]
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return str(obj)


def dump_sam3d_raw_exports(
    raw_npz_path: Path,
    out_dir: Path,
    *,
    write_jsonl: bool,
    write_csv: bool,
) -> None:
    """Load raw NPZ (unchanged from inference) and write JSONL + optional CSV."""
    if not write_jsonl and not write_csv:
        return
    raw = np.load(raw_npz_path, allow_pickle=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {}
    for key in raw.files:
        if key == "per_frame_outputs":
            continue
        try:
            manifest[key] = _to_jsonable(raw[key])
        except Exception as e:
            manifest[key] = f"<serialization skipped: {e}>"
    manifest_path = out_dir / "sam3d_raw_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("SAM 3D Body wrote raw manifest %s", manifest_path)

    per_frame = raw["per_frame_outputs"].tolist()
    mode = str(np.asarray(raw["mode"]).item()) if "mode" in raw.files else "unknown"
    cam_order = [str(x) for x in raw["camera_names_order"].tolist()] if "camera_names_order" in raw.files else []

    if write_jsonl:
        jlp = out_dir / "sam3d_raw_per_frame.jsonl"
        with jlp.open("w", encoding="utf-8") as f:
            for fi, fb in enumerate(per_frame):
                rec = {"frame_index": fi, "mode": mode, "per_frame": _to_jsonable(fb)}
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        logging.info("SAM 3D Body wrote raw JSONL %s", jlp)

    if write_csv:
        csv_path = out_dir / "sam3d_raw_person_rows.csv"
        fieldnames = [
            "frame_index",
            "camera",
            "person_index",
            "focal_length",
            "pred_cam_t_0",
            "pred_cam_t_1",
            "pred_cam_t_2",
            "bbox_0",
            "bbox_1",
            "bbox_2",
            "bbox_3",
            "pred_keypoints_2d_json",
            "pred_keypoints_3d_json",
            "pred_vertices_json",
            "global_rot_json",
            "body_pose_params_json",
            "hand_pose_params_json",
            "shape_params_json",
            "scale_params_json",
            "expr_params_json",
            "pred_pose_raw_json",
            "mhr_model_params_json",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for fi, fb in enumerate(per_frame):
                if mode == "multi" and isinstance(fb, dict):
                    for cam in cam_order:
                        plist = fb.get(cam, [])
                        for pi, person in enumerate(plist):
                            row = _csv_row(fi, cam, pi, person)
                            w.writerow(row)
                elif isinstance(fb, list):
                    cam = cam_order[0] if cam_order else ""
                    for pi, person in enumerate(fb):
                        row = _csv_row(fi, cam, pi, person)
                        w.writerow(row)
        logging.info("SAM 3D Body wrote raw CSV %s", csv_path)


def _csv_row(frame_index: int, camera: str, person_index: int, person: dict[str, Any]) -> dict[str, Any]:
    p = dict(person)
    bbox = np.asarray(p.get("bbox", np.zeros(4)), dtype=np.float64).ravel()
    pct = np.asarray(p.get("pred_cam_t", np.zeros(3)), dtype=np.float64).ravel()
    row: dict[str, Any] = {
        "frame_index": frame_index,
        "camera": camera,
        "person_index": person_index,
        "focal_length": float(p.get("focal_length", np.nan)),
        "pred_cam_t_0": float(pct[0]) if pct.size > 0 else "",
        "pred_cam_t_1": float(pct[1]) if pct.size > 1 else "",
        "pred_cam_t_2": float(pct[2]) if pct.size > 2 else "",
        "bbox_0": float(bbox[0]) if bbox.size > 0 else "",
        "bbox_1": float(bbox[1]) if bbox.size > 1 else "",
        "bbox_2": float(bbox[2]) if bbox.size > 2 else "",
        "bbox_3": float(bbox[3]) if bbox.size > 3 else "",
        "pred_keypoints_2d_json": json.dumps(_to_jsonable(p.get("pred_keypoints_2d"))),
        "pred_keypoints_3d_json": json.dumps(_to_jsonable(p.get("pred_keypoints_3d"))),
        "pred_vertices_json": json.dumps(_to_jsonable(p.get("pred_vertices"))),
        "global_rot_json": json.dumps(_to_jsonable(p.get("global_rot"))),
        "body_pose_params_json": json.dumps(_to_jsonable(p.get("body_pose_params"))),
        "hand_pose_params_json": json.dumps(_to_jsonable(p.get("hand_pose_params"))),
        "shape_params_json": json.dumps(_to_jsonable(p.get("shape_params"))),
        "scale_params_json": json.dumps(_to_jsonable(p.get("scale_params"))),
        "expr_params_json": json.dumps(_to_jsonable(p.get("expr_params"))),
        "pred_pose_raw_json": json.dumps(_to_jsonable(p.get("pred_pose_raw"))),
        "mhr_model_params_json": json.dumps(_to_jsonable(p.get("mhr_model_params"))),
    }
    return row
