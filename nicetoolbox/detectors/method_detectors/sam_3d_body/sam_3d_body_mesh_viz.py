"""Project MHR mesh vertices onto camera frames (visualization_3d).

Uses cv2.solvePnPRansac on pred_keypoints_3d  pred_keypoints_2d then
cv2.projectPoints for pred_vertices. Falls back to a crude focal length if
calibration is missing.
"""

from typing import Any

import cv2
import numpy as np


def _intrinsics_from_calibration_or_image(
    cal: dict[str, Any] | None, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    if cal and "intrinsic_matrix" in cal:
        K = np.asarray(cal["intrinsic_matrix"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(cal.get("distortions", np.zeros(5, dtype=np.float64)), dtype=np.float64).ravel()
        if dist.size < 5:
            dist = np.pad(dist, (0, 5 - dist.size))
        else:
            dist = dist[:5]
        return K, dist
    f = float(max(width, height))
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return K, dist


def _estimate_pose_pnp(
    kp3: np.ndarray,
    kp2: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Return (rvec, tvec, z_flip) for projecting 3D points consistent with kp2, or None."""
    obj = np.asarray(kp3, dtype=np.float64)
    img = np.asarray(kp2, dtype=np.float64)
    if obj.ndim != 2 or obj.shape[1] < 3 or img.ndim != 2 or img.shape[1] < 2:
        return None
    obj = obj[:, :3].astype(np.float32)
    img = img[:, :2].astype(np.float32)
    valid = np.isfinite(obj).all(axis=1) & np.isfinite(img).all(axis=1)
    if int(valid.sum()) < 6:
        return None
    obj = obj[valid]
    img = img[valid]

    def _try_once(obj_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        ok, rvec, tvec, _inliers = cv2.solvePnPRansac(
            obj_pts,
            img,
            K.astype(np.float64),
            dist.astype(np.float64),
            iterationsCount=200,
            reprojectionError=12.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or rvec is None or tvec is None:
            return None
        return rvec.astype(np.float64), tvec.astype(np.float64)

    out = _try_once(obj)
    if out is not None:
        return out[0], out[1], False
    obj_flip = obj.copy()
    obj_flip[:, 2] *= -1.0
    out = _try_once(obj_flip)
    if out is not None:
        return out[0], out[1], True
    return None


def _project_points(
    pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    z_flip: bool,
) -> np.ndarray:
    X = np.asarray(pts, dtype=np.float64).reshape(-1, 3).astype(np.float32)
    if z_flip:
        X = X.copy()
        X[:, 2] *= -1.0
    uv, _jac = cv2.projectPoints(X, rvec, tvec, K.astype(np.float64), dist.astype(np.float64))
    return uv.reshape(-1, 2)


def _unique_edges_from_faces(faces: np.ndarray) -> list[tuple[int, int]]:
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] < 3:
        return []
    edges: set[tuple[int, int]] = set()
    for row in f:
        a, b, c = int(row[0]), int(row[1]), int(row[2])
        for i, j in ((a, b), (b, c), (c, a)):
            if i == j:
                continue
            edges.add((i, j) if i < j else (j, i))
    return list(edges)


def draw_mesh_wireframe_on_image(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    kp3: np.ndarray,
    kp2: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    color_bgr: tuple[int, int, int],
    line_thickness: int = 1,
) -> bool:
    """Draw mesh wireframe on image (in place). Returns True if anything was drawn."""
    pose = _estimate_pose_pnp(kp3, kp2, K, dist)
    if pose is None:
        return False
    rvec, tvec, z_flip = pose
    V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if V.shape[0] < 3:
        return False
    uv = _project_points(V, rvec, tvec, K, dist, z_flip)
    h, w = image.shape[:2]
    edges = _unique_edges_from_faces(faces)
    drawn = False
    for i, j in edges:
        if i >= uv.shape[0] or j >= uv.shape[0]:
            continue
        p0, p1 = uv[i], uv[j]
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            continue
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        if (0 <= x0 < w and 0 <= y0 < h) or (0 <= x1 < w and 0 <= y1 < h):
            cv2.line(image, (x0, y0), (x1, y1), color_bgr, line_thickness, cv2.LINE_AA)
            drawn = True
    return drawn
