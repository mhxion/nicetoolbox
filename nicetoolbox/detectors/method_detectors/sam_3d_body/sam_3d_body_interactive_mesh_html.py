#!/usr/bin/env python3
"""
Build a single self-contained Plotly HTML: MHR meshes in 3D (no camera image overlay).

Meshes are drawn as wireframes (triangle edges only) with one subject per trace, shifted
along +x for side-by-side inspection. Edges and frames are subsampled by default
(see --max-edges / --max-frames) to keep the HTML browser-sized.

Invoked from the main nicetoolbox environment (plotly is a project dependency) or manually:

    python sam_3d_body_interactive_mesh_html.py \\
        --npz /path/to/body_mesh/sam_3d_body.npz \\
        --out-dir /path/to/body_mesh/sam_3d_body/visualization_3d_interactive \\
        --fps 30
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_SUBJECT_LINE_COLORS = [
    "rgb(188, 214, 248)",
    "rgb(248, 202, 214)",
    "rgb(198, 232, 204)",
    "rgb(244, 224, 176)",
    "rgb(220, 206, 246)",
    "rgb(230, 230, 200)",
]


def _load_vertices_and_faces(npz_path: Path, *, prefer_world: bool) -> tuple[np.ndarray, np.ndarray, bool]:
    z = np.load(npz_path, allow_pickle=True)
    if "faces" not in z.files:
        raise KeyError("NPZ missing 'faces'")
    faces = np.asarray(z["faces"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (T,3), got {faces.shape}")

    used_world = False
    V: np.ndarray | None = None
    if prefer_world and "vertices_world" in z.files:
        w = np.asarray(z["vertices_world"], dtype=np.float64)
        if w.size > 0 and np.any(np.isfinite(w)):
            V = w
            used_world = True
    if V is None:
        if "vertices" not in z.files:
            raise KeyError("NPZ missing 'vertices'")
        V = np.asarray(z["vertices"], dtype=np.float64)
    if V.ndim != 5:
        raise ValueError(f"vertices tensor must be 5D, got shape {V.shape}")
    return V, faces, used_world


def _camera_axis_index(V: np.ndarray, used_world: bool, camera_index: int) -> int:
    n_cam = V.shape[1]
    if used_world:
        return 0
    if camera_index < 0 or camera_index >= n_cam:
        raise ValueError(f"camera_index {camera_index} out of range for n_cam={n_cam}")
    return camera_index


def _fill_nan_vertices(verts: np.ndarray) -> np.ndarray:
    """Return (N,3) finite coords; NaNs filled from median of valid rows (viz-only)."""
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    if v.shape[0] == 0:
        return v
    mask = np.isfinite(v).all(axis=1)
    v2 = v.copy()
    if np.any(mask):
        fill = np.nanmedian(v[mask], axis=0)
    else:
        fill = np.zeros(3, dtype=np.float64)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    for i in range(3):
        bad = ~np.isfinite(v2[:, i])
        v2[bad, i] = fill[i]
    return v2


def _clamp_faces(faces: np.ndarray, n_verts: int) -> np.ndarray:
    if n_verts <= 0:
        return np.zeros((0, 3), dtype=np.int64)
    f = np.asarray(faces, dtype=np.int64)
    ok = (f >= 0).all(axis=1) & (f < n_verts).all(axis=1)
    return f[ok]


def _unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Undirected unique edges as (E,2) int64."""
    edges: set[tuple[int, int]] = set()
    for row in np.asarray(faces, dtype=np.int64):
        a, b, c = int(row[0]), int(row[1]), int(row[2])
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edges.add((u, v))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(edges), dtype=np.int64)


def _subsample_edges(edges: np.ndarray, max_edges: int) -> np.ndarray:
    """Subsample edges to max_edges for smaller HTML."""
    n = int(edges.shape[0])
    cap = int(max_edges)
    if cap < 1 or n <= cap:
        return edges
    pos = np.linspace(0, n - 1, num=cap, dtype=np.float64)
    idx = np.unique(np.clip(np.round(pos).astype(np.int64), 0, n - 1))
    out = edges[idx]
    if out.shape[0] > cap:
        out = out[:cap]
    return out


def _cap_frame_indices(frame_indices: list[int], max_frames: int) -> list[int]:
    if max_frames < 1 or len(frame_indices) <= max_frames:
        return frame_indices
    pos = np.linspace(0, len(frame_indices) - 1, num=max_frames, dtype=np.float64)
    idx = np.unique(np.clip(np.round(pos).astype(np.int64), 0, len(frame_indices) - 1))
    return [frame_indices[i] for i in idx]


def _wireframe_line_coords(v: np.ndarray, edges: np.ndarray) -> tuple[list, list, list]:
    """Plotly line trace: segment breaks use None."""
    xe: list[float | None] = []
    ye: list[float | None] = []
    ze: list[float | None] = []
    if edges.size == 0 or v.shape[0] == 0:
        return xe, ye, ze
    nv = v.shape[0]
    for e0, e1 in edges:
        e0i, e1i = int(e0), int(e1)
        if 0 <= e0i < nv and 0 <= e1i < nv:
            xe.extend((float(v[e0i, 0]), float(v[e1i, 0]), None))
            ye.extend((float(v[e0i, 1]), float(v[e1i, 1]), None))
            ze.extend((float(v[e0i, 2]), float(v[e1i, 2]), None))
    return xe, ye, ze


def _subject_offsets_x(
    n_sub: int,
    V: np.ndarray,
    ci: int,
    fi_ref: int,
    n_v: int,
    *,
    fixed_gap: float | None,
) -> np.ndarray:
    """Per-subject translation along x (meters / model units); centers the group at 0."""
    offsets = np.zeros((n_sub, 3), dtype=np.float64)
    if n_sub <= 1:
        return offsets
    chunks: list[np.ndarray] = []
    for s in range(n_sub):
        raw = np.asarray(V[s, ci, fi_ref, :, :3], dtype=np.float64).reshape(n_v, 3)
        m = np.isfinite(raw).all(axis=1)
        if np.any(m):
            chunks.append(raw[m])
    if not chunks:
        return offsets
    allv = np.vstack(chunks)
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    extent = float(np.linalg.norm(hi - lo))
    if fixed_gap is not None and fixed_gap > 0:
        gap = float(fixed_gap)
    else:
        gap = max(extent * 0.26, 1e-4)
    for s in range(n_sub):
        offsets[s, 0] = (float(s) - 0.5 * float(n_sub - 1)) * gap
    return offsets


def build_interactive_mesh_figure(
    V: np.ndarray,
    faces: np.ndarray,
    *,
    used_world: bool,
    camera_index: int,
    frame_stride: int,
    fps: int,
    subject_spacing: float | None,
    line_width: int,
    max_edges: int,
    max_frames: int,
) -> object:
    import plotly.graph_objects as go

    ci = _camera_axis_index(V, used_world, camera_index)
    n_sub, _, n_frames, n_v, _ = V.shape
    if n_v == 0 or n_frames == 0:
        raise ValueError("Empty vertices array (n_v or n_frames is 0)")

    stride = max(1, int(frame_stride))
    frame_indices = list(range(0, n_frames, stride))
    if not frame_indices:
        frame_indices = [0]
    n_frames_after_stride = len(frame_indices)
    frame_indices = _cap_frame_indices(frame_indices, int(max_frames))

    f0 = _clamp_faces(faces, n_v)
    if f0.shape[0] == 0:
        raise ValueError("No valid faces after clamping to vertex count")
    edges_full = _unique_edges_from_faces(f0)
    if edges_full.shape[0] == 0:
        raise ValueError("No edges derived from faces")
    n_edges_full = int(edges_full.shape[0])
    edges = _subsample_edges(edges_full, int(max_edges))
    n_edges_use = int(edges.shape[0])
    if n_edges_use < n_edges_full or len(frame_indices) < n_frames_after_stride:
        logging.info(
            "Interactive mesh caps: edges %d -> %d; frames %d -> %d",
            n_edges_full,
            n_edges_use,
            n_frames_after_stride,
            len(frame_indices),
        )

    fi_ref = int(frame_indices[0])
    offsets = _subject_offsets_x(n_sub, V, ci, fi_ref, n_v, fixed_gap=subject_spacing)

    lw = max(1, min(int(line_width), 6))

    def traces_for_frame(fi: int) -> list:
        out = []
        for s in range(n_sub):
            raw = np.asarray(V[s, ci, fi, :, :3], dtype=np.float64).reshape(n_v, 3)
            v = _fill_nan_vertices(raw) + offsets[s : s + 1, :]
            xe, ye, ze = _wireframe_line_coords(v, edges)
            color = _SUBJECT_LINE_COLORS[s % len(_SUBJECT_LINE_COLORS)]
            empty_wire = len(xe) == 0
            if empty_wire:
                xe, ye, ze = [0.0, None], [0.0, None], [0.0, None]
            out.append(
                go.Scatter3d(
                    x=xe,
                    y=ye,
                    z=ze,
                    mode="lines",
                    name=f"subject_{s}",
                    line=dict(color=color, width=lw if not empty_wire else 0),
                    opacity=0.0 if empty_wire else 1.0,
                    showlegend=(fi == frame_indices[0]) and not empty_wire,
                    hoverinfo="skip",
                )
            )
        return out

    init = traces_for_frame(frame_indices[0])
    frames = [go.Frame(data=traces_for_frame(fi), name=str(fi)) for fi in frame_indices]

    fig = go.Figure(data=init, frames=frames)
    axis_kw = dict(
        showbackground=True,
        backgroundcolor="rgb(252, 252, 252)",
        gridcolor="rgb(235, 235, 235)",
        showgrid=True,
        zeroline=False,
    )
    fig.update_layout(
        title=dict(
            text="SAM 3D Body — interactive mesh wireframe (drag to rotate; slider / play for time)",
            x=0.5,
        ),
        paper_bgcolor="white",
        scene=dict(
            xaxis=dict(title="x", **axis_kw),
            yaxis=dict(title="y", **axis_kw),
            zaxis=dict(title="z", **axis_kw),
            aspectmode="data",
            bgcolor="rgb(255, 255, 255)",
            camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.65, y=-1.65, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode=False,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        sliders=[
            {
                "active": 0,
                "pad": {"t": 40},
                "steps": [
                    {
                        "args": [
                            [str(fi)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": f"{fi:04d}",
                        "method": "animate",
                    }
                    for fi in frame_indices
                ],
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "y": 0,
                "x": 1.02,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": max(1, int(1000 / max(int(fps), 1))),
                                    "redraw": True,
                                },
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return fig


def write_interactive_mesh_html(
    npz_path: Path,
    out_dir: Path,
    *,
    output_name: str = "sam_3d_body_interactive_mesh.html",
    prefer_world: bool = True,
    camera_index: int = 0,
    frame_stride: int = 1,
    fps: int = 30,
    subject_spacing: float | None = None,
    line_width: int = 1,
    max_edges: int = 12_000,
    max_frames: int = 48,
    plotlyjs_cdn: bool = False,
) -> Path:
    V, faces, used_world = _load_vertices_and_faces(npz_path, prefer_world=prefer_world)
    fig = build_interactive_mesh_figure(
        V,
        faces,
        used_world=used_world,
        camera_index=camera_index,
        frame_stride=frame_stride,
        fps=fps,
        subject_spacing=subject_spacing,
        line_width=line_width,
        max_edges=max_edges,
        max_frames=max_frames,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    js = "cdn" if plotlyjs_cdn else True
    fig.write_html(str(out_path), include_plotlyjs=js, full_html=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True, help="Path to body_mesh sam_3d_body.npz")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory (created if missing)")
    p.add_argument(
        "--output-name",
        type=str,
        default="sam_3d_body_interactive_mesh.html",
        help="HTML filename inside out-dir",
    )
    p.add_argument(
        "--prefer-world",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vertices_world when present (default: true)",
    )
    p.add_argument("--camera-index", type=int, default=0, help="Camera slice of vertices when not using world")
    p.add_argument("--stride", type=int, default=1, help="Use every N-th frame to shrink HTML size")
    p.add_argument("--fps", type=int, default=30, help="Target FPS hint for the Play button")
    p.add_argument(
        "--subject-spacing",
        type=float,
        default=None,
        help="Fixed gap along +x between subject centers (model units). Omit for auto from bbox.",
    )
    p.add_argument(
        "--line-width",
        type=int,
        default=1,
        help="Wireframe line width (1–6)",
    )
    p.add_argument(
        "--max-edges",
        type=int,
        default=12_000,
        help="Cap unique mesh edges in the wireframe (prevents multi-hundred-MB HTML).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=48,
        help="Max animation keyframes after --stride (uniform subsample).",
    )
    p.add_argument(
        "--embed-plotlyjs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bundle plotly.js in the HTML (default). Use --no-embed-plotlyjs for CDN (needs network).",
    )
    args = p.parse_args(argv)

    if not args.npz.is_file():
        logging.error("NPZ not found: %s", args.npz)
        return 1
    try:
        out = write_interactive_mesh_html(
            args.npz,
            args.out_dir,
            output_name=args.output_name,
            prefer_world=bool(args.prefer_world),
            camera_index=int(args.camera_index),
            frame_stride=int(args.stride),
            fps=int(args.fps),
            subject_spacing=args.subject_spacing,
            line_width=int(args.line_width),
            max_edges=int(args.max_edges),
            max_frames=int(args.max_frames),
            plotlyjs_cdn=not bool(args.embed_plotlyjs),
        )
    except ImportError as e:
        logging.error("Plotly is required: pip install plotly (%s)", e)
        return 1
    except Exception as e:
        logging.error("Failed to build interactive mesh HTML: %s", e)
        return 1
    logging.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
