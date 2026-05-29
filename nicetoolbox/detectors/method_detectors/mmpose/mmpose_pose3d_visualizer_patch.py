"""
Runtime patches for MMPose Pose3dLocalVisualizer used by NICE MotionBERT 3D vis.

1. RGB reshape: upstream multiplies figure width by num_instances twice vs the Agg
   buffer — fixed via canvas.get_width_height().

2. Optional NICE multi-person layout (when visualizer._nicetoolbox_multi_3d_layout
   is True, set from ``mmpose_inference_3d`` (3D subprocess) before exporting frames under
   <result>/<algorithm>/visualization/<camera>/):
   - No stitched 2D keypoint panel on the left (same as draw_2d=False, enforced here
     because Pose3DInferencer does not forward that kw).
   - Two (or more) people are drawn in one 3D axes with a shared box and view.
   - Horizontal separation along 3D X only: ΔX ∝ pelvis_sep_scale * (pelvis Δu in
     pixels) * (2 * axis_limit / image_width) so wider image spacing maps to wider 3D
     spacing; pose orientation is unchanged (translation only). Offsets are anchored at
     the image-right pelvis (rightmost instance has ΔX=0) so image-left appears
     on the viewer's left under the default axis_azimuth (anchoring at left mirrored
     the pair in 3D).
   - Left-right stability: instances are ordered by COCO hip-midpoint u (smaller
     x = left in the image) each frame. This does not fix identity swap when people
     cross; it only fixes detector list order flicker when positions are stable.

Submodule mmpose stays unmodified.
"""

from __future__ import annotations

import logging

import mmcv
import numpy as np
from matplotlib import pyplot as plt
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample

_LOG = logging.getLogger(__name__)


def _pelvis_x_pixels(kp_row: np.ndarray) -> float:
    """Horizontal reference in the image from 2D keypoints (COCO 17: hips 11, 12)."""
    kp = np.asarray(kp_row, dtype=np.float64)
    if kp.shape[0] > 12:
        lh, rh = kp[11, :2], kp[12, :2]
        if np.all(np.isfinite(lh)) and np.all(np.isfinite(rh)):
            return float((lh[0] + rh[0]) * 0.5)
    valid = np.isfinite(kp[:, 0])
    if np.any(valid):
        return float(np.nanmean(kp[valid, 0]))
    return float("nan")


def apply_pose3d_visualizer_rgb_reshape_fix() -> None:
    """Idempotent: patch add_datasample and _draw_3d_data_samples."""
    try:
        from mmpose.visualization.local_visualizer_3d import Pose3dLocalVisualizer
    except ImportError:
        return
    if getattr(Pose3dLocalVisualizer, "_nicetoolbox_pose3d_patched", False):
        return
    Pose3dLocalVisualizer._nicetoolbox_orig_add_datasample = Pose3dLocalVisualizer.add_datasample
    Pose3dLocalVisualizer.add_datasample = _add_datasample_nicetoolbox
    Pose3dLocalVisualizer._draw_3d_data_samples = _draw_3d_data_samples_nicetoolbox
    Pose3dLocalVisualizer._nicetoolbox_pose3d_patched = True


def _add_datasample_nicetoolbox(
    self,
    name: str,
    image: np.ndarray,
    data_sample: PoseDataSample,
    det_data_sample=None,
    draw_gt: bool = True,
    draw_pred: bool = True,
    draw_2d: bool = True,
    draw_bbox: bool = False,
    show_kpt_idx: bool = False,
    skeleton_style: str = "mmpose",
    dataset_2d: str = "coco",
    dataset_3d: str = "h36m",
    convert_keypoint: bool = True,
    axis_azimuth: float = 70,
    axis_limit: float = 1.7,
    axis_dist: float = 10.0,
    axis_elev: float = 15.0,
    num_instances: int = -1,
    show: bool = False,
    wait_time: float = 0,
    out_file=None,
    kpt_thr: float = 0.3,
    step: int = 0,
):
    nice = bool(getattr(self, "_nicetoolbox_multi_3d_layout", False))
    if not nice:
        return type(self)._nicetoolbox_orig_add_datasample(
            self,
            name,
            image,
            data_sample,
            det_data_sample,
            draw_gt=draw_gt,
            draw_pred=draw_pred,
            draw_2d=draw_2d,
            draw_bbox=draw_bbox,
            show_kpt_idx=show_kpt_idx,
            skeleton_style=skeleton_style,
            dataset_2d=dataset_2d,
            dataset_3d=dataset_3d,
            convert_keypoint=convert_keypoint,
            axis_azimuth=axis_azimuth,
            axis_limit=axis_limit,
            axis_dist=axis_dist,
            axis_elev=axis_elev,
            num_instances=num_instances,
            show=show,
            wait_time=wait_time,
            out_file=out_file,
            kpt_thr=kpt_thr,
            step=step,
        )

    draw_2d = False
    self._nicetoolbox_lr_order = None
    self._nicetoolbox_pelvis_dx_px = None
    self._nicetoolbox_nice_image_rgb = image

    try:
        if (
            det_data_sample is not None
            and "pred_instances" in det_data_sample
            and "pred_instances" in data_sample
            and "keypoints" in data_sample.pred_instances
            and "keypoints" in det_data_sample.pred_instances
        ):
            det_pi = det_data_sample.pred_instances
            n_d, n_3 = len(det_pi), len(data_sample.pred_instances)
            if n_d == n_3 and n_3 >= 2:
                px = np.array([_pelvis_x_pixels(det_pi.keypoints[i]) for i in range(n_d)])
                bad = ~np.isfinite(px)
                if np.any(bad):
                    px[bad] = np.linspace(0.0, float(n_d), num=n_d, endpoint=False)[bad]
                order = np.argsort(px)
                px_s = px[order]
                self._nicetoolbox_lr_order = order
                # Anchor at image-right (smaller 3D X for that person) so left/right match
                # the input frame under default 3D azimuth; px_s[-1] - px_s is >= 0.
                self._nicetoolbox_pelvis_dx_px = px_s[-1] - px_s
            elif n_d != n_3 and n_3 >= 2:
                _LOG.debug(
                    "NICE 3D vis: det instances %s != 3D instances %s; skip LR ordering.",
                    n_d,
                    n_3,
                )

        scores_2d = None

        pred_img_data = self._draw_3d_data_samples(
            image.copy(),
            data_sample,
            draw_gt=draw_gt,
            num_instances=num_instances,
            axis_azimuth=axis_azimuth,
            axis_limit=axis_limit,
            show_kpt_idx=show_kpt_idx,
            axis_dist=axis_dist,
            axis_elev=axis_elev,
            scores_2d=scores_2d,
        )

        drawn_img = pred_img_data
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

        return self.get_image()
    finally:
        self._nicetoolbox_lr_order = None
        self._nicetoolbox_pelvis_dx_px = None
        self._nicetoolbox_nice_image_rgb = None


def _draw_3d_data_samples_nicetoolbox(
    self,
    image: np.ndarray,
    pose_samples: PoseDataSample,
    draw_gt: bool = True,
    kpt_thr: float = 0.3,
    num_instances=-1,
    axis_azimuth: float = 70,
    axis_limit: float = 1.7,
    axis_dist: float = 10.0,
    axis_elev: float = 15.0,
    show_kpt_idx: bool = False,
    scores_2d: np.ndarray | None = None,
):
    vis_width = max(image.shape)
    vis_height = vis_width

    if "pred_instances" in pose_samples:
        pred_instances = pose_samples.pred_instances
    else:
        pred_instances = InstanceData()
    if num_instances < 0:
        if "keypoints" in pred_instances:
            num_instances = len(pred_instances)
        else:
            num_instances = 0
    else:
        if len(pred_instances) > num_instances:
            pred_instances_ = InstanceData()
            for k in pred_instances:
                new_val = pred_instances[k][:num_instances]
                pred_instances_.set_field(new_val, k)
            pred_instances = pred_instances_
        elif num_instances < len(pred_instances):
            num_instances = len(pred_instances)

    lr_order = getattr(self, "_nicetoolbox_lr_order", None)
    dx_px = getattr(self, "_nicetoolbox_pelvis_dx_px", None)
    nice = bool(getattr(self, "_nicetoolbox_multi_3d_layout", False))
    pelvis_scale = float(getattr(self, "_nicetoolbox_pelvis_sep_scale", 1.0))

    use_shared_multi = (
        nice
        and lr_order is not None
        and dx_px is not None
        and len(lr_order) == num_instances
        and num_instances >= 2
        and not draw_gt
    )

    if use_shared_multi:
        plt.ioff()
        fig = plt.figure(figsize=(vis_width * 0.01, vis_height * 0.01))
        _draw_predictions_shared_axes(
            self,
            pred_instances,
            scores_2d,
            kpt_thr,
            axis_azimuth,
            axis_elev,
            axis_dist,
            axis_limit,
            show_kpt_idx,
            fig,
            image,
            lr_order,
            dx_px,
            pelvis_scale,
        )
    else:
        num_fig = num_instances
        if draw_gt:
            vis_width *= 2
            num_fig *= 2

        plt.ioff()
        fig = plt.figure(figsize=(vis_width * num_instances * 0.01, vis_height * 0.01))

        def _draw_3d_instances_kpts(
            keypoints,
            scores,
            scores_2d,
            _keypoints_visible,
            fig_idx,
            show_kpt_idx,
            title=None,
        ):
            for idx, (kpts, score, score_2d) in enumerate(zip(keypoints, scores, scores_2d)):
                valid = (score >= kpt_thr) & (score_2d >= kpt_thr) & np.any(~np.isnan(kpts), axis=-1)

                kpts_valid = kpts[valid]
                ax = fig.add_subplot(1, num_fig, fig_idx * (idx + 1), projection="3d")
                ax.view_init(elev=axis_elev, azim=axis_azimuth)
                ax.set_aspect("auto")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                if title:
                    ax.set_title(f"{title} ({idx})")
                ax.dist = axis_dist

                x_c = np.mean(kpts_valid[:, 0]) if valid.any() else 0
                y_c = np.mean(kpts_valid[:, 1]) if valid.any() else 0
                z_c = np.mean(kpts_valid[:, 2]) if valid.any() else 0

                ax.set_xlim3d([x_c - axis_limit / 2, x_c + axis_limit / 2])
                ax.set_ylim3d([y_c - axis_limit / 2, y_c + axis_limit / 2])
                ax.set_zlim3d([min(0, z_c - axis_limit / 2), z_c + axis_limit / 2])

                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f"the length of kpt_color ({len(self.kpt_color)}) does not matches "
                        f"that of keypoints ({len(kpts)})"
                    )

                x_3d, y_3d, z_3d = np.split(kpts_valid[:, :3], [1, 2], axis=1)

                kpt_color = kpt_color[valid] / 255.0

                ax.scatter(x_3d, y_3d, z_3d, marker="o", c=kpt_color)

                if show_kpt_idx:
                    for kpt_idx in range(len(x_3d)):
                        ax.text(x_3d[kpt_idx][0], y_3d[kpt_idx][0], z_3d[kpt_idx][0], str(kpt_idx))

                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f"the length of link_color ({len(self.link_color)}) does not matches "
                            f"that of skeleton ({len(self.skeleton)})"
                        )

                    for sk_id, sk in enumerate(self.skeleton):
                        sk_indices = [_i for _i in sk]
                        xs_3d = kpts[sk_indices, 0]
                        ys_3d = kpts[sk_indices, 1]
                        zs_3d = kpts[sk_indices, 2]
                        kpt_score = score[sk_indices]
                        kpt_score_2d = score_2d[sk_indices]
                        if kpt_score.min() > kpt_thr and kpt_score_2d.min() > kpt_thr:
                            _color = link_color[sk_id] / 255.0
                            ax.plot(xs_3d, ys_3d, zs_3d, color=_color, zdir="z")

        if "keypoints" in pred_instances:
            keypoints = pred_instances.get("keypoints", pred_instances.keypoints)

            if "keypoint_scores" in pred_instances:
                scores = pred_instances.keypoint_scores
            else:
                scores = np.ones(keypoints.shape[:-1])

            if scores_2d is None:
                scores_2d = np.ones(keypoints.shape[:-1])

            if "keypoints_visible" in pred_instances:
                keypoints_visible = pred_instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            _draw_3d_instances_kpts(keypoints, scores, scores_2d, keypoints_visible, 1, show_kpt_idx, "Prediction")

        if draw_gt and "gt_instances" in pose_samples:
            gt_instances = pose_samples.gt_instances
            if "lifting_target" in gt_instances:
                keypoints = gt_instances.get("lifting_target", gt_instances.lifting_target)
                scores = np.ones(keypoints.shape[:-1])

                if "lifting_target_visible" in gt_instances:
                    keypoints_visible = gt_instances.lifting_target_visible
                else:
                    keypoints_visible = np.ones(keypoints.shape[:-1])
            elif "keypoints_gt" in gt_instances:
                keypoints = gt_instances.get("keypoints_gt", gt_instances.keypoints_gt)
                scores = np.ones(keypoints.shape[:-1])

                if "keypoints_visible" in gt_instances:
                    keypoints_visible = gt_instances.keypoints_visible
                else:
                    keypoints_visible = np.ones(keypoints.shape[:-1])
            else:
                raise ValueError(
                    "to visualize ground truth results, data sample must contain " '"lifting_target" or "keypoints_gt"'
                )

            if scores_2d is None:
                scores_2d = np.ones(keypoints.shape[:-1])

            _draw_3d_instances_kpts(keypoints, scores, scores_2d, keypoints_visible, 2, show_kpt_idx, "Ground Truth")

    fig.tight_layout()
    fig.canvas.draw()

    pred_img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

    if not pred_img_data.any():
        pred_img_data = np.full((vis_height, vis_width, 3), 255)
    else:
        w_px, h_px = fig.canvas.get_width_height()
        expected = w_px * h_px * 3
        if pred_img_data.size != expected:
            raise ValueError(
                f"3D vis RGB buffer size {pred_img_data.size} does not match "
                f"canvas {w_px}x{h_px} ({expected} bytes)."
            )
        pred_img_data = pred_img_data.reshape(h_px, w_px, 3)

    plt.close(fig)

    return pred_img_data


def _draw_predictions_shared_axes(
    self,
    pred_instances: InstanceData,
    scores_2d,
    kpt_thr: float,
    axis_azimuth: float,
    axis_elev: float,
    axis_dist: float,
    axis_limit: float,
    show_kpt_idx: bool,
    fig,
    image: np.ndarray,
    lr_order: np.ndarray,
    dx_px: np.ndarray,
    pelvis_scale: float,
) -> None:
    keypoints = np.asarray(pred_instances.get("keypoints", pred_instances.keypoints))
    if "keypoint_scores" in pred_instances:
        scores = np.asarray(pred_instances.keypoint_scores)
    else:
        scores = np.ones(keypoints.shape[:-1])
    if scores_2d is None:
        scores_2d = np.ones(keypoints.shape[:-1])
    else:
        scores_2d = np.asarray(scores_2d)

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=axis_elev, azim=axis_azimuth)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = axis_dist

    img_w = max(int(image.shape[1]), 1)
    mm_per_px = (2.0 * axis_limit / img_w) * pelvis_scale
    offsets_x = mm_per_px * np.asarray(dx_px, dtype=np.float64)

    all_pts: list[np.ndarray] = []

    for plot_i, src_i in enumerate(lr_order):
        kpts = np.asarray(keypoints[src_i], dtype=np.float64).copy()
        score = np.asarray(scores[src_i])
        score_2d = np.asarray(scores_2d[src_i])
        ox = float(offsets_x[plot_i]) if plot_i < len(offsets_x) else 0.0
        kpts[:, 0] += ox

        valid = (score >= kpt_thr) & (score_2d >= kpt_thr) & np.any(~np.isnan(kpts), axis=-1)
        kpts_valid = kpts[valid]
        if kpts_valid.size:
            all_pts.append(kpts_valid[:, :3])

        if self.kpt_color is None or isinstance(self.kpt_color, str):
            kpt_color = [self.kpt_color] * len(kpts)
        elif len(self.kpt_color) == len(kpts):
            kpt_color = self.kpt_color
        else:
            raise ValueError(
                f"the length of kpt_color ({len(self.kpt_color)}) does not matches " f"that of keypoints ({len(kpts)})"
            )
        kpt_color = np.asarray(kpt_color)[valid] / 255.0

        x_3d, y_3d, z_3d = np.split(kpts_valid[:, :3], [1, 2], axis=1)
        ax.scatter(x_3d, y_3d, z_3d, marker="o", c=kpt_color)

        if show_kpt_idx:
            for kpt_idx in range(len(x_3d)):
                ax.text(x_3d[kpt_idx][0], y_3d[kpt_idx][0], z_3d[kpt_idx][0], str(kpt_idx))

        if self.skeleton is not None and self.link_color is not None:
            if isinstance(self.link_color, str):
                link_color = [self.link_color] * len(self.skeleton)
            elif len(self.link_color) == len(self.skeleton):
                link_color = self.link_color
            else:
                raise ValueError(
                    f"the length of link_color ({len(self.link_color)}) does not matches "
                    f"that of skeleton ({len(self.skeleton)})"
                )
            for sk_id, sk in enumerate(self.skeleton):
                sk_indices = [_i for _i in sk]
                xs_3d = kpts[sk_indices, 0]
                ys_3d = kpts[sk_indices, 1]
                zs_3d = kpts[sk_indices, 2]
                kpt_score = score[sk_indices]
                kpt_score_2d = score_2d[sk_indices]
                if kpt_score.min() > kpt_thr and kpt_score_2d.min() > kpt_thr:
                    _color = link_color[sk_id] / 255.0
                    ax.plot(xs_3d, ys_3d, zs_3d, color=_color, zdir="z")

    if all_pts:
        stack = np.concatenate(all_pts, axis=0)
        valid = np.all(np.isfinite(stack), axis=1)
        if np.any(valid):
            lo = stack[valid].min(axis=0)
            hi = stack[valid].max(axis=0)
            pad = 0.12 * axis_limit
            ax.set_xlim3d([lo[0] - pad, hi[0] + pad])
            ax.set_ylim3d([lo[1] - pad, hi[1] + pad])
            z_lo = min(0.0, float(lo[2] - pad))
            ax.set_zlim3d([z_lo, float(hi[2] + pad)])
    else:
        ax.set_xlim3d([-axis_limit, axis_limit])
        ax.set_ylim3d([-axis_limit, axis_limit])
        ax.set_zlim3d([0, axis_limit])
