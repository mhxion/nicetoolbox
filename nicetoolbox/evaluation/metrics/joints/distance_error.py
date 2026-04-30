import numpy as np

from ....configs.schemas.evaluation_aggr import AggSpec
from ....configs.schemas.evaluation_metrics_config import DistanceErrorConfig
from ...data.input_loader import ArrayAxes, LoadedArray, align_arrays, get_meta_type, load_input
from ...data.plots import plot_candle_per_group, plot_frame_line, plot_score
from ...data.summary import aggregate_summary, split_aligned_arrays, summarize_with_group_by
from ..base_metric import BaseMetric
from ..metric_result import FrameResult, MetricResult, PlotResult, SummaryResult


class DistanceErrorMetric(BaseMetric):
    """Per-frame, per-joint distance error between predictions and ground truth.

    Supports L1 and L2 norms. Works with both 2D and 3D keypoints.
    The last coordinate dimension is assumed to be confidence and is dropped.
    """

    metric_config: DistanceErrorConfig

    def compute(self) -> MetricResult:
        # load and align predictions with ground truth (frame intersection)
        preds = load_input(self.metric_config.predictions)
        gt = load_input(self.metric_config.ground_truth)
        pairs = align_arrays(preds, gt, self.metric_config.broadcast_single)
        if not pairs:
            raise ValueError("Failed to compute Distance Error: no aligned prediction/GT pairs found!")

        # compute per-frame, per-joint error
        error_arrays: list[LoadedArray] = []
        for pred, gt in pairs:
            error_arrays.append(self._compute_error(pred, gt))

        meta = get_meta_type(error_arrays)

        # detailed summary (user-configurable aggregations)
        summary = summarize_with_group_by(
            error_arrays,
            self.metric_config.summary_group_by,
            self.metric_config.summary_aggr,
        )

        # score is always mean error regardless of summary_aggr
        mean_summary = summarize_with_group_by(
            error_arrays,
            self.metric_config.summary_group_by,
            AggSpec.of_type("mean"),
        )
        score = aggregate_summary(mean_summary, agg_col="mean", meta_type=meta)

        # optional visualization
        figures = {}
        if self.metric_config.visualize:
            compare_dim = meta.comparable_dim()

            figures = plot_candle_per_group(
                error_arrays,
                x_col="label",
                series_col=compare_dim,
                base_title=self.metric_name,
                y_label="Distance Error",
            )

            # per-frame line chart (mean error over time)
            figures |= plot_frame_line(
                error_arrays,
                series_col=compare_dim,
                base_title=self.metric_name,
                y_label="Distance Error",
            )

            figures["mean_error_score"] = plot_score(
                score,
                x_col=compare_dim,
                y_col="mean",
                title="Mean Distance Error Score",
                x_label=compare_dim,
                y_label="Mean Error (lower is better)",
            )

        pred_aligned, gt_aligned = split_aligned_arrays(*pairs)
        return MetricResult(
            self.metric_name,
            frames=FrameResult(
                {"distance_error": error_arrays, "predictions": pred_aligned, "ground_truth": gt_aligned}
            ),
            plots=PlotResult(figures),
            summary=SummaryResult({"mean_error_score": score, "summary": summary}),
        )

    def _compute_error(self, pred: LoadedArray, gt: LoadedArray) -> LoadedArray:
        """Compute per-frame, per-joint distance error.

        Returns LoadedArray with shape (subjects, cameras, frames, joints).
        """
        # drop last coord (confidence)
        p = pred.data[..., :-1]
        g = gt.data[..., :-1]

        diff = p - g

        if self.metric_config.norm == "l2":
            error = np.linalg.norm(diff, axis=-1)
        else:  # l1
            error = np.sum(np.abs(diff), axis=-1)

        return LoadedArray(
            meta=pred.meta,
            data=error,
            axes=ArrayAxes(
                subjects=pred.axes.subjects,
                cameras=pred.axes.cameras,
                frames=pred.axes.frames,
                labels=pred.axes.labels,
            ),
        )
