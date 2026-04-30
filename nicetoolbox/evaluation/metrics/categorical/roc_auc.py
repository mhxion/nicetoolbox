import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from ....configs.schemas.evaluation_metrics_config import RocAucConfig
from ...data.input_loader import align_arrays, get_meta_type, load_input
from ...data.plots import plot_roc_curves
from ...data.summary import pair_arrays_to_df, resolve_group_levels, split_aligned_arrays
from ..base_metric import BaseMetric
from ..metric_result import FrameResult, MetricResult, PlotResult, SummaryResult


class RocAucMetric(BaseMetric):
    """ROC curve and AUC score per pre-compute group.

    Predictions must be float confidence scores; ground truth must be bool.
    AUC is computed once per group by pooling all frames — same rationale as
    ConfusionMatrixMetric (avoids averaging per-pair scores across groups with
    different support and class balance).
    """

    metric_config: RocAucConfig

    def compute(self) -> MetricResult:
        preds = load_input(self.metric_config.predictions)
        gt = load_input(self.metric_config.ground_truth)
        pairs = align_arrays(preds, gt, self.metric_config.broadcast_single)
        if not pairs:
            raise ValueError(f"Failed to compute ROC AUC '{self.metric_name}': no aligned pred/GT pairs found.")

        df = pair_arrays_to_df(pairs)

        n_before = len(df)
        df = df.dropna(subset=["pred", "gt"])
        n_dropped = n_before - len(df)
        if n_dropped:
            logging.warning(
                f"[{self.metric_name}] Dropped {n_dropped}/{n_before} rows with NaN pred or gt "
                f"({100 * n_dropped / n_before:.1f}%). Missing predictions are excluded."
            )

        _validate_roc_inputs(df, self.metric_name)

        meta_type = get_meta_type([p for p, _ in pairs])
        group_levels = resolve_group_levels(df, meta_type, self.metric_config.compute_group_by)

        # user dims only (exclude always-iterate) — used for short curve labels
        always = meta_type.always_iterate()
        label_levels = [lvl for lvl in group_levels if lvl not in always] or group_levels

        # compute per-group: AUC scalar + curve arrays
        records: list[dict] = []
        curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []

        for keys, group in df[group_levels + ["pred", "gt"]].groupby(group_levels, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)

            key_map = dict(zip(group_levels, keys))
            y_true = group["gt"].to_numpy(dtype=bool)
            y_score = group["pred"].to_numpy(dtype=float)
            if self.metric_config.negate_scores:
                y_score = -y_score

            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                logging.warning(
                    "[%s] Skipping degenerate group %s — only one class present (positives=%d/%d).",
                    self.metric_name,
                    keys,
                    int(y_true.sum()),
                    len(y_true),
                )
                auc = float("nan")
                fpr = tpr = np.array([0.0, 1.0])
            else:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc = float(roc_auc_score(y_true, y_score))

            label = " | ".join(f"{lvl}={key_map[lvl]}" for lvl in label_levels)
            curves.append((label, fpr, tpr, auc))
            records.append({**key_map, "auc": auc, "support": len(y_true)})

        summary = pd.DataFrame(records)

        figures = {}
        if self.metric_config.visualize:
            figures["roc_curves"] = plot_roc_curves(curves, self.metric_name)

        pred_aligned, gt_aligned = split_aligned_arrays(*pairs)
        return MetricResult(
            self.metric_name,
            frames=FrameResult({"predictions": pred_aligned, "ground_truth": gt_aligned}),
            summary=SummaryResult({"summary": summary}),
            plots=PlotResult(figures),
        )


def _validate_roc_inputs(df: pd.DataFrame, metric_name: str) -> None:
    if not pd.api.types.is_bool_dtype(df["gt"]):
        raise TypeError(f"[{metric_name}] Column 'gt' must be bool, got {df['gt'].dtype}.")
    if not pd.api.types.is_float_dtype(df["pred"]):
        raise TypeError(
            f"[{metric_name}] Column 'pred' must be float (confidence scores), got {df['pred'].dtype}. "
            "Bool predictions belong in confusion_matrix, not roc_auc."
        )
