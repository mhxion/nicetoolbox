from typing import Literal

from pydantic import BaseModel, PrivateAttr, model_validator

from ..models.models_registry import ModelsRegistry
from .evaluation_aggr import AggSpec
from .evaluation_group_by import GroupBySpec
from .evaluation_input_block import BaseInputBlock, InputBlock

METRICS_REGISTRY = ModelsRegistry()
metric_config = METRICS_REGISTRY.register


class BaseMetricConfig(BaseModel):
    """Base schema for all evaluation metrics."""

    metric_type: str
    _metric_name: str = PrivateAttr()

    def input_blocks(self) -> list[BaseInputBlock]:
        """Return all InputBlock fields on this metric config."""
        result = []
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, BaseInputBlock):
                result.append(value)
        return result


# =============================================================================
# Pose estimation metrics
# =============================================================================


@metric_config("bone_length")
class BoneLengthConfig(BaseMetricConfig):
    visualize: bool = True

    predictions: InputBlock

    summary_group_by: GroupBySpec
    summary_aggr: AggSpec

    @model_validator(mode="after")
    def _require_bone_dims(self) -> "BoneLengthConfig":
        mandatory = {"subject", "sequence", "label"}
        if not self.summary_group_by.contains(*mandatory):
            missing = mandatory - set(self.summary_group_by.dims)
            raise ValueError(
                f"group_by for bone_length must include {missing}. "
                f"Bone length is a per-person, per-sequence, per-bone property — "
                f"pooling across any of these dimensions produces meaningless averages."
            )
        return self


@metric_config("distance_error")
class DistanceErrorConfig(BaseMetricConfig):
    visualize: bool = True
    norm: Literal["l1", "l2"] = "l2"
    broadcast_single: bool = False

    predictions: InputBlock
    ground_truth: InputBlock

    summary_group_by: GroupBySpec
    summary_aggr: AggSpec


@metric_config("missing_points")
class MissingPointsConfig(BaseMetricConfig):
    visualize: bool = True

    predictions: InputBlock
    min_confidence: float | None = None

    missing_points_summary_group_by: GroupBySpec
    missing_points_summary_aggr: AggSpec


# =============================================================================
# Categorical metrics
# =============================================================================


@metric_config("roc_auc")
class RocAucConfig(BaseMetricConfig):
    visualize: bool = True
    broadcast_single: bool = False
    negate_scores: bool = False  # set True when lower score = more positive (e.g. distance)

    predictions: InputBlock  # float confidence scores
    ground_truth: InputBlock  # bool labels

    compute_group_by: GroupBySpec


@metric_config("confusion_matrix")
class ConfusionMatrixConfig(BaseMetricConfig):
    visualize: bool = True
    broadcast_single: bool = False

    predictions: InputBlock
    ground_truth: InputBlock

    compute_group_by: GroupBySpec
