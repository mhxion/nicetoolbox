from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .evaluation_metrics_config import METRICS_REGISTRY, BaseMetricConfig


class EvaluationConfig(BaseModel):
    """Top-level evaluation configuration loaded from TOML."""

    log_level: str
    log_file_path: Path

    default_experiment: Path
    output_folder: Path

    predictions_mapping: Path

    # subset of metric names to run; "*" means run all
    run_metrics: list[str] | str = "*"
    metrics: dict[str, BaseMetricConfig] = Field(default_factory=dict)

    compare: dict[str, Any] = Field(default_factory=dict)
    summaries: dict[str, Any] = Field(default_factory=dict)

    def metrics_to_run(self) -> dict[str, BaseMetricConfig]:
        """Return the subset of metrics selected by run_metrics."""
        if self.run_metrics == "*":
            return self.metrics
        run = [self.run_metrics] if isinstance(self.run_metrics, str) else self.run_metrics
        unknown = set(run) - self.metrics.keys()
        if unknown:
            raise ValueError(f"run_metrics refers to unknown metrics: {unknown}")
        return {name: self.metrics[name] for name in run}

    @model_validator(mode="after")
    def fill_default_experiment_paths(self) -> "EvaluationConfig":
        for metric in self.metrics.values():
            for block in metric.input_blocks():
                if block.path is None:
                    block.path = self.default_experiment
        return self

    @model_validator(mode="before")
    @classmethod
    def parse_metrics(cls, values):
        values = deepcopy(values)
        metrics_raw = values.get("metrics", {})
        parsed_metrics = METRICS_REGISTRY.parse_dict_by_type(metrics_raw, type_field_name="metric_type")
        for name, cfg in parsed_metrics.items():
            cfg._metric_name = name
        values["metrics"] = parsed_metrics
        return values
