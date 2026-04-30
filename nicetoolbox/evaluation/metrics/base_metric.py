from abc import ABC, abstractmethod

from ...configs.schemas.evaluation_metrics_config import BaseMetricConfig
from ..config_handler import ConfigHandler
from .metric_result import MetricResult


class BaseMetric(ABC):
    """
    Base class for all evaluation metrics.

    Each metric is self-contained: receives its config, loads its own data,
    computes results, and returns a MetricResult.
    """

    metric_name: str
    metric_config: BaseMetricConfig
    config_handler: ConfigHandler

    def __init__(self, metrics_config: BaseMetricConfig, config_handler: ConfigHandler):
        """
        Args:
            metrics_config: Metric-specific configuration.
            config_handler: Shared config handler with project and evaluation configs.
        """
        self.config_handler = config_handler
        self.metric_config = metrics_config
        self.metric_name = metrics_config._metric_name

        self._init_metric()

    def _init_metric(self):  # noqa: B027
        """Optional hook for subclass initialization. Called at the end of __init__."""
        pass

    @abstractmethod
    def compute(self) -> MetricResult:
        """Execute the metric end-to-end: load data, compute, return results.

        Returns:
            MetricResult containing summary tables, per-frame arrays, and plots.
        """
        pass
