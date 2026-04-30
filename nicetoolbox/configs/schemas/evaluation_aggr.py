from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from pydantic import BaseModel, model_validator

AGG_REGISTRY: dict[str, str | Callable[[pd.Series], float]] = {
    "min": "min",
    "mean": "mean",
    "max": "max",
    "std": "std",
    "sum": "sum",
    "count": "count",
    "q90": lambda s: s.quantile(0.9),
    "q25": lambda s: s.quantile(0.25),
    "median": lambda s: s.quantile(0.5),
    "q75": lambda s: s.quantile(0.75),
    "cv": lambda s: (s.std() / s.mean()) if s.mean() != 0 else float("nan"),
    "one_minus_mean": lambda s: 1.0 - s.mean(),
}


class AggEntry(BaseModel):
    """A single aggregation function with an optional output column name."""

    fn: str
    name: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: object) -> object:
        if isinstance(v, str):
            return {"fn": v}
        return v

    @property
    def output_name(self) -> str:
        return self.name if self.name is not None else self.fn


class AggSpec(BaseModel):
    """List of aggregation functions, each with an optional custom output column name."""

    entries: list[AggEntry]

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: object) -> object:
        if isinstance(v, list):
            return {"entries": v}
        return v

    @classmethod
    def of_type(cls, *fns: str, **named: str) -> AggSpec:
        """
        Positional args are function names with default output names.
        Keyword args map fn_name=output_name for custom column names.

            AggSpec.of("mean", "std")
            AggSpec.of("std", mean="detection_rate")
            AggSpec.of("min", "max", mean="avg_detection_rate")
        """
        entries = [AggEntry(fn=fn) for fn in fns]
        entries += [AggEntry(fn=fn, name=out) for fn, out in named.items()]
        return cls(entries=entries)

    @model_validator(mode="after")
    def _validate_fns(self) -> AggSpec:
        unknown = {e.fn for e in self.entries} - AGG_REGISTRY.keys()
        if unknown:
            raise ValueError(f"Unknown aggregation function(s): {unknown}. Available: {set(AGG_REGISTRY)}")
        return self

    def fn_names(self) -> list[str]:
        """Return the raw aggregation function names (keys into AGG_REGISTRY)."""
        return [e.fn for e in self.entries]

    def resolve(self) -> dict[str, str | Callable[[pd.Series], float]]:
        """Return {output_name: agg_fn} ready to pass as **kwargs to DataFrame.agg()."""
        return {e.output_name: AGG_REGISTRY[e.fn] for e in self.entries}
