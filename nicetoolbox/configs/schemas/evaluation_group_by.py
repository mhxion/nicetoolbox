from pydantic import BaseModel, model_validator

_ALWAYS_POOLED = frozenset({"frame"})

_LEVEL_REQUIRES: dict[str, set[str]] = {
    "session": {"dataset"},
    "sequence": {"dataset", "session"},
    "subsequence": {"dataset", "session", "sequence"},
}


class GroupBySpec(BaseModel):
    dims: list[str] | None  # None = wildcard ("*"), [] = pool everything

    @property
    def is_wildcard(self) -> bool:
        return self.dims is None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: object) -> object:
        if isinstance(v, str):
            return {"dims": None if v == "*" else [v]}
        if isinstance(v, list):
            return {"dims": v}
        if v is None:
            return {"dims": []}
        return v

    @model_validator(mode="after")
    def _extend_hierarchy(self) -> "GroupBySpec":
        if self.is_wildcard or not self.dims:
            return self
        if invalid := _ALWAYS_POOLED & set(self.dims):
            raise ValueError(f"Cannot group by {invalid} — these dimensions are always pooled.")
        extra: set[str] = set()
        for lvl in self.dims:
            extra |= _LEVEL_REQUIRES.get(lvl, set())
        extra -= set(self.dims)
        if extra:
            self.dims = [*extra, *self.dims]
        return self

    def resolve(self, available_levels: list[str]) -> list[str]:
        """Expand wildcard against available levels, or return dims as-is."""
        if self.is_wildcard:
            return [lvl for lvl in available_levels if lvl not in _ALWAYS_POOLED]
        return self.dims

    def intersect(self, available_levels: list[str]) -> list[str]:
        """Like resolve, but always returns intersection with available_levels."""
        return [lvl for lvl in self.resolve(available_levels) if lvl in available_levels]

    def contains(self, *dims: str) -> bool:
        """Return True if all given dims are included (wildcard always passes)."""
        return self.is_wildcard or set(dims).issubset(self.dims)
