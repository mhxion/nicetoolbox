from typing import Annotated

from pydantic import AfterValidator


def _reject_wildcard(v: str) -> str:
    if v == "*":
        raise ValueError("Wildcards are not allowed, must be an explicit value.")
    return v


NoWildcardStr = Annotated[str, AfterValidator(_reject_wildcard)]
