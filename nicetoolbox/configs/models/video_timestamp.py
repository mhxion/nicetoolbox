import re
from typing import Annotated

from pydantic import BeforeValidator

# Accepts both colons (00:01:30) and dashes (00-01-30) as separators
_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<hours>\d{2})(?P<sep>[:\-])(?P<minutes>\d{2})(?P=sep)(?P<seconds>\d{2})" r"(?:\.(?P<milliseconds>\d{3}))?$"
)
ALLOWED_FORMATS_MSG = "'HH-MM-SS', 'HH-MM-SS.mmm', 'HH:MM:SS' or 'HH:MM:SS.mmm'"  # only for error message


def _validate_video_timestamp(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected a string in following format {ALLOWED_FORMATS_MSG}, got {type(value).__name__}")

    match = _TIMESTAMP_PATTERN.match(value)
    if match is None:
        raise ValueError(f"Invalid timestamp format: '{value}'. Expected {ALLOWED_FORMATS_MSG}")

    minutes = int(match["minutes"])
    seconds = int(match["seconds"])

    if minutes > 59:
        raise ValueError(f"Minutes must be 0-59, got {minutes}")
    if seconds > 59:
        raise ValueError(f"Seconds must be 0-59, got {seconds}")

    # timestamps placeholders can be used in path, but colons aren't allowed in windows paths
    # so we normalize colons to dashes
    return value.replace(":", "-")


def timestamp_to_frame_index(value: int | str, fps: int) -> int:
    """Convert a video timestamp string or frame index int to a frame index."""
    if isinstance(value, str):
        match = _TIMESTAMP_PATTERN.match(value)
        if match is None:
            raise ValueError(f"Invalid timestamp format: '{value}'")

        hours = int(match["hours"])
        minutes = int(match["minutes"])
        seconds = int(match["seconds"])
        milliseconds = int(match["milliseconds"]) if match["milliseconds"] else 0

        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        return int(total_seconds * fps)
    return value


# Pydantic type: validates format and normalizes to dash-separated string
VideoTimestamp = Annotated[str, BeforeValidator(_validate_video_timestamp)]
