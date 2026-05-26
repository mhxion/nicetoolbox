from pathlib import Path

from pydantic import BaseModel, NonNegativeInt

from ...configs.models.video_timestamp import VideoTimestamp


class NapariSequenceConfig(BaseModel):
    input: Path
    output: Path
    video: Path
    start: NonNegativeInt | VideoTimestamp
    end: int | VideoTimestamp  # -1 means end of video
    reset_frames: bool


class NapariImportBodyJointsConfig(BaseModel):
    log_level: str
    log_file_path: Path
    export_csv: bool

    run: dict[str, NapariSequenceConfig]

    subjects: dict[str, str] = {}
    cameras: dict[str, str] = {}
