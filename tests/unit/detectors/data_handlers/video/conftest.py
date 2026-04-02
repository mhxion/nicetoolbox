from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from nicetoolbox.detectors.data_handlers.video_handler import FILENAME_TEMPLATE, VideoDataHandler

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------
CONFIG_FPS = 30
CONFIG_START = 0
CONFIG_STOP = 100

VIDEO_FPS = 30.0
VIDEO_FRAMES = 300
VIDEO_DURATION_IN_SEC = 10

# ---------------------------------------------------------------------------
# Video processing patches
# ---------------------------------------------------------------------------

VID_MODULE = "nicetoolbox.detectors.data_handlers.video_handler.vid"


@dataclass
class FakeVideoInfo:
    video_path: Path
    codec: str = "h264"
    fps: Optional[float] = VIDEO_FPS
    frames: Optional[int] = VIDEO_FRAMES
    width: int = 1920
    height: int = 1080
    duration_in_sec: Optional[int] = VIDEO_DURATION_IN_SEC


@pytest.fixture()
def default_vid_patches():
    """Patches vid.* functions used by VideoDataHandler with sensible defaults."""
    video_path = Path("fake.mp4")
    with (
        patch(f"{VID_MODULE}.probe_video", return_value={}) as mock_probe,
        patch(f"{VID_MODULE}.json_to_video_info", return_value=FakeVideoInfo(video_path)) as mock_info,
        patch(f"{VID_MODULE}.split_into_frames") as mock_split,
    ):
        yield {
            "probe_video": mock_probe,
            "json_to_video_info": mock_info,
            "split_into_frames": mock_split,
        }


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_sequence_context(
    camera_names: list[str],
    fps: int = CONFIG_FPS,
    video_start: int | str = CONFIG_START,
    video_length: int | str = CONFIG_STOP,
    session_id: str = "session_01",
    sequence_id: str = "seq_01",
    start_frame_index: int = 0,
):
    ctx = MagicMock()
    ctx.fps = fps
    ctx.video_start = video_start
    ctx.video_length = video_length
    ctx.all_camera_names = camera_names
    ctx.session_id = session_id
    ctx.sequence_id = sequence_id
    ctx.subjects_descr = ["subject_1"]
    ctx.dataset_properties = MagicMock()
    ctx.dataset_properties.start_frame_index = start_frame_index
    return ctx


def make_io(
    tmp_path: Path,
    data_source_folder: Path,
    calibration_file: Path = None,
):
    io = MagicMock()
    nice_input = tmp_path / "nice_input"
    nice_input.mkdir(parents=True, exist_ok=True)
    io.nice_input_folder = nice_input
    io.get_data_source_folder.return_value = data_source_folder
    io.get_calibration_file.return_value = calibration_file
    return io


def make_frames_cache(io, ctx):
    for cam in ctx.all_camera_names:
        frames_dir = io.nice_input_folder / cam / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(ctx.video_start, ctx.video_start + ctx.video_length):
            (frames_dir / FILENAME_TEMPLATE.format(idx=idx)).write_bytes(b"\x89PNG")


def make_simple_handler(tmp_path: Path, cameras: list[str], **ctx_kwargs):
    """Create a VideoDataHandler with a flat data source layout (one {cam}.mp4 per camera)."""
    data_source_folder = tmp_path / "data_source"
    for cam in cameras:
        video_path = data_source_folder / f"{cam}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()
    ctx = make_sequence_context(cameras, **ctx_kwargs)
    io = make_io(tmp_path, data_source_folder)
    return VideoDataHandler(io=io, sequence_context=ctx), ctx, io


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def assert_handler_output(
    handler: VideoDataHandler,
    cameras: list[str],
    fps: int = CONFIG_FPS,
    start_frame: int = CONFIG_START,
    length_frames: int = CONFIG_STOP,
    is_available: bool = True,
    calibration_keys: set[str] | None = None,
    calibration_is_none: bool = False,
):
    assert handler.fps == fps
    assert handler.start_frame == start_frame
    assert handler.length_frames == length_frames
    assert handler.is_available is is_available

    if calibration_is_none:
        assert handler.calibration is None
    elif calibration_keys is not None:
        assert handler.calibration is not None
        assert set(handler.calibration.keys()) == calibration_keys

    recipe = handler.get_recipe()
    assert recipe.camera_names == sorted(cameras)
    assert "{camera}/frames/" in recipe.filename_template
    assert recipe.range_start == start_frame
    assert recipe.range_end == start_frame + length_frames
    assert recipe.step == 1
