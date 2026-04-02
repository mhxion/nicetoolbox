# ruff: noqa: ARG001
import pytest

from nicetoolbox.detectors.data_handlers.video_handler import VideoDataHandler
from tests.unit.detectors.data_handlers.video.conftest import (
    assert_handler_output,
    make_frames_cache,
    make_io,
    make_sequence_context,
)

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_two_cameras_subfolder(tmp_path, default_vid_patches):
    cameras = ["cam_front", "cam_top"]
    data_source_folder = tmp_path / "data_source"
    expected_paths = {}
    for cam in cameras:
        sub = data_source_folder / cam
        sub.mkdir(parents=True)
        p = sub / "recording.mp4"
        p.touch()
        expected_paths[cam] = p

    ctx = make_sequence_context(cameras)
    io = make_io(tmp_path, data_source_folder)
    handler = VideoDataHandler(io=io, sequence_context=ctx)
    handler.prepare()

    assert handler.camera_video_paths == expected_paths
    assert_handler_output(handler, cameras)
    assert default_vid_patches["split_into_frames"].call_count == 2


def test_two_cameras_subfolder_cache(tmp_path, default_vid_patches):
    cameras = ["cam_front", "cam_top"]
    data_source_folder = tmp_path / "data_source"
    expected_paths = {}
    for cam in cameras:
        sub = data_source_folder / cam
        sub.mkdir(parents=True)
        p = sub / "recording.mp4"
        p.touch()
        expected_paths[cam] = p

    ctx = make_sequence_context(cameras)
    io = make_io(tmp_path, data_source_folder)
    make_frames_cache(io, ctx)
    handler = VideoDataHandler(io=io, sequence_context=ctx)
    handler.prepare()

    assert handler.camera_video_paths == expected_paths
    assert_handler_output(handler, cameras)
    default_vid_patches["split_into_frames"].assert_not_called()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_one_camera_missing_raises(tmp_path, default_vid_patches):
    # cam_front/recording.mp4 exists, cam_top subfolder does not
    cameras = ["cam_front", "cam_top"]
    data_source_folder = tmp_path / "data_source"
    sub = data_source_folder / "cam_front"
    sub.mkdir(parents=True)
    (sub / "recording.mp4").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="No video file found"):
        handler.prepare()


def test_one_camera_wrong_extension_raises(tmp_path, default_vid_patches):
    # cam_front/recording.mp4 ok, cam_top/recording.mp3 not in VIDEO_EXTENSIONS
    cameras = ["cam_front", "cam_top"]
    data_source_folder = tmp_path / "data_source"
    for cam, ext in zip(cameras, [".mp4", ".mp3"]):
        sub = data_source_folder / cam
        sub.mkdir(parents=True)
        (sub / f"recording{ext}").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="No video file found"):
        handler.prepare()


def test_one_camera_ambiguous_raises(tmp_path, default_vid_patches):
    # cam_front/recording.mp4 ok, cam_top/a.mp4 + cam_top/b.mp4 → ambiguity
    cameras = ["cam_front", "cam_top"]
    data_source_folder = tmp_path / "data_source"
    (data_source_folder / "cam_front").mkdir(parents=True)
    (data_source_folder / "cam_front" / "recording.mp4").touch()
    (data_source_folder / "cam_top").mkdir(parents=True)
    (data_source_folder / "cam_top" / "a.mp4").touch()
    (data_source_folder / "cam_top" / "b.mp4").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        handler.prepare()
