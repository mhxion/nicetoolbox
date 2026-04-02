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
# Happy path — one level deep
# ---------------------------------------------------------------------------


def test_single_camera_subfolder(tmp_path, default_vid_patches):
    # cam_front/recording.mp4 — directory name matches camera
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    sub = data_source_folder / "cam_front"
    sub.mkdir(parents=True)
    expected_video_path = sub / "recording.mp4"
    expected_video_path.touch()

    ctx = make_sequence_context(cameras)
    io = make_io(tmp_path, data_source_folder)
    handler = VideoDataHandler(io=io, sequence_context=ctx)
    handler.prepare()

    assert handler.camera_video_paths == {"cam_front": expected_video_path}
    assert_handler_output(handler, cameras)
    default_vid_patches["split_into_frames"].assert_called_once()


def test_single_camera_subfolder_cache(tmp_path, default_vid_patches):
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    sub = data_source_folder / "cam_front"
    sub.mkdir(parents=True)
    expected_video_path = sub / "recording.mp4"
    expected_video_path.touch()

    ctx = make_sequence_context(cameras)
    io = make_io(tmp_path, data_source_folder)
    make_frames_cache(io, ctx)
    handler = VideoDataHandler(io=io, sequence_context=ctx)
    handler.prepare()

    assert handler.camera_video_paths == {"cam_front": expected_video_path}
    assert_handler_output(handler, cameras)
    default_vid_patches["split_into_frames"].assert_not_called()


# ---------------------------------------------------------------------------
# Deep nesting — camera name as an ancestor directory
# ---------------------------------------------------------------------------


def test_single_camera_deep_nesting(tmp_path, default_vid_patches):
    # cam_front/session_01/clip/recording.mp4 — cam_front is an ancestor dir
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    deep = data_source_folder / "cam_front" / "session_01" / "clip"
    deep.mkdir(parents=True)
    expected_video_path = deep / "recording.mp4"
    expected_video_path.touch()

    ctx = make_sequence_context(cameras)
    io = make_io(tmp_path, data_source_folder)
    make_frames_cache(io, ctx)
    handler = VideoDataHandler(io=io, sequence_context=ctx)
    handler.prepare()

    assert handler.camera_video_paths == {"cam_front": expected_video_path}
    assert_handler_output(handler, cameras)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_partial_subfolder_name_no_match(tmp_path, default_vid_patches):
    # cam_front_test/recording.mp4 — partial dir name must not match 'cam_front'
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    (data_source_folder / "cam_front_test").mkdir(parents=True)
    (data_source_folder / "cam_front_test" / "recording.mp4").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="No video file found"):
        handler.prepare()


def test_subfolder_txt_extension_no_match(tmp_path, default_vid_patches):
    # cam_front/recording.txt — wrong extension must not match
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    sub = data_source_folder / "cam_front"
    sub.mkdir(parents=True)
    (sub / "recording.txt").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="No video file found"):
        handler.prepare()


def test_subfolder_unsupported_extension_no_match(tmp_path, default_vid_patches):
    # cam_front/recording.mp3 — .mp3 not in VIDEO_EXTENSIONS
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    sub = data_source_folder / "cam_front"
    sub.mkdir(parents=True)
    (sub / "recording.mp3").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="No video file found"):
        handler.prepare()


def test_subfolder_no_files(tmp_path, default_vid_patches):
    # cam_front/ exists but is empty
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    (data_source_folder / "cam_front").mkdir(parents=True)

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="No video file found"):
        handler.prepare()


def test_subfolder_video_and_txt_sibling(tmp_path, default_vid_patches):
    # cam_front/recording.mp4 and cam_front/recording.txt coexist — only video matches
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    sub = data_source_folder / "cam_front"
    sub.mkdir(parents=True)
    expected_video_path = sub / "recording.mp4"
    expected_video_path.touch()
    (sub / "recording.txt").touch()

    ctx = make_sequence_context(cameras)
    io = make_io(tmp_path, data_source_folder)
    make_frames_cache(io, ctx)
    handler = VideoDataHandler(io=io, sequence_context=ctx)
    handler.prepare()

    assert handler.camera_video_paths == {"cam_front": expected_video_path}


def test_subfolder_two_videos_ambiguous(tmp_path, default_vid_patches):
    # cam_front/a.mp4 and cam_front/b.mp4 — two matches in same subfolder
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    sub = data_source_folder / "cam_front"
    sub.mkdir(parents=True)
    (sub / "a.mp4").touch()
    (sub / "b.mp4").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        handler.prepare()
