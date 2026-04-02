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


def test_single_camera_flat(tmp_path, default_vid_patches):
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    expected_video_path = data_source_folder / "cam_front.mp4"
    expected_video_path.parent.mkdir(parents=True, exist_ok=True)
    expected_video_path.touch()

    ctx = make_sequence_context(cameras)
    io = make_io(tmp_path, data_source_folder)
    handler = VideoDataHandler(io=io, sequence_context=ctx)
    handler.prepare()

    assert handler.camera_video_paths == {"cam_front": expected_video_path}
    assert_handler_output(handler, cameras)
    default_vid_patches["probe_video"].assert_called()
    default_vid_patches["split_into_frames"].assert_called_once()


def test_single_camera_flat_cache(tmp_path, default_vid_patches):
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    expected_video_path = data_source_folder / "cam_front.mp4"
    expected_video_path.parent.mkdir(parents=True, exist_ok=True)
    expected_video_path.touch()

    ctx = make_sequence_context(cameras)
    io = make_io(tmp_path, data_source_folder)
    make_frames_cache(io, ctx)
    handler = VideoDataHandler(io=io, sequence_context=ctx)
    handler.prepare()

    assert handler.camera_video_paths == {"cam_front": expected_video_path}
    assert_handler_output(handler, cameras)
    default_vid_patches["probe_video"].assert_called()
    default_vid_patches["split_into_frames"].assert_not_called()


# ---------------------------------------------------------------------------
# File discovery edge cases
# ---------------------------------------------------------------------------


def test_partial_name_no_match(tmp_path, default_vid_patches):
    # cam_front_test.mp4 must not match camera 'cam_front'
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    data_source_folder.mkdir(parents=True)
    (data_source_folder / "cam_front_test.mp4").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="No video file found"):
        handler.prepare()


def test_txt_extension_no_match(tmp_path, default_vid_patches):
    # cam_front.txt must not match even though the stem is correct
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    data_source_folder.mkdir(parents=True)
    (data_source_folder / "cam_front.txt").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="No video file found"):
        handler.prepare()


def test_unsupported_video_extension_no_match(tmp_path, default_vid_patches):
    # cam_front.mp3 is not in VIDEO_EXTENSIONS so must not match
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    data_source_folder.mkdir(parents=True)
    (data_source_folder / "cam_front.mp3").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="No video file found"):
        handler.prepare()


def test_no_files(tmp_path, default_vid_patches):
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    data_source_folder.mkdir(parents=True)

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="No video file found"):
        handler.prepare()


def test_video_and_txt_sibling(tmp_path, default_vid_patches):
    # cam_front.mp4 and cam_front.txt coexist — only the video should match
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    data_source_folder.mkdir(parents=True)
    expected_video_path = data_source_folder / "cam_front.mp4"
    expected_video_path.touch()
    (data_source_folder / "cam_front.txt").touch()

    ctx = make_sequence_context(cameras)
    io = make_io(tmp_path, data_source_folder)
    make_frames_cache(io, ctx)
    handler = VideoDataHandler(io=io, sequence_context=ctx)
    handler.prepare()

    assert handler.camera_video_paths == {"cam_front": expected_video_path}


def test_stem_and_subfolder_match_ambiguous(tmp_path, default_vid_patches):
    # cam_front.mp4 (stem match) and cam_front/recording.mp4 (dir match) — ambiguity error
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    (data_source_folder / "cam_front").mkdir(parents=True)
    (data_source_folder / "cam_front.mp4").touch()
    (data_source_folder / "cam_front" / "recording.mp4").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        handler.prepare()


def test_two_extensions_ambiguous(tmp_path, default_vid_patches):
    # cam_front.mp4 and cam_front.mov both match — ambiguity error
    cameras = ["cam_front"]
    data_source_folder = tmp_path / "data_source"
    data_source_folder.mkdir(parents=True)
    (data_source_folder / "cam_front.mp4").touch()
    (data_source_folder / "cam_front.mov").touch()

    handler = VideoDataHandler(
        io=make_io(tmp_path, data_source_folder),
        sequence_context=make_sequence_context(cameras),
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        handler.prepare()
