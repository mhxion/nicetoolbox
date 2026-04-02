# ruff: noqa: ARG001
import pytest

from nicetoolbox.detectors.data_handlers.video_handler import VideoDataHandler
from tests.unit.detectors.data_handlers.video.conftest import (
    make_frames_cache,
    make_io,
    make_sequence_context,
    make_simple_handler,
)


def test_recipe_root_path(tmp_path, default_vid_patches):
    cameras = ["cam_front"]
    handler, ctx, io = make_simple_handler(tmp_path, cameras)
    make_frames_cache(io, ctx)
    handler.prepare()

    recipe = handler.get_recipe()
    assert recipe.root_path == str(tmp_path / "nice_input")


def test_recipe_unsorted_cameras_are_sorted(tmp_path, default_vid_patches):
    cameras = ["cam_top", "cam_front"]  # intentionally unsorted
    handler, ctx, io = make_simple_handler(tmp_path, cameras)
    make_frames_cache(io, ctx)
    handler.prepare()

    recipe = handler.get_recipe()
    assert recipe.camera_names == ["cam_front", "cam_top"]


def test_recipe_before_prepare_raises(tmp_path):
    cameras = ["cam_front"]
    ctx = make_sequence_context(cameras)
    io = make_io(tmp_path, tmp_path / "data_source")
    handler = VideoDataHandler(io=io, sequence_context=ctx)

    with pytest.raises((AttributeError, TypeError)):
        handler.get_recipe()
