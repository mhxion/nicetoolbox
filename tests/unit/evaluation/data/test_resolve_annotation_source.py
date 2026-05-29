from pathlib import Path

import pytest

from nicetoolbox.configs.schemas.evaluation_input_block import AnnotationInput
from nicetoolbox.evaluation.data.input_loader import _resolve_annotation_source
from tests.unit.evaluation.data.conftest import (
    MULTI_DATASET,
    SINGLE_DATASET,
    THREE_DATASETS,
    expected_annotation_metas,
    make_experiment_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ann_path(tmp_path: Path, suffix: str = "gt.npz") -> Path:
    """Annotation path template rooted in tmp_path with standard placeholders."""
    return tmp_path / "<cur_dataset_name>" / "<cur_session_ID>" / "<cur_sequence_ID>" / suffix


def _with_annotations(datasets: dict, tmp_path: Path, comp: str = "body_joints") -> dict:
    """Return a copy of datasets with annotation_components added to each dataset."""
    ann_path = _ann_path(tmp_path)
    return {ds_name: {**ds_spec, "annotation_components": {comp: ann_path}} for ds_name, ds_spec in datasets.items()}


def _resolved_ann(tmp_path: Path, ds: str, session: str, sequence: str, suffix: str = "gt.npz") -> Path:
    """Return the fully-resolved annotation path for a specific video."""
    return tmp_path / ds / session / sequence / suffix


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_single_entry(self, tmp_path):
        datasets = _with_annotations(SINGLE_DATASET, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        block = AnnotationInput(component="body_joints", npz_key="landmarks")

        result = _resolve_annotation_source(block, cfg)

        assert result == expected_annotation_metas(datasets, "landmarks")

    def test_two_datasets(self, tmp_path):
        datasets = _with_annotations(MULTI_DATASET, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        block = AnnotationInput(component="body_joints", npz_key="landmarks")

        result = _resolve_annotation_source(block, cfg)

        assert result == expected_annotation_metas(datasets, "landmarks")
        assert len(result) == 2

    def test_multiple_sequences_produce_one_entry_each(self, tmp_path):
        multi_video = {
            "ds1": {
                "fps": 30,
                "videos": [
                    {"session_ID": "s01", "sequence_ID": "seq01"},
                    {"session_ID": "s01", "sequence_ID": "seq02"},
                ],
            }
        }
        datasets = _with_annotations(multi_video, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        block = AnnotationInput(component="body_joints", npz_key="landmarks")

        result = _resolve_annotation_source(block, cfg)

        assert len(result) == 2
        assert {m.sequence for m in result} == {"seq01", "seq02"}

    def test_multiple_subsequences_same_sequence_no_duplicates(self, tmp_path):
        """Multiple videos sharing the same session/sequence (subsequences) must
        produce only one annotation entry — annotations are per-sequence, not per-subsequence."""
        same_sequence = {
            "ds1": {
                "fps": 30,
                "videos": [
                    {"session_ID": "s01", "sequence_ID": "seq01", "video_start": 0, "video_length": 100},
                    {"session_ID": "s01", "sequence_ID": "seq01", "video_start": 100, "video_length": 100},
                    {"session_ID": "s01", "sequence_ID": "seq01", "video_start": 200, "video_length": 100},
                ],
            }
        }
        datasets = _with_annotations(same_sequence, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        block = AnnotationInput(component="body_joints", npz_key="landmarks")

        result = _resolve_annotation_source(block, cfg)

        assert len(result) == 1
        assert result[0].sequence == "seq01"

    def test_npz_key_propagated(self, tmp_path):
        datasets = _with_annotations(SINGLE_DATASET, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        block = AnnotationInput(component="body_joints", npz_key="distances")

        result = _resolve_annotation_source(block, cfg)

        assert result[0].npz_key == "distances"


# ---------------------------------------------------------------------------
# Missing files — silently skipped
# ---------------------------------------------------------------------------


class TestMissingFiles:
    def test_missing_annotation_skipped(self, tmp_path):
        datasets = _with_annotations(SINGLE_DATASET, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        _resolved_ann(tmp_path, "ds1", "s01", "seq01").unlink()

        block = AnnotationInput(component="body_joints", npz_key="landmarks")
        result = _resolve_annotation_source(block, cfg)

        assert result == []

    def test_partial_missing_returns_present_only(self, tmp_path):
        datasets = _with_annotations(MULTI_DATASET, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        _resolved_ann(tmp_path, "ds1", "s01", "seq01").unlink()

        block = AnnotationInput(component="body_joints", npz_key="landmarks")
        result = _resolve_annotation_source(block, cfg)

        assert len(result) == 1
        assert result[0].dataset == "ds2"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_dataset_missing_from_properties_raises(self, tmp_path):
        datasets = _with_annotations(SINGLE_DATASET, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        del cfg.dataset_config["ds1"]

        block = AnnotationInput(component="body_joints", npz_key="landmarks")

        with pytest.raises(KeyError, match="not in dataset_properties"):
            _resolve_annotation_source(block, cfg)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_filter_by_dataset(self, tmp_path):
        datasets = _with_annotations(MULTI_DATASET, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        block = AnnotationInput(component="body_joints", npz_key="landmarks", dataset="ds1")

        result = _resolve_annotation_source(block, cfg)

        assert len(result) == 1
        assert result[0].dataset == "ds1"

    def test_filter_by_session(self, tmp_path):
        datasets = _with_annotations(MULTI_DATASET, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        block = AnnotationInput(component="body_joints", npz_key="landmarks", session="s01")

        result = _resolve_annotation_source(block, cfg)

        assert len(result) == 1
        assert result[0].session == "s01"

    def test_filter_by_dataset_list(self, tmp_path):
        datasets = _with_annotations(THREE_DATASETS, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        block = AnnotationInput(component="body_joints", npz_key="landmarks", dataset=["ds1", "ds3"])

        result = _resolve_annotation_source(block, cfg)

        assert len(result) == 2
        assert {m.dataset for m in result} == {"ds1", "ds3"}

    def test_filter_no_match_returns_empty(self, tmp_path):
        datasets = _with_annotations(SINGLE_DATASET, tmp_path)
        cfg = make_experiment_config(tmp_path, datasets, {})
        block = AnnotationInput(component="body_joints", npz_key="landmarks", dataset="nonexistent")

        result = _resolve_annotation_source(block, cfg)

        assert result == []
