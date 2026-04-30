from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from nicetoolbox.configs.placeholders import resolve_placeholders
from nicetoolbox.evaluation.data.input_loader import (
    AnnotationMeta,
    ArrayAxes,
    ExperimentMeta,
    LoadedArray,
    NpzMeta,
    SubsequenceInfo,
)

# ---------------------------------------------------------------------------
# LoadedArray factory
# ---------------------------------------------------------------------------


def make_loaded_array(
    meta: NpzMeta,
    subjects: tuple[str, ...] = ("s1",),
    cameras: tuple[str, ...] = ("c1",),
    frames: tuple[str, ...] = ("f0",),
    labels: tuple[str, ...] = ("j1",),
    data: tuple[str, ...] = (),
) -> LoadedArray:
    """Build a LoadedArray with sequential integer data matching the given axis labels.

    Data values are sequential integers so tests can verify correct slices after intersection.
    """
    shape = (len(subjects), len(cameras), len(frames), len(labels))
    if data:
        shape += (len(data),)
    arr_data = np.arange(int(np.prod(shape)), dtype=float).reshape(shape)
    axes = ArrayAxes(list(subjects), list(cameras), list(frames), list(labels), list(data))
    return LoadedArray(meta=meta, data=arr_data, axes=axes)


# ---------------------------------------------------------------------------
# NPZ factory
# ---------------------------------------------------------------------------


def make_npz(
    path: Path,
    key: str = "landmarks",
    subjects: tuple = ("s1", "s2"),
    cameras: tuple = ("cam1",),
    frames: tuple = ("f0", "f1", "f2"),
    labels: tuple = ("x", "y", "z"),
    data: tuple | None = None,
) -> Path:
    """Create a minimal well-formed npz file at *path* and return it.

    Data values are sequential integers so tests can verify correct slices.
    """
    descr: dict = {
        key: {
            "axis0": list(subjects),
            "axis1": list(cameras),
            "axis2": list(frames),
            "axis3": list(labels),
        }
    }
    shape = (len(subjects), len(cameras), len(frames), len(labels))
    if data is not None:
        descr[key]["axis4"] = list(data)
        shape = shape + (len(data),)

    npz_data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data_description=np.array(descr, dtype=object), **{key: npz_data})
    return path


# ---------------------------------------------------------------------------
# Shared dataset / algorithm constants
# ---------------------------------------------------------------------------


SINGLE_DATASET = {
    "ds1": {
        "fps": 30,
        "components": ["body_joints"],
        "videos": [{"session_ID": "s01", "sequence_ID": "seq01"}],
    },
}

MULTI_DATASET = {
    "ds1": {
        "fps": 30,
        "components": ["body_joints"],
        "videos": [{"session_ID": "s01", "sequence_ID": "seq01"}],
    },
    "ds2": {
        "fps": 60,
        "components": ["body_joints"],
        "videos": [{"session_ID": "s02", "sequence_ID": "seq02"}],
    },
}

THREE_DATASETS = {
    "ds1": {
        "fps": 30,
        "components": ["body_joints"],
        "videos": [{"session_ID": "s01", "sequence_ID": "seq01"}],
    },
    "ds2": {
        "fps": 60,
        "components": ["body_joints"],
        "videos": [{"session_ID": "s02", "sequence_ID": "seq02"}],
    },
    "ds3": {
        "fps": 25,
        "components": ["body_joints"],
        "videos": [{"session_ID": "s03", "sequence_ID": "seq03"}],
    },
}

MULTI_VIDEO = {
    "ds1": {
        "fps": 30,
        "components": ["body_joints"],
        "videos": [
            {"session_ID": "s01", "sequence_ID": "seq01", "video_start": 0, "video_length": 100},
            {"session_ID": "s01", "sequence_ID": "seq01", "video_start": 100, "video_length": 100},
        ],
    },
}

TWO_ALGORITHMS = {"body_joints": ["algo_a", "algo_b"]}
THREE_ALGORITHMS = {"body_joints": ["algo_a", "algo_b", "algo_c"]}


# ---------------------------------------------------------------------------
# Lightweight stand-ins for experiment config objects
# ---------------------------------------------------------------------------


@dataclass
class FakeVideo:
    session_ID: str
    sequence_ID: str
    video_start: int | str = 0
    video_length: int | str = 100


@dataclass
class FakeRunDataset:
    components: list[str]
    videos: list[FakeVideo]


@dataclass
class FakeAnnotationComponent:
    path: Path


@dataclass
class FakeAnnotation:
    components: dict[str, FakeAnnotationComponent] = field(default_factory=dict)


@dataclass
class FakeDatasetConfig:
    fps: int = 30
    session_IDs: list[str] = field(default_factory=list)
    sequence_IDs: list[str] = field(default_factory=list)
    annotation: FakeAnnotation = field(default_factory=FakeAnnotation)


@dataclass
class FakeRunIO:
    detector_final_result_folder: Path = Path("")


@dataclass
class FakeRunFile:
    run: dict[str, FakeRunDataset] = field(default_factory=dict)
    component_algorithm_mapping: dict[str, list[str]] = field(default_factory=dict)
    io: FakeRunIO = field(default_factory=FakeRunIO)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_experiment_config(tmp_path: Path, datasets: dict, comp_algo_mapping: dict[str, list[str]]) -> MagicMock:
    """Build a mock DetectorsExperimentConfig for resolver tests.

    Each dataset entry supports:
    - ``fps``, ``components``, ``videos`` — for experiment source
    - ``annotation_components`` — optional dict of ``{comp_name: path_with_placeholders}``
      for annotation source. Paths may contain ``<cur_dataset_name>``,
      ``<cur_session_ID>``, ``<cur_sequence_ID>`` placeholders.
    """
    placeholder_path = Path("<cur_dataset_name>") / "<cur_session_ID>" / "<cur_sequence_ID>" / "<cur_component_name>"
    result_template = tmp_path / placeholder_path

    run_datasets: dict[str, FakeRunDataset] = {}
    dataset_configs: dict[str, FakeDatasetConfig] = {}

    for ds_name, ds_spec in datasets.items():
        videos = [FakeVideo(**v) for v in ds_spec["videos"]]
        run_datasets[ds_name] = FakeRunDataset(
            components=ds_spec["components"],
            videos=videos,
        )

        # Build annotation components
        annotation_components: dict[str, FakeAnnotationComponent] = {}
        for comp_name, ann_path in ds_spec.get("annotation_components", {}).items():
            annotation_components[comp_name] = FakeAnnotationComponent(path=ann_path)
            # Create annotation files on disk for each video
            for video in videos:
                ctx = {
                    "cur_dataset_name": ds_name,
                    "cur_session_ID": video.session_ID,
                    "cur_sequence_ID": video.sequence_ID,
                }
                resolved = Path(resolve_placeholders(ann_path, ctx))
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.touch()

        dataset_configs[ds_name] = FakeDatasetConfig(
            fps=ds_spec.get("fps", 30),
            session_IDs=list({v["session_ID"] for v in ds_spec["videos"]}),
            sequence_IDs=list({v["sequence_ID"] for v in ds_spec["videos"]}),
            annotation=FakeAnnotation(components=annotation_components),
        )

        # Create experiment .npz files on disk
        for video in videos:
            for comp in ds_spec["components"]:
                for algo in comp_algo_mapping.get(comp, []):
                    npz_dir = tmp_path / ds_name / video.session_ID / video.sequence_ID / comp
                    npz_dir.mkdir(parents=True, exist_ok=True)
                    (npz_dir / f"{algo}.npz").touch()

    run_file = FakeRunFile(
        run=run_datasets,
        component_algorithm_mapping=comp_algo_mapping,
        io=FakeRunIO(detector_final_result_folder=result_template),
    )

    cfg = MagicMock()
    cfg.run_config = run_file
    cfg.dataset_config = dataset_configs
    return cfg


# ---------------------------------------------------------------------------
# Expected output builders
# ---------------------------------------------------------------------------


def expected_experiment_metas(
    tmp_path: Path,
    datasets: dict,
    comp_algo_mapping: dict[str, list[str]],
    npz_key: str,
) -> list[ExperimentMeta]:
    """Build the expected ExperimentMeta list matching make_experiment_config output."""
    out: list[ExperimentMeta] = []
    for ds_name, ds_spec in datasets.items():
        fps = ds_spec.get("fps", 30)
        for subseq_idx, v in enumerate(ds_spec["videos"]):
            video_start = v.get("video_start", 0)
            video_length = v.get("video_length", 100)
            for comp in ds_spec["components"]:
                for algo in comp_algo_mapping.get(comp, []):
                    out.append(
                        ExperimentMeta(
                            dataset=ds_name,
                            session=v["session_ID"],
                            sequence=v["sequence_ID"],
                            component=comp,
                            algorithm=algo,
                            fps=fps,
                            subsequence=SubsequenceInfo(
                                subsequence_index=subseq_idx,
                                video_start=video_start,
                                video_length=video_length,
                            ),
                            npz_path=tmp_path / ds_name / v["session_ID"] / v["sequence_ID"] / comp / f"{algo}.npz",
                            npz_key=npz_key,
                        )
                    )
    return sorted(out, key=lambda x: str(x.npz_path))


def expected_annotation_metas(
    datasets: dict,
    npz_key: str,
) -> list[AnnotationMeta]:
    """Build the expected AnnotationMeta list matching make_experiment_config output."""
    out: list[AnnotationMeta] = []
    for ds_name, ds_spec in datasets.items():
        for v in ds_spec["videos"]:
            ctx = {
                "cur_dataset_name": ds_name,
                "cur_session_ID": v["session_ID"],
                "cur_sequence_ID": v["sequence_ID"],
            }
            for comp_name, ann_path in ds_spec.get("annotation_components", {}).items():
                resolved = Path(resolve_placeholders(ann_path, ctx))
                out.append(
                    AnnotationMeta(
                        dataset=ds_name,
                        session=v["session_ID"],
                        sequence=v["sequence_ID"],
                        component=comp_name,
                        npz_path=resolved,
                        npz_key=npz_key,
                    )
                )
    return sorted(out, key=lambda x: str(x.npz_path))
