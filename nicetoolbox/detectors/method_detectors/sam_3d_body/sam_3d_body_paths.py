"""
Paths for SAM 3D Body: Hugging Face cache, upstream repo (git submodule), and raw inference .npz.
"""

import os
from pathlib import Path

SAM3D_ASSETS_DIRNAME = "sam_3d_body"
SAM3D_REPO_DIRNAME = "sam-3d-body"
SUBMODULES_DIRNAME = "submodules"

SAM3D_BODY_OUTPUT_NPZ_STEM = "sam_3d_body"
SAM3D_BODY_LOCAL_NPZ_STEM = "sam_3d_body_camera"
RAW_INFERENCE_NPZ_NAME = "_sam_3d_body_inference_raw.npz"


def repo_root_from_this_file() -> Path:
    """Repository root (parent of the nicetoolbox package)."""
    return Path(__file__).resolve().parents[4]


def default_sam3d_repo_path() -> Path:
    """Default checkout: <repo>/submodules/sam-3d-body."""
    return repo_root_from_this_file() / SUBMODULES_DIRNAME / SAM3D_REPO_DIRNAME


def default_sam3d_assets_root() -> Path:
    return repo_root_from_this_file() / "nicetoolbox" / "detectors" / "assets" / SAM3D_ASSETS_DIRNAME


def default_hf_home(assets_root: Path | None = None) -> Path:
    root = assets_root or default_sam3d_assets_root()
    return root / "hf_home"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_sam3d_repo(repo: str | None) -> Path:
    """Resolve sam-3d-body checkout containing the sam_3d_body/ package."""
    target = Path(repo.strip()).expanduser().resolve() if repo and str(repo).strip() else default_sam3d_repo_path()
    if not (target / "sam_3d_body").is_dir():
        raise RuntimeError(
            f"SAM 3D Body upstream repo missing or incomplete at {target} (expected package dir "
            f"'{target / 'sam_3d_body'}'). Clone the fork via git submodule: "
            "`git submodule update --init submodules/sam-3d-body` "
            "(see `.gitmodules`), or set `sam3d_repo_path` in `[algorithms.sam_3d_body]` to a "
            "local checkout that contains `sam_3d_body/`."
        )
    return target


def ensure_hf_hub_cache_env(assets_root: Path | None = None) -> Path:
    """Set HF_HOME / HUGGINGFACE_HUB_CACHE under assets/sam_3d_body/hf_home if unset."""
    hf_home = default_hf_home(assets_root)
    ensure_directory(hf_home)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    return hf_home
