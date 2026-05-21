from pathlib import Path

from pydantic import BaseModel, Field


class MachineSpecificConfig(BaseModel):
    """
    Schema for machine-specific paths configuration.
    Contains paths and secrets that are fixed per physical machine (conda installation, etc.).
    """

    conda_path: Path
    hugging_face_token: str = Field(
        default="",
        description="Optional token for Hugging Face gated models (e.g. sam_3d_body). Leave empty if unused.",
    )
