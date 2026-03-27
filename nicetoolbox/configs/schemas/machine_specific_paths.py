from pathlib import Path

from pydantic import BaseModel


class MachineSpecificConfig(BaseModel):
    """
    Schema for machine-specific paths configuration.
    Contains only paths that are fixed per physical machine (conda installation, etc.).
    """

    conda_path: Path
