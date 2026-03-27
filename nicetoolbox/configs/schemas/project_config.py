from pathlib import Path

from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """
    Schema for project-specific paths configuration.
    Contains paths that vary per project/work context (not per machine).
    """

    configs_folder_path: Path
    datasets_folder_path: Path
    output_folder_path: Path
