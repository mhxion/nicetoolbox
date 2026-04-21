from pathlib import Path

from .config_loader import ConfigLoader
from .placeholders import PLACEHOLDERS_TYPE
from .schemas.project_config import ProjectConfig
from .utils import default_auto_placeholders, default_runtime_placeholders

PROJECT_CONFIG_FILENAME = "nice_project.toml"


class ProjectConfigHandler:
    """
    Base class for all pipeline configuration handlers.
    Handles the common setup of default placeholders and loading project config.
    """

    # Input paths
    project_folder: Path
    project_config_path: Path

    # Config loader and placeholders
    cfg_loader: ConfigLoader
    auto_placeholders: dict[str, PLACEHOLDERS_TYPE]
    runtime_placeholders: set[str]

    # Loaded configs
    project_config: ProjectConfig

    def __init__(self, project_folder_path: Path):
        """
        Load shared base configuration files.

        Args:
            project_folder_path (Path): Path to the project folder containing nice_project.toml.
        """
        # init config loader with default auto placeholders and runtime
        self.auto_placeholders = default_auto_placeholders()
        self.runtime_placeholders = default_runtime_placeholders()
        self.cfg_loader = ConfigLoader(self.auto_placeholders, self.runtime_placeholders)

        # add project_folder_path to the global context placeholders
        # it can be used to resolve path to other configs
        self.project_folder = project_folder_path.resolve()  # get absolute path for logging and debugging
        if not self.project_folder.is_dir():
            raise FileNotFoundError(f"Project path '{project_folder_path}' isn't a directory!")
        self.cfg_loader.extend_global_ctx({"project_folder_path": str(self.project_folder)})

        # load project config
        self.project_config_path = self.project_folder / PROJECT_CONFIG_FILENAME
        if not self.project_config_path.is_file():
            raise FileNotFoundError(
                f"Project config {PROJECT_CONFIG_FILENAME} isn't found in '{self.project_folder}' "
                "Have you called 'make create_project' during installation?"
            )
        self.project_config = self.cfg_loader.load_config(self.project_config_path, ProjectConfig)
        self.cfg_loader.extend_global_ctx(self.project_config)
