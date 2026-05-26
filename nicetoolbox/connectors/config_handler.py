from pathlib import Path
from typing import Generic, TypeVar

from ..configs.project_config_handler import ProjectConfigHandler
from ..configs.schemas.machine_specific_paths import MachineSpecificConfig
from ..utils import logging_utils as log_ut

T = TypeVar("T")


class ConnectorConfigHandler(ProjectConfigHandler, Generic[T]):
    connector_config: T

    def __init__(
        self,
        project_folder_path: Path,
        machine_specifics: Path,
        connector_config_path: Path,
        connector_config_schema: type[T],
    ):
        log_ut.init_console_logging()
        super().__init__(project_folder_path)

        machine_specific_path = self.cfg_loader.resolve(machine_specifics)
        machine_cfg = self.cfg_loader.load_config(machine_specific_path, MachineSpecificConfig)
        self.cfg_loader.extend_global_ctx(machine_cfg)

        resolved_connector_config = self.cfg_loader.resolve(connector_config_path)
        self.connector_config = self.cfg_loader.load_config(resolved_connector_config, connector_config_schema)

        log_ut.init_file_logging(self.connector_config.log_file_path, self.connector_config.log_level)
