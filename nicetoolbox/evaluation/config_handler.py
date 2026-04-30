from pathlib import Path

from ..configs.project_config_handler import ProjectConfigHandler
from ..configs.schemas.evaluation_config import EvaluationConfig
from ..configs.schemas.experiment_config import CodeConfig, EvaluationExperimentConfig
from ..configs.schemas.machine_specific_paths import MachineSpecificConfig
from ..configs.schemas.predictions_mapping import PredictionsMappingConfig
from ..configs.utils import model_to_dict
from ..utils.config import save_config


class ConfigHandler(ProjectConfigHandler):
    """Handles loading and resolving all configurations required for the evaluation pipeline."""

    # Input paths
    machine_specific_path: Path
    eval_config_file_path: Path

    # Loaded configs
    machine_specific_config: MachineSpecificConfig
    eval_config: EvaluationConfig
    predictions_mapping: PredictionsMappingConfig

    def __init__(self, project_folder_path: Path, machine_specifics_path: Path, eval_config_path: Path):
        """Load and resolve all configurations required for the evaluation pipeline.

        Args:
            project_folder_path (Path): Path to the project folder containing nice_project.toml.
            machine_specifics_path (Path): Path to machine_specific_paths.toml.
            eval_config_path (Path): Path to evaluation_config.toml
        """
        super().__init__(project_folder_path)

        # Resolve file paths (may contain <configs_folder_path> etc.)
        self.machine_specific_path = self.cfg_loader.resolve(machine_specifics_path)
        self.eval_config_file_path = self.cfg_loader.resolve(eval_config_path)

        # Load machine-specific config and expose its paths as placeholders
        self.machine_specific_config = self.cfg_loader.load_config(self.machine_specific_path, MachineSpecificConfig)
        self.cfg_loader.extend_global_ctx(self.machine_specific_config)

        # Load and resolve evaluation config
        self.eval_config = self.cfg_loader.load_config(self.eval_config_file_path, EvaluationConfig)

        # Load predictions mapping
        mapping_path = self.eval_config.predictions_mapping
        self.predictions_mapping = self.cfg_loader.load_config(mapping_path, PredictionsMappingConfig)

    def save_experiment_config(self, output_folder) -> Path:
        """Save the full resolved config snapshot to a TOML file.

        Args:
            output_folder: Directory where the config file will be written.

        Returns:
            Path to the saved config file.
        """
        # we save current auto_placeholders for reproduction purposes
        code_config = CodeConfig(**self.auto_placeholders)
        # save all experiment configurations
        config = EvaluationExperimentConfig(
            project_folder=self.project_folder,
            project_config_path=self.project_config_path,
            machine_specific_path=self.machine_specific_path,
            evaluation_config_path=self.eval_config_file_path,
            code_config=code_config,
            machine_specific_config=self.machine_specific_config,
            project_config=self.project_config,
            evaluation_config=self.eval_config,
            predictions_mapping=self.predictions_mapping,
        )

        path = output_folder / f"config_{code_config.time}.toml"
        save_config(model_to_dict(config), path)

        return path
