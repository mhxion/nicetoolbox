"""
Parsing and handling of configuration files for evaluation.
Provides loop over evaluation tasks.
"""

import logging
from pathlib import Path
from typing import Iterator, List, Tuple

from ..configs.project_config_handler import ProjectConfigHandler
from ..configs.schemas.dataset_properties import DatasetConfig, DatasetConfigEvaluation, DatasetProperties
from ..configs.schemas.detectors_run_file import DetectorsRunConfig, DetectorsRunIO
from ..configs.schemas.evaluation_config import AggregationConfig, EvaluationConfig, EvaluationIO, EvaluationMetricType
from ..configs.schemas.experiment_config import DetectorsExperimentConfig
from ..configs.schemas.machine_specific_paths import MachineSpecificConfig
from ..configs.utils import get_latest_experiment_config_path, model_to_dict
from ..utils.logging_utils import log_configs
from .config_schema import FinalEvaluationConfig


class ConfigHandler(ProjectConfigHandler):
    # Input paths
    machine_specific_path: Path
    eval_config_file_path: Path

    # Loaded configs
    machine_specific_config: MachineSpecificConfig
    global_settings: EvaluationConfig
    io_config: EvaluationIO
    metric_type_configs: dict[str, EvaluationMetricType]
    summaries_configs: dict[str, AggregationConfig]

    experiment_config: DetectorsExperimentConfig
    experiment_io: DetectorsRunIO
    component_algorithm_mapping: dict[str, list[str]]
    all_run_configs: dict[str, DetectorsRunConfig]
    all_dataset_properties: DatasetProperties

    def __init__(self, project_folder: Path, machine_specifics_file: Path, eval_config_file: Path) -> None:
        """
        Handles loading and parsing of configuration files for evaluation.

        Args:
            project_folder (Path): Path to the project folder containing nice_project.toml.
            machine_specifics_file (Path): Path to machine_specific_paths.toml, may contain placeholders.
            eval_config_file (Path): Path to evaluation_config.toml, may contain placeholders.
        """
        # initialize config loader, default placeholders and project config
        super().__init__(project_folder)

        # this paths we need to resolve manually, because they are external arguments
        self.machine_specific_path = self.cfg_loader.resolve(machine_specifics_file)
        self.eval_config_file_path = self.cfg_loader.resolve(eval_config_file)

        # machine specific config
        self.machine_specific_config = self.cfg_loader.load_config(self.machine_specific_path, MachineSpecificConfig)
        self.cfg_loader.extend_global_ctx(self.machine_specific_config)

        # Evaluation config
        self.global_settings = self.cfg_loader.load_config(self.eval_config_file_path, EvaluationConfig)
        # Shortcuts for evaluation config fields
        self.io_config = self.global_settings.io
        self.metric_type_configs = self.global_settings.metrics
        self.summaries_configs = self.global_settings.summaries

        # Load the latest run configuration from the experiment folder
        # It contains the run_config and dataset_properties
        experiment_folder = self.io_config.experiment_folder
        experiment_cfg_path = get_latest_experiment_config_path(experiment_folder)
        logging.info(f"Loading latest experiment run configuration from: {experiment_cfg_path}")
        self.experiment_config = self.cfg_loader.load_config(
            experiment_cfg_path,
            DetectorsExperimentConfig,
            ignore_auto_and_global=True,
        )

        # verify that the evaluation project matches the experiment project
        exp_configs_folder = self.experiment_config.project_config.configs_folder_path
        eval_configs_folder = self.project_config.configs_folder_path.resolve()
        if exp_configs_folder != eval_configs_folder:
            raise ValueError(
                f"Project mismatch: evaluation project configs_folder_path '{eval_configs_folder}' "
                f"differs from experiment project '{exp_configs_folder}'"
            )

        # Shortucts for experiment config
        run_config = self.experiment_config.run_config
        self.experiment_io = run_config.io
        self.component_algorithm_mapping = run_config.component_algorithm_mapping
        self.all_run_configs = run_config.run
        self.all_dataset_properties = self.experiment_config.dataset_config

    def save_experiment_config(self, output_folder: Path) -> None:
        """
        Save the effective configuration for the overall evaluation experiment run.
        This includes the evaluation config, run config, dataset properties,
        detector config, and machine-specific config.

        Args:
            output_folder (Path): The folder where the configuration will be saved.
        """
        log_configs(
            dict(
                machine_specific_config=model_to_dict(self.machine_specific_config),
                io_config=model_to_dict(self.io_config),
                global_eval_config=model_to_dict(self.global_settings),
                run_configs={name: model_to_dict(config) for name, config in self.all_run_configs.items()},
                dataset_properties={name: model_to_dict(props) for name, props in self.all_dataset_properties.items()},
                component_algorithm_mapping=self.component_algorithm_mapping,
            ),
            str(output_folder),
            file_name="config_<time>",
        )

    def get_evaluation_and_dataset_configs(self) -> Iterator[Tuple]:
        """
        Generator that yields tuples of (RunConfig, DatasetProperties, EvaluationConfig)
        for each dataset defined in the run configurations.

        Each tuple defines a task for the EvaluationEngine to process. This method is
        thus the main loop over datasets and their evaluation settings and tasks.

        Yields:
            Iterator[Tuple[RunConfig, DatasetProperties, EvaluationConfig]]:
                Tuples containing the run configuration, dataset properties, and
                evaluation configuration for each dataset.
        """
        for dataset_name, run_config in self.all_run_configs.items():
            if dataset_name not in self.all_dataset_properties:
                logging.error(
                    f"Dataset '{dataset_name}' from run_configs not found in" " dataset_properties. Skipping."
                )
                continue

            # (1) Dataset properties and run config per dataset
            dataset_properties: DatasetConfig = self.all_dataset_properties[dataset_name]

            # (2) Build evaluation configs based on dataset properties
            evaluation_entries: List[DatasetConfigEvaluation] = dataset_properties.evaluation

            # (3) Prepare evaluation config
            metric_types_dict = {}
            prediction_components = {}
            annotation_components = {}

            for eval_entry_obj in evaluation_entries:
                annot_components = eval_entry_obj.annotation_components

                for metric_type in eval_entry_obj.metric_types:
                    if metric_type not in self.metric_type_configs:
                        logging.warning(
                            f"Metric type '{metric_type}' for dataset '{dataset_name}'"
                            " not in global metric_type_configs. Skipping."
                        )
                        continue

                    metric_type_config = self.metric_type_configs[metric_type]
                    gt_required = metric_type_config.gt_required

                    # Add MetricTypeConfig to the dictionary
                    metric_types_dict[metric_type] = metric_type_config

                    if gt_required:
                        prediction_components[metric_type] = annot_components
                        annotation_components[metric_type] = annot_components
                    else:
                        gt_components = metric_type_config.gt_components
                        prediction_components[metric_type] = gt_components
                        annotation_components[metric_type] = ["none"]

            # (4) Finalize evaluation config
            evaluation_config = FinalEvaluationConfig(
                device=self.global_settings.device,
                verbose=self.global_settings.verbose,
                batchsize=self.global_settings.batchsize,
                prediction_components=prediction_components,
                annotation_components=annotation_components,
                metric_types=metric_types_dict,
                component_algorithm_mapping=self.component_algorithm_mapping,
            )

            yield run_config, dataset_properties, evaluation_config
