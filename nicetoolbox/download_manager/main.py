import argparse
import logging
from pathlib import Path

from ..detectors import config_handler as confh
from ..utils import logging_utils as log_ut
from .manager import AssetManager


def entry_point():
    """
    CLI Entry point for the Asset Manager.
    """
    parser = argparse.ArgumentParser(description="NICE Toolbox On-Demand Asset Downloader")

    parser.add_argument(
        "--run-file", type=Path, default=Path("configs/detectors_run_file.toml"), help="Path to detectors_run_file.toml"
    )
    parser.add_argument(
        "--machine-specifics",
        type=Path,
        default=Path("machine_specific_paths.toml"),
        help="Path to machine_specific_paths.toml",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--components", nargs="+", help="Download assets for specific components (e.g., gaze_individual body_joints)"
    )
    group.add_argument("--all", action="store_true", help="Download all available assets in the manifest")

    args = parser.parse_args()

    run_file = str(args.run_file)
    machine_specifics = str(args.machine_specifics)
    config = confh.Configuration(run_file, machine_specifics)

    log_level = config.run_config.log_level

    main_output_folder = Path(config.run_config.io.out_folder)
    main_output_folder.mkdir(parents=True, exist_ok=True)

    log_file = main_output_folder / "nicetoolbox.log"
    log_ut.setup_logging(log_file, log_level.name)

    logging.info(f"Initialising asset manager; machine_specifics: '{run_file}', run_file: '{machine_specifics}'")
    manager = AssetManager(config)

    if args.all:
        manager.verify_and_download(list(manager.manifest.keys()))

    elif args.components:
        # Filter logic using parsed dicts
        mapping = config.run_config.component_algorithm_mapping
        active_algos = []
        for comp in args.components:
            if comp in mapping:
                active_algos.extend(mapping[comp])
            else:
                logging.warning(f"Component '{comp}' not found in component_algorithm_mapping.")

        required_assets = []
        algos_dict = config.detectors_config.algorithms
        for algo in set(active_algos):
            if algo in algos_dict and hasattr(algos_dict[algo], "required_assets"):
                required_assets.extend(algos_dict[algo].required_assets.values())

        manager.verify_and_download(required_assets)

    else:
        # Run the smart config extraction directly
        manager.ensure_assets_for_config(config)


if __name__ == "__main__":
    entry_point()
