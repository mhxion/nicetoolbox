"""elan_connector CLI entry point."""

import argparse
from pathlib import Path

from .tasks import import_gaze

DEFAULT_IMPORT_GAZE = Path("<configs_folder_path>/connectors/elan_import_gaze.toml")


def entry_point() -> None:
    parser = argparse.ArgumentParser(prog="elan_connector", description="NICE ELAN connector")
    sub = parser.add_subparsers(dest="task", required=True)

    for name in ("import_gaze",):
        p = sub.add_parser(name)
        p.add_argument("--project_folder_path", default=Path("."), type=Path)
        p.add_argument("--machine_specifics", default=Path("machine_specific_paths.toml"), type=Path)
        p.add_argument("--connector_config", type=Path)

    args = parser.parse_args()

    if args.task == "import_gaze":
        connector_config = args.connector_config or DEFAULT_IMPORT_GAZE
        import_gaze(args.project_folder_path, args.machine_specifics, connector_config)


if __name__ == "__main__":
    entry_point()
