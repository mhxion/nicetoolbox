"""napari_connector CLI entry point."""

import argparse
from pathlib import Path

from .tasks import import_body_joints

DEFAULT_IMPORT_BODY_JOINTS = Path("<configs_folder_path>/connectors/napari_import_body_joints.toml")


def entry_point() -> None:
    parser = argparse.ArgumentParser(prog="napari_connector", description="NICE napari connector")
    sub = parser.add_subparsers(dest="task", required=True)

    for name in ("import_body_joints",):
        p = sub.add_parser(name)
        p.add_argument("--project_folder_path", default=Path("."), type=Path)
        p.add_argument("--machine_specifics", default=Path("machine_specific_paths.toml"), type=Path)
        p.add_argument("--connector_config", type=Path)

    args = parser.parse_args()

    if args.task == "import_body_joints":
        connector_config = args.connector_config or DEFAULT_IMPORT_BODY_JOINTS
        import_body_joints(args.project_folder_path, args.machine_specifics, connector_config)


if __name__ == "__main__":
    entry_point()
