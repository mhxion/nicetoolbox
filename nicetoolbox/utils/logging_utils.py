"""
Helper functions for logging.
"""

import logging
import sys
from pathlib import Path

LOGGING_DEFAULT = logging.INFO
LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s"


def init_console_logging() -> None:
    """
    Initialize the logging before we loaded the configuration.
    This will log it only as a console output as we don't know log level or output path.
    """
    logging.basicConfig(level=LOGGING_DEFAULT, format=LOGGING_FORMAT)


def init_file_logging(log_path: Path, level: int | str = logging.INFO) -> None:
    """
    Initialize the logging console and file output with specific logging level.

    Args:
        log_path (str): The path to the log file.
        level (int | str, optional): Determines from which level the logger will record the
            messages.
            For instance, when the level is set as logging.INFO, the messages with a
            severity below INFO (i.e. DEBUG) will be ignored.
            The possible levels are:
                - logging.DEBUG: Detailed information, typically of interest only when
                diagnosing problems.
                - logging.INFO: Confirmation that things are working as expected.
                - logging.WARNING: An indication that something unexpected happened, or
                indicative of some problem in the near future (e.g. 'disk space low').
                    The software is still working as expected.
                - logging.ERROR: Due to a more serious problem, the software has not
                been able to perform some function.
                - logging.CRITICAL: A serious error, indicating that the program itself
                may be unable to continue running.
    """
    # ensure log file parent folder exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Important to start log in "attach" mode
    # With "write" mode it will corrupt log output as subprocesses will write to it too
    # To reset log from old sessions, we manually delete old log first
    log_path.unlink(missing_ok=True)
    logging.basicConfig(
        level=level,
        format=LOGGING_FORMAT,
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler(sys.stdout)],
        force=True,  # force=True removes any existing handlers (i.e. previous console log)
    )


def abbrev_list(labels: list, n: int = 5) -> list:
    """Return a truncated list with an ellipsis marker when it exceeds n items."""
    return labels[:n] + ["..."] if len(labels) > n else labels


def assert_and_log(condition, message):
    """
    Asserts a condition and logs an error message if the condition is not met.

    Args:
        condition (bool): The condition to be checked.
        message (str): The error message to be logged if the condition is not met.

    Returns:
        None

    Raises:
        AssertionError: If the condition is not met.
        SystemExit: If the condition is not met, the function will terminate the
            program with a status code of 1.
    """
    try:
        assert condition, message
    except AssertionError as e:
        logging.error(f"Assertion failed: {e}")
        sys.exit(1)


def log_main_banner(msg: str, banner_sym: str = "#", level=logging.INFO) -> None:
    banner = banner_sym * 80
    logging.log(level, f"\n{banner}\n\n{msg}\n\n{banner}\n\n", stacklevel=3)


def log_banner(msg: str, banner_sym: str = "=", level=logging.INFO) -> None:
    banner = banner_sym * 80
    logging.log(level, f"\n{banner}\n{msg}\n{banner}\n\n", stacklevel=3)


def log_with_underscore(msg: str, underline_sym: str = "-", level=logging.INFO) -> None:
    logging.log(level, f"{msg}\n{underline_sym * 80}", stacklevel=3)
