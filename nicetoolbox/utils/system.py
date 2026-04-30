"""
Helper functions for system-related operations.
"""

import logging
import platform

from .logging_utils import log_banner


def system_capability_check() -> None:
    """Checks current system-specific capabilities (i.e. max path length)"""
    check_long_path_support()


def check_long_path_support() -> None:
    """Check if Windows long path support is enabled and warn if not.

    On Windows, the default MAX_PATH limit is 260 characters. This can cause
    failures when dataset/session names produce long folder paths. Enabling
    long path support removes this limit.

    On Linux, this check is a no-op (no 260-char restriction).
    """
    if platform.system() != "Windows":
        return

    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        )
        value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        winreg.CloseKey(key)

        if value == 1:
            return
    except (FileNotFoundError, OSError):
        logging.warning("Can't read Windows keys registry to validate `LongPathsEnabled`!")

    log_banner(
        "Windows long path support is NOT enabled. "
        "Paths longer than 260 characters will fail. "
        "To enable, run PowerShell as Administrator:\n\n"
        'Set-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" '
        '-Name "LongPathsEnabled" -Value 1\n\n'
        "See: https://learn.microsoft.com/en-us/windows/win32/fileio/"
        "maximum-file-path-limitation",
        level=logging.WARNING,
    )


def detect_os_type() -> str:
    """
    Detects the underlying operating system.

    Returns:
        str: A string representing the operating system type.
            It can be either 'windows' or 'linux'.

    Notes:
        This function uses the platform module to determine the operating system.
        It checks the platform.system() function's return value and returns
        'windows' if it's 'Windows', and 'linux' if it's either 'Linux' or 'Darwin'.
    """
    if platform.system() == "Windows":
        os_type = "windows"
    elif platform.system() == "Linux" or platform.system() == "Darwin":
        os_type = "linux"
    return os_type
