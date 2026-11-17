"""Windows runtime setup for the project-local OpenSees Python environment."""
from __future__ import annotations

import os
import sys
from pathlib import Path


if sys.platform == "win32":
    _dll_handles = []
    _dll_dirs = [
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.base_prefix) / "Library" / "bin",
        Path.home() / "anaconda3" / "Library" / "bin",
    ]
    try:
        _base = Path(sys.base_prefix)
        if _base.parent.name.lower() == "pkgs":
            _dll_dirs.extend(_base.parent.glob("libexpat-*/Library/bin"))
    except (OSError, IndexError):
        pass

    for _dll_dir in _dll_dirs:
        try:
            if _dll_dir.is_dir():
                _dll_handles.append(os.add_dll_directory(str(_dll_dir)))
        except (OSError, ValueError):
            pass

    # Some Windows images omit this standard shell-folder value; pip's
    # platformdirs lookup otherwise aborts before reading PIP_CONFIG_FILE.
    try:
        import winreg

        _query_value_ex = winreg.QueryValueEx

        def _query_value_ex_with_common_data(key, name):
            try:
                return _query_value_ex(key, name)
            except FileNotFoundError:
                if name == "Common AppData":
                    return (os.environ.get("PROGRAMDATA", r"C:\ProgramData"), 1)
                raise

        winreg.QueryValueEx = _query_value_ex_with_common_data
    except (ImportError, OSError):
        pass
