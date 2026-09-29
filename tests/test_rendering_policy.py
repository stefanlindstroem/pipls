from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pipls
import pipls.inspection as inspection


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_runtime_package_exposes_no_plotting_api() -> None:
    assert importlib.util.find_spec("pipls.plotting") is None

    for module in (pipls, inspection):
        assert all(not name.startswith("plot_") for name in module.__all__)


def test_runtime_imports_without_rendering_dependencies() -> None:
    root = _repository_root()
    script = r'''
import builtins

blocked = {"matplotlib", "textalloc"}
original_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in blocked:
        raise ModuleNotFoundError(f"blocked optional dependency: {name}")
    return original_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded_import
import pipls
import pipls.inspection
assert pipls.PiPLSRegression is not None
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
