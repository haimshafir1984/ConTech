import os
import sys
from importlib import import_module
from types import ModuleType


def _ensure_project_root_on_path() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def load_root_module(module_name: str) -> ModuleType:
    _ensure_project_root_on_path()
    return import_module(module_name)
