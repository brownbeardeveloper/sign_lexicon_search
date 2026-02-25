import functools
import yaml
from pathlib import Path

from . import _CONFIG_PATH


@functools.cache
def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Load and return the YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}
