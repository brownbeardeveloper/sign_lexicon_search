from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yml"
_SIGNS_JSON = _PROJECT_ROOT / "dataset" / "signs.json"