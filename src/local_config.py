"""Read provider overrides without exporting them to child processes."""

import json
import os
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config.local.json"


class ConfigError(ValueError):
    """Invalid or inaccessible local configuration."""


def load_settings() -> dict[str, str]:
    override = os.environ.get("JCF_LOCAL_CONFIG")
    path = Path(override).expanduser() if override else DEFAULT_PATH
    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        if not override:
            return {}
        raise ConfigError("The configured JCF_LOCAL_CONFIG file was not found.") from None
    except (OSError, UnicodeError, ValueError):
        raise ConfigError("Unable to read the local JSON configuration.") from None

    settings = document.get("settings") if isinstance(document, dict) else None
    if not isinstance(settings, dict) or any(
        not isinstance(value, str) for value in settings.values()
    ):
        raise ConfigError("The local configuration must contain a settings object with string values.")
    return settings


def setting(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    return load_settings().get(name) or default
