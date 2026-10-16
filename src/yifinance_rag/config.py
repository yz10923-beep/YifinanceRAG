"""Settings. Priority (low -> high): built-in defaults -> ~/.yfr/.env -> ./.env ->
process env vars.

~/.yfr/.env is the home for secrets like the API key, so the code finds them from any
directory. ./.env (current directory) overrides it per checkout.

Only settings the code reads live here. POSTGRES_PASSWORD is read by docker compose, not
by Python, so it is not a field (it is inside `pg_url`).

The FlowPay dataset is course material and is never committed: it lives outside the repo
at FLOWPAY_DATASET_ROOT, the unpacked `flowpay-dataset/` folder.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator

ENV_PREFIX = "YFR_"
# Read without the prefix: the key's usual name, and the dataset root named in CLAUDE.md.
_UNPREFIXED = {
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "FLOWPAY_DATASET_ROOT": "dataset_root",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
# Top-level entries of the unpacked dataset that the code relies on.
DATASET_MARKERS = ("data", "benchmark")


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    anthropic_api_key: SecretStr | None = None
    dataset_root: Path | None = None
    # The local pgvector from docker-compose.yml. A secret: it carries the password.
    pg_url: SecretStr = SecretStr("postgresql://flowpay:flowpay@127.0.0.1:5432/flowpay")

    @field_validator("dataset_root")
    @classmethod
    def _expand_home(cls, v: Path | None) -> Path | None:
        # python-dotenv leaves "~" as is; expand it here so every caller gets a real path.
        return None if v is None else v.expanduser()


class ConfigError(Exception):
    pass


def _from_env(env: Mapping[str, str | None]) -> dict[str, str]:
    fields = Settings.model_fields
    out: dict[str, str] = {}
    for key, value in env.items():
        if not value:  # unset and empty both mean "use the default"
            continue
        if key in _UNPREFIXED:
            out[_UNPREFIXED[key]] = value
            continue
        if not key.startswith(ENV_PREFIX):
            continue
        name = key.removeprefix(ENV_PREFIX).lower()
        if name in fields and name not in _UNPREFIXED.values():
            out[name] = value
    return out


def default_dotenv_paths() -> list[Path]:
    """Lowest priority first."""
    return [Path.home() / ".yfr" / ".env", Path(".env")]


def _read_dotenv(path: Path) -> dict[str, str]:
    try:
        return _from_env(dotenv_values(path, encoding="utf-8-sig"))  # -sig: tolerate a BOM
    except UnicodeDecodeError as e:
        # Typically `echo ... > .env` in Windows PowerShell 5.1, which writes UTF-16.
        raise ConfigError(f"{path} is not UTF-8 text; re-save it as UTF-8") from e


def load_settings(
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | list[Path] | None = None,
    *,
    use_default_dotenv: bool = True,
) -> Settings:
    """Merge .env files and environment variables over defaults.

    `dotenv_path` replaces the default .env locations (tests pass explicit files);
    `use_default_dotenv=False` with no path reads no .env at all.
    Raises ConfigError on unreadable files or invalid values.
    """
    if dotenv_path is None:
        paths = default_dotenv_paths() if use_default_dotenv else []
    else:
        paths = dotenv_path if isinstance(dotenv_path, list) else [dotenv_path]
    merged: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            merged |= _read_dotenv(path)
    merged |= _from_env(os.environ if env is None else env)
    try:
        return Settings.model_validate(merged)
    except ValidationError as e:
        raise ConfigError(str(e)) from e


def dataset_root(settings: Settings, *, repo_root: Path = REPO_ROOT) -> Path:
    """The unpacked dataset folder, checked: set, a folder with the expected layout, and
    outside this repo (so it can't be committed by accident).

    Only the layout is checked here; file hashes are checked by the loaders (S1).
    """
    root = settings.dataset_root
    if root is None:
        raise ConfigError(
            "FLOWPAY_DATASET_ROOT is not set; point it at the unpacked flowpay-dataset/ folder "
            "(e.g. `mkdir -p ~/.yfr && cp .env.example ~/.yfr/.env`, which sets ~/flowpay-dataset)"
        )
    if root.suffix.lower() == ".zip":
        raise ConfigError(f"{root} is the zip; unzip it and point at the flowpay-dataset/ folder")
    if not root.is_dir():
        raise ConfigError(f"FLOWPAY_DATASET_ROOT={root} is not a folder")
    root = root.resolve()
    if root == repo_root or repo_root in root.parents:
        raise ConfigError(f"the dataset must live outside the repo, not at {root}")
    missing = [m for m in DATASET_MARKERS if not (root / m).is_dir()]
    if missing:
        listed = ", ".join(f"{m}/" for m in missing)
        raise ConfigError(f"{root} has no {listed}; expected the flowpay-dataset/ folder itself")
    return root
