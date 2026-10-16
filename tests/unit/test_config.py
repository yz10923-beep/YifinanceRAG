from __future__ import annotations

from pathlib import Path

import pytest

from yifinance_rag.config import (
    ConfigError,
    Settings,
    cache_dir,
    dataset_root,
    default_dotenv_paths,
    load_settings,
)


def _fake_dataset(parent: Path) -> Path:
    root = parent / "flowpay-dataset"
    (root / "data").mkdir(parents=True)
    (root / "benchmark").mkdir()
    return root


def test_defaults() -> None:
    s = load_settings(env={}, use_default_dotenv=False)
    assert s.anthropic_api_key is None
    assert s.dataset_root is None
    assert s.pg_url.get_secret_value().startswith("postgresql://")
    assert "127.0.0.1:5432" in s.pg_url.get_secret_value()
    assert s.ollama_url == "http://127.0.0.1:11434"
    assert cache_dir(s) == Path.home() / ".yfr" / "cache"


def test_dataset_root_read_unprefixed_and_tilde_expanded(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("FLOWPAY_DATASET_ROOT=~/flowpay-dataset\n")
    s = load_settings(env={}, dotenv_path=dotenv)
    assert s.dataset_root == Path.home() / "flowpay-dataset"


def test_env_overrides_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("FLOWPAY_DATASET_ROOT=/from-dotenv\nYFR_PG_URL=postgresql://d/x\n")
    s = load_settings(env={"FLOWPAY_DATASET_ROOT": "/from-env"}, dotenv_path=dotenv)
    assert s.dataset_root == Path("/from-env")  # env wins
    assert s.pg_url.get_secret_value() == "postgresql://d/x"  # dotenv fills the rest


def test_the_shipped_env_example_loads() -> None:
    example = Path(__file__).parents[2] / ".env.example"
    s = load_settings(env={}, dotenv_path=example)
    assert s.anthropic_api_key is None  # ANTHROPIC_API_KEY= (empty) means unset
    assert s.dataset_root == Path.home() / "flowpay-dataset"


def test_unrelated_and_unknown_vars_are_ignored() -> None:
    s = load_settings(
        env={"YFR_NOT_A_SETTING": "x", "POSTGRES_PASSWORD": "p", "PATH": "/bin"},
        use_default_dotenv=False,
    )
    assert s == load_settings(env={}, use_default_dotenv=False)


def test_prefixed_names_are_not_a_second_way_in() -> None:
    s = load_settings(
        env={"YFR_ANTHROPIC_API_KEY": "k", "YFR_DATASET_ROOT": "/x"}, use_default_dotenv=False
    )
    assert s.anthropic_api_key is None and s.dataset_root is None


def test_secrets_hidden_in_repr(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("ANTHROPIC_API_KEY=sk-test-123\nYFR_PG_URL=postgresql://u:pw-456@h/d\n")
    s = load_settings(env={}, dotenv_path=dotenv)
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-test-123"
    for text in (repr(s), s.model_dump_json()):
        assert "sk-test-123" not in text and "pw-456" not in text


def test_cwd_dotenv_overrides_home_dotenv(tmp_path: Path) -> None:
    home, local = tmp_path / "home.env", tmp_path / "local.env"
    home.write_text("ANTHROPIC_API_KEY=from-home\nFLOWPAY_DATASET_ROOT=/home-data\n")
    local.write_text("FLOWPAY_DATASET_ROOT=/local-data\n")
    s = load_settings(env={}, dotenv_path=[home, local])
    assert s.dataset_root == Path("/local-data")
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "from-home"


def test_default_dotenv_paths_are_home_then_cwd() -> None:
    home, cwd = default_dotenv_paths()
    assert home == Path.home() / ".yfr" / ".env" and cwd == Path(".env")


def test_default_load_reads_the_home_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    (Path.home() / ".yfr").mkdir()
    (Path.home() / ".yfr" / ".env").write_text("ANTHROPIC_API_KEY=from-home-file\n")
    monkeypatch.chdir(Path.home())  # no ./.env here
    key = load_settings(env={}).anthropic_api_key
    assert key is not None and key.get_secret_value() == "from-home-file"


@pytest.mark.parametrize(
    "data",
    [
        b"ANTHROPIC_API_KEY=k\r\n",
        "﻿ANTHROPIC_API_KEY=k\n".encode(),
        b'ANTHROPIC_API_KEY="k"\n',
    ],
)
def test_windows_style_dotenv_files_load(tmp_path: Path, data: bytes) -> None:
    p = tmp_path / ".env"
    p.write_bytes(data)
    key = load_settings(env={}, dotenv_path=p).anthropic_api_key
    assert key is not None and key.get_secret_value() == "k"


def test_utf16_dotenv_is_a_clear_config_error(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_bytes("ANTHROPIC_API_KEY=k\r\n".encode("utf-16"))
    with pytest.raises(ConfigError, match="UTF-8"):
        load_settings(env={}, dotenv_path=p)


# --- dataset_root: the checked path every loader starts from ---


def test_dataset_root_accepts_the_unpacked_folder(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path)
    s = Settings(dataset_root=root)
    assert dataset_root(s, repo_root=tmp_path / "repo") == root.resolve()


def test_dataset_root_unset_is_a_clear_error() -> None:
    with pytest.raises(ConfigError, match="FLOWPAY_DATASET_ROOT is not set"):
        dataset_root(Settings())


def test_dataset_root_pointing_at_the_zip_says_unzip(tmp_path: Path) -> None:
    z = tmp_path / "flowpay-dataset.zip"
    z.write_bytes(b"PK")
    with pytest.raises(ConfigError, match="unzip"):
        dataset_root(Settings(dataset_root=z))


def test_dataset_root_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a folder"):
        dataset_root(Settings(dataset_root=tmp_path / "nope"))


def test_dataset_root_one_level_too_high_is_caught(tmp_path: Path) -> None:
    _fake_dataset(tmp_path)  # tmp_path holds flowpay-dataset/, it isn't the dataset
    with pytest.raises(ConfigError, match="data/, benchmark/"):
        dataset_root(Settings(dataset_root=tmp_path), repo_root=tmp_path / "repo")


def test_dataset_inside_the_repo_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = _fake_dataset(repo / "data")
    with pytest.raises(ConfigError, match="outside the repo"):
        dataset_root(Settings(dataset_root=root), repo_root=repo.resolve())
