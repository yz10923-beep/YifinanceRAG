from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defaults under ~/.yfr must never touch the real home directory: every test (and
    every subprocess it starts) gets its own HOME (USERPROFILE on Windows)."""
    home = str(tmp_path_factory.mktemp("home"))
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
