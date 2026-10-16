"""The unpacked dataset on disk: safe paths and the release manifest's hashes.

The dataset is course material and lives outside the repo (`config.dataset_root`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, ValidationError

from yifinance_rag.config import Settings, dataset_root

RELEASE_MANIFEST = "data/releases/flowpay_dataset_v1/release_manifest.json"


class DataError(Exception):
    """The dataset is missing a file or a file is not in the expected format."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Artifact(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    path: str
    sha256: str
    bytes: int


class ArtifactGroup(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    group_id: str
    visibility: str
    file_count: int
    artifacts: list[Artifact]


class Release(BaseModel):
    """The fields of release_manifest.json we use (the course reads the same ones)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    status: str
    release_id: str
    artifact_groups: list[ArtifactGroup]


@dataclass(frozen=True)
class Dataset:
    root: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> Dataset:
        return cls(dataset_root(settings))

    def path(self, relative: str) -> Path:
        """A file inside the dataset; a relative path that escapes the root is refused."""
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts or "\\" in relative:
            raise DataError(f"not a dataset-relative path: {relative!r}")
        return self.root / relative

    def read_text(self, relative: str) -> str:
        path = self.path(relative)
        if not path.is_file():
            raise DataError(f"missing {relative} under {self.root}")
        return path.read_text(encoding="utf-8")

    def read_jsonl(self, relative: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for n, line in enumerate(self.read_text(relative).splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise DataError(f"{relative}:{n} is not a JSON object")
            rows.append(row)
        return rows

    def release(self) -> Release:
        try:
            return Release.model_validate_json(self.read_text(RELEASE_MANIFEST))
        except ValidationError as e:
            raise DataError(f"{RELEASE_MANIFEST} has an unexpected shape: {e}") from e

    def verify_release(self) -> list[str]:
        """Problems with the files the release manifest lists (size and sha256 of each)."""
        problems: list[str] = []
        release = self.release()
        if release.status != "frozen":
            problems.append(f"release status is {release.status!r}, expected 'frozen'")
        for group in release.artifact_groups:
            if len(group.artifacts) != group.file_count:
                problems.append(
                    f"group {group.group_id}: lists {len(group.artifacts)} files, "
                    f"declares {group.file_count}"
                )
            for artifact in group.artifacts:
                path = self.path(artifact.path)
                if not path.is_file():
                    problems.append(f"missing {artifact.path}")
                elif path.stat().st_size != artifact.bytes:
                    problems.append(f"size differs: {artifact.path}")
                elif sha256_file(path) != artifact.sha256:
                    problems.append(f"sha256 differs: {artifact.path}")
        return problems
