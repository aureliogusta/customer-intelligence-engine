from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hand_history_watcher import ingest_file as legacy_ingest_file


SCHEMA_VERSION = "2026.04.ingest.v1"


@dataclass(frozen=True)
class IngestionEnvelope:
    source_path: str
    source_hash: str
    schema_version: str = SCHEMA_VERSION


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def ingest_hand_file(filepath: Path, conn, hero_name: str, site: str = "auto") -> dict[str, Any]:
    envelope = IngestionEnvelope(source_path=str(filepath), source_hash=file_hash(filepath))
    inserted = legacy_ingest_file(filepath, conn, hero_name=hero_name, site=site)
    return {
        "inserted_hands": int(inserted),
        "envelope": envelope.__dict__,
    }
