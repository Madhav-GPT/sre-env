"""JSONL trajectory storage and summary helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .types import EpisodeRecord


class TrajectoryStore:
    """Append-only JSONL store for episode trajectories."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_episode(self, record: EpisodeRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")

    def load_episodes(self) -> list[EpisodeRecord]:
        if not self.path.exists():
            return []
        records: list[EpisodeRecord] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(EpisodeRecord.model_validate(json.loads(line)))
        return records
