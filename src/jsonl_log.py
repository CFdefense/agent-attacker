"""Append-only JSONL log with resume support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
Row = dict[str, JsonValue]


class JsonlLog:
    """Append-only JSONL file of rows. Rows with a `key` can be looked up, so a rerun skips finished work."""

    def __init__(self, path: Path | str) -> None:
        """Open `path`, creating its folder, and load any rows already saved there."""
        self.path: Path = Path(path)
        _ = self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[Row] = []
        self._by_key: dict[str, Row] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line:
                    self._add(cast("Row", json.loads(line)))

    def _add(self, row: Row) -> None:
        """Remember a row in memory, indexed by its `key` if it has one."""
        self.rows.append(row)
        key = row.get("key")
        if isinstance(key, str):
            self._by_key[key] = row

    def get(self, key: str) -> Row | None:
        """The saved row with this key, or None."""
        return self._by_key.get(key)

    def append(self, row: Row) -> None:
        """Remember a row and write it to the end of the file."""
        self._add(row)
        with self.path.open("a", encoding="utf-8") as f:
            _ = f.write(json.dumps(row, ensure_ascii=False) + "\n")
