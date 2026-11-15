"""Append-only JSONL log with resume support."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
Row = dict[str, JsonValue]


class JsonlLog:
    """Append-only JSONL file of rows. Rows with a `key` can be looked up, so a rerun skips finished work.

    A row may also name its parent in `belongs_to` (the `key` of another row). On load, rows whose
    parent is missing are ignored, so a write that was cut off before the parent row was saved
    leaves no half-finished group behind. A torn last line is cut off the file on load. A repeated
    `key` is an error.
    """

    def __init__(self, path: Path | str) -> None:
        """Open `path`, creating its folder, and load any rows already saved there."""
        self.path: Path = Path(path)
        _ = self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[Row] = []
        self._by_key: dict[str, Row] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        """Read the file, dropping a torn last line and rows whose parent row is missing.

        If anything is dropped, the file is rewritten without it, so dropped rows can never be
        picked up again later (for example by a re-run that saves the missing parent row).
        """
        lines = self.path.read_text(encoding="utf-8").split("\n")
        good: list[Row] = []
        dropped = False
        for i, line in enumerate(lines):
            if not line:
                continue
            try:
                good.append(cast("Row", json.loads(line)))
            except json.JSONDecodeError:
                if any(lines[i + 1 :]):
                    raise
                dropped = True  # torn last line from an interrupted write
        keys = {k for r in good if isinstance(k := r.get("key"), str)}
        kept = [
            r
            for r in good
            if not isinstance(p := r.get("belongs_to"), str) or p in keys
        ]
        if dropped or len(kept) != len(good):
            _ = self.path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                encoding="utf-8",
            )
        for row in kept:
            self._add(row)

    def _add(self, row: Row) -> None:
        """Remember a row in memory, indexed by its `key` if it has one."""
        key = row.get("key")
        if isinstance(key, str):
            if key in self._by_key:
                raise ValueError(f"duplicate key in {self.path}: {key}")
            self._by_key[key] = row
        self.rows.append(row)

    def get(self, key: str) -> Row | None:
        """The saved row with this key, or None."""
        return self._by_key.get(key)

    def append(self, row: Row) -> None:
        """Remember a row and write it to the end of the file."""
        self.extend([row])

    def extend(self, rows: list[Row]) -> None:
        """Remember several rows and write them to the end of the file in one write."""
        for row in rows:
            self._add(row)
        with self.path.open("a", encoding="utf-8") as f:
            _ = f.write(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            f.flush()
            os.fsync(f.fileno())
