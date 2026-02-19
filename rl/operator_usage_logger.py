"""Lightweight per-iteration operator usage logging utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OperatorUsageLogger:
    """Accumulates RL-controlled ALNS iteration metadata & persists it once per episode.

    The logger buffers per-iteration dictionaries in-memory and writes them to ``logs/``
    when ``flush`` is called. The default format is CSV because it is easy to inspect in a
    text editor and can be ingested directly by tools such as pandas.

    Example analysis snippet (CSV):
        >>> import pandas as pd
        >>> df = pd.read_csv("logs/operator_usage_train_20250101_120000.csv")
        >>> df.groupby(["operator_type", "operator_name"]).size().sort_values()

    Example analysis snippet (JSONL):
        >>> import pandas as pd
        >>> df = pd.read_json("logs/operator_usage_eval_20250101_120000.jsonl", lines=True)
    """

    mode: str = "train"
    output_dir: Path = Path("logs")
    fmt: str = "csv"
    enabled: bool = True
    _buffer: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _fieldnames: Optional[List[str]] = field(default=None, init=False, repr=False)
    _file_path: Optional[Path] = field(default=None, init=False, repr=False)

    def start_episode(self) -> None:
        if not self.enabled:
            return
        self._buffer.clear()

    def append(self, row: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._fieldnames is None:
            # Preserve insertion order from first row for consistent CSV header.
            self._fieldnames = list(row.keys())
        else:
            for key in row.keys():
                if key not in self._fieldnames:
                    self._fieldnames.append(key)
        self._buffer.append(row)

    def flush(self) -> Optional[Path]:
        if not self.enabled or not self._buffer:
            if self.enabled:
                self._buffer.clear()
            return self._file_path

        file_path = self._ensure_file()

        if self.fmt.lower() == "csv":
            self._write_csv(file_path)
        elif self.fmt.lower() in {"jsonl", "jsonlines"}:
            self._write_jsonl(file_path)
        else:
            raise ValueError(f"Unsupported operator usage log format: {self.fmt}")

        self._buffer.clear()
        return file_path

    def _ensure_file(self) -> Path:
        if self._file_path is not None:
            return self._file_path

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        suffix = ".jsonl" if self.fmt.lower() in {"jsonl", "jsonlines"} else ".csv"
        filename = f"operator_usage_{self.mode}_{timestamp}{suffix}"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self.output_dir / filename
        return self._file_path

    def _write_csv(self, file_path: Path) -> None:
        fieldnames = self._fieldnames or sorted(self._buffer[0].keys())
        file_exists = file_path.exists()

        with file_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists or file_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerows(self._buffer)

    def _write_jsonl(self, file_path: Path) -> None:
        with file_path.open("a", encoding="utf-8") as handle:
            for row in self._buffer:
                json.dump(row, handle)
                handle.write("\n")

    def reset(self) -> None:
        """Clear buffered rows and forget target file."""
        self._buffer.clear()
        self._file_path = None
        self._fieldnames = None


__all__ = ["OperatorUsageLogger"]
