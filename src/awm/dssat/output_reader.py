"""Cached reader for the newest complete-season DSSAT output block."""
from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from typing import Any

_DSSAT_SECTION_MARKER = b"*DSSAT Cropping System"
_TAIL_CHUNK_BYTES = 64 * 1024


class CachedDSSATOutputReader:
    """Keep terminal Summary.OUT separate from policy-facing daily state."""

    def __init__(
        self,
        *,
        summary_out: str,
        out_files: list[str],
        str_fields=None,
        date_fields=None,
        verbose: int = 0,
    ) -> None:
        self.summary_out = str(summary_out)
        self.out_files = [str(path) for path in out_files]
        self.str_fields = set(str_fields or ())
        self.date_fields = set(date_fields or ())
        self.verbose = int(verbose or 0)
        self.invalidate()
        self.refresh_count = 0
        self.disk_read_count = 0
        self.cache_hit_count = 0
        self.last_refresh_seconds = 0.0
        self.last_refresh_bytes = 0

    def invalidate(self) -> None:
        self._summary: dict[str, Any] = {}
        self._daily_by_yrdoy: dict[str, dict[str, Any]] = {}
        self._ready = False

    def _log(self, message: str) -> None:
        if self.verbose > 0:
            print(message)

    def _read_lines(self, path: str) -> list[str]:
        self.disk_read_count += 1
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except FileNotFoundError:
            self._log(f"DSSAT output missing: {path}")
            return []
        self.last_refresh_bytes += len(text.encode("utf-8", errors="replace"))
        return [line for line in text.splitlines() if line.strip()]

    def _read_latest_section_lines(self, path: str) -> list[str]:
        self.disk_read_count += 1
        try:
            with open(path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                position = handle.tell()
                buffer = b""
                while position > 0:
                    size = min(_TAIL_CHUNK_BYTES, position)
                    position -= size
                    handle.seek(position)
                    buffer = handle.read(size) + buffer
                    marker = buffer.rfind(_DSSAT_SECTION_MARKER)
                    if marker >= 0:
                        section = buffer[marker:]
                        self.last_refresh_bytes += len(section)
                        text = section.decode("utf-8", errors="replace")
                        return [line for line in text.splitlines() if line.strip()]
                self.last_refresh_bytes += len(buffer)
                text = buffer.decode("utf-8", errors="replace")
                return [line for line in text.splitlines() if line.strip()]
        except FileNotFoundError:
            self._log(f"DSSAT output missing: {path}")
            return []

    def _convert_value(self, key: str, token: str):
        text = str(token).strip()
        if key in self.str_fields:
            return text
        if key in self.date_fields:
            if text == "-99":
                return -99
            if text.isdigit() and len(text) in (5, 7):
                return int(text[-3:])
        try:
            value = float(text)
        except ValueError:
            return text
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return value

    @staticmethod
    def _header_tokens(line: str) -> list[str]:
        return [token.strip() for token in line.lstrip("@").split() if token.strip()]

    def _parse_summary(self) -> dict[str, Any]:
        lines = self._read_lines(self.summary_out)
        if not lines:
            return {}
        indices = [i for i, line in enumerate(lines) if line.lstrip().startswith("@")]
        if not indices:
            raise ValueError(f"Summary.OUT has no header: {self.summary_out}")
        index = indices[-1]
        headers = self._header_tokens(lines[index])
        data = [
            line
            for line in lines[index + 1 :]
            if line and not line.lstrip().startswith(("*", "!", "@"))
        ]
        if not data:
            raise ValueError(f"Summary.OUT has no data rows: {self.summary_out}")
        values = data[-1].split()
        if len(values) != len(headers):
            raise ValueError(
                "Summary.OUT row/header mismatch: "
                f"headers={len(headers)}, values={len(values)}"
            )
        return {
            key.replace(".", ""): self._convert_value(key, value)
            for key, value in zip(headers, values, strict=True)
        }

    def _parse_daily_file(self, path: str) -> dict[str, dict[str, Any]]:
        lines = self._read_latest_section_lines(path)
        if not lines:
            return {}
        index = next(
            (i for i, line in enumerate(lines) if line.lstrip().startswith("@")),
            None,
        )
        if index is None:
            return {}
        headers = self._header_tokens(lines[index])
        if "YEAR" not in headers or "DOY" not in headers:
            return {}
        year_index = headers.index("YEAR")
        doy_index = headers.index("DOY")
        rows: dict[str, dict[str, Any]] = {}
        for line in lines[index + 1 :]:
            stripped = line.strip()
            if not stripped or stripped.startswith(("*", "!", "@")):
                continue
            values = stripped.split()
            if len(values) < len(headers):
                continue
            if not values[year_index].isdigit() or not values[doy_index].isdigit():
                continue
            year = int(values[year_index])
            doy = int(values[doy_index])
            if not 1 <= doy <= 366:
                continue
            yrdoy = f"{year % 100:02d}{doy:03d}"
            rows[yrdoy] = {
                name: self._convert_value(name, values[i])
                for i, name in enumerate(headers)
            }
        return rows

    def refresh(self) -> None:
        started = time.perf_counter()
        self.last_refresh_bytes = 0
        summary = self._parse_summary()
        merged: dict[str, dict[str, Any]] = defaultdict(dict)
        for path in self.out_files:
            for yrdoy, row in self._parse_daily_file(path).items():
                merged[yrdoy].update(row)
        self._summary = summary
        self._daily_by_yrdoy = dict(merged)
        self._ready = True
        self.refresh_count += 1
        self.last_refresh_seconds = time.perf_counter() - started

    def daily_state(self, yrdoy: str) -> dict[str, Any]:
        key = str(yrdoy).strip()
        if len(key) != 5 or not key.isdigit():
            raise ValueError(f"yrdoy must be DSSAT YYDDD, got {yrdoy!r}")
        if not self._ready:
            self.refresh()
        if key not in self._daily_by_yrdoy:
            available = sorted(self._daily_by_yrdoy)
            bounds = f"{available[0]}..{available[-1]}" if available else "<empty>"
            raise KeyError(f"No DSSAT daily state for {key}; cached range={bounds}")
        self.cache_hit_count += 1
        return dict(self._daily_by_yrdoy[key])

    def season_summary(self) -> dict[str, Any]:
        if not self._ready:
            self.refresh()
        return dict(self._summary)

    @property
    def metrics(self) -> dict[str, float | int]:
        return {
            "refresh_count": int(self.refresh_count),
            "disk_read_count": int(self.disk_read_count),
            "cache_hit_count": int(self.cache_hit_count),
            "last_refresh_seconds": float(self.last_refresh_seconds),
            "last_refresh_bytes": int(self.last_refresh_bytes),
        }


__all__ = ["CachedDSSATOutputReader"]
