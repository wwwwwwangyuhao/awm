"""Cached reader for DSSAT OUT files.

One DSSAT execution writes a complete season. Re-scanning every OUT file for
every RL step is unnecessary, and repeatedly scanning historical DSSAT blocks
from the beginning is also wasteful. The reader therefore:

1. reads only the newest DSSAT block from the tail of each daily OUT file;
2. parses that block once after a DSSAT execution;
3. indexes daily rows in memory for O(1) no-rerun step lookup;
4. keeps seasonal Summary.OUT data separate from policy daily state.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import numpy as np


_DSSAT_SECTION_MARKER = b"*DSSAT Cropping System"
_TAIL_CHUNK_BYTES = 64 * 1024


class CachedDSSATOutputReader:
    def __init__(
        self,
        *,
        summary_out: str,
        out_files: list[str],
        str_fields=None,
        date_fields=None,
        verbose: int = 0,
    ):
        self.summary_out = str(summary_out)
        self.out_files = [str(path) for path in out_files]
        self.str_fields = set(str_fields or ())
        self.date_fields = set(date_fields or ())
        self.verbose = int(verbose or 0)

        self._summary: dict[str, Any] = {}
        self._daily_by_yrdoy: dict[str, dict[str, Any]] = {}
        self._ready = False

        self.refresh_count = 0
        self.disk_read_count = 0
        self.cache_hit_count = 0
        self.last_refresh_seconds = 0.0
        self.last_refresh_bytes = 0

    def invalidate(self) -> None:
        """Mark cached output stale before a new DSSAT execution/episode."""
        self._summary = {}
        self._daily_by_yrdoy = {}
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
        """Read only the newest DSSAT run block by scanning backward from EOF."""
        self.disk_read_count += 1
        try:
            with open(path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                position = handle.tell()
                buffer = b""

                while position > 0:
                    read_size = min(_TAIL_CHUNK_BYTES, position)
                    position -= read_size
                    handle.seek(position)
                    buffer = handle.read(read_size) + buffer
                    marker_index = buffer.rfind(_DSSAT_SECTION_MARKER)
                    if marker_index >= 0:
                        section = buffer[marker_index:]
                        self.last_refresh_bytes += len(section)
                        text = section.decode("utf-8", errors="replace")
                        return [
                            line
                            for line in text.splitlines()
                            if line.strip()
                        ]

                # Some DSSAT-style diagnostic files may not contain the normal
                # section marker. In that case the backward scan has already
                # collected the complete file; parse it as a single section.
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
                return float(text)
            except ValueError:
                return text
        try:
            value = float(text)
        except ValueError:
            return text
        if np.isfinite(value) and value.is_integer():
            return int(value)
        return value

    @staticmethod
    def _header_tokens(header_line: str) -> list[str]:
        return [
            token.strip()
            for token in header_line.lstrip("@").split()
            if token.strip()
        ]

    def _parse_summary(self) -> dict[str, Any]:
        lines = self._read_lines(self.summary_out)
        if not lines:
            return {}
        # Summary.OUT can contain repeated experiment sections. Use the final
        # header followed by its final data row rather than assuming the first
        # header belongs to the newest DSSAT execution.
        header_indices = [
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith("@")
        ]
        if not header_indices:
            raise ValueError(f"Summary.OUT has no header: {self.summary_out}")
        header_index = header_indices[-1]
        headers = self._header_tokens(lines[header_index].lstrip())
        data_lines = [
            line
            for line in lines[header_index + 1 :]
            if line and not line.lstrip().startswith(("*", "!", "@"))
        ]
        if not data_lines:
            raise ValueError(f"Summary.OUT has no data rows: {self.summary_out}")
        values = data_lines[-1].split()
        if len(values) != len(headers):
            raise ValueError(
                "Summary.OUT row/header mismatch: "
                f"headers={len(headers)}, values={len(values)}."
            )
        return {
            key.replace(".", ""): self._convert_value(key, value)
            for key, value in zip(headers, values, strict=True)
        }

    def _parse_daily_file(self, path: str) -> dict[str, dict[str, Any]]:
        lines = self._read_latest_section_lines(path)
        if not lines:
            return {}
        header_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.lstrip().startswith("@")
            ),
            None,
        )
        if header_index is None:
            self._log(f"DSSAT output has no table header: {path}")
            return {}
        headers = self._header_tokens(lines[header_index].lstrip())
        if "YEAR" not in headers or "DOY" not in headers:
            self._log(f"DSSAT daily output has no YEAR/DOY: {path}")
            return {}
        year_idx = headers.index("YEAR")
        doy_idx = headers.index("DOY")
        min_columns = max(year_idx, doy_idx) + 1

        rows: dict[str, dict[str, Any]] = {}
        for line in lines[header_index + 1 :]:
            stripped = line.strip()
            if not stripped or stripped.startswith(("*", "!", "@")):
                continue
            values = stripped.split()
            if len(values) < min_columns:
                continue
            year_text = values[year_idx].strip()
            doy_text = values[doy_idx].strip()
            if not year_text.isdigit() or not doy_text.isdigit():
                continue
            calendar_year = int(year_text)
            doy = int(doy_text)
            if not 1 <= doy <= 366:
                continue
            yrdoy = f"{calendar_year % 100:02d}{doy:03d}"

            if len(values) < len(headers):
                self._log(
                    f"Skipping malformed DSSAT row in {os.path.basename(path)}: "
                    f"{len(values)} values for {len(headers)} headers."
                )
                continue
            row = {
                key: self._convert_value(key, values[index])
                for index, key in enumerate(headers)
            }
            rows[yrdoy] = row
        return rows

    def refresh(self) -> None:
        """Parse the newest DSSAT execution once and replace the memory cache."""
        started = time.perf_counter()
        self.last_refresh_bytes = 0
        summary = self._parse_summary()
        merged: dict[str, dict[str, Any]] = defaultdict(dict)
        for path in self.out_files:
            rows = self._parse_daily_file(path)
            for yrdoy, row in rows.items():
                merged[yrdoy].update(row)

        self._summary = summary
        self._daily_by_yrdoy = dict(merged)
        self._ready = True
        self.refresh_count += 1
        self.last_refresh_seconds = time.perf_counter() - started

    @staticmethod
    def _validate_yrdoy(yrdoy: str) -> str:
        key = str(yrdoy).strip()
        if len(key) != 5 or not key.isdigit():
            raise ValueError(f"yrdoy must be DSSAT YYDDD, got {yrdoy!r}.")
        return key

    def daily_state(self, yrdoy: str) -> dict[str, Any]:
        """Return only decision-time daily fields for one YYDDD date."""
        key = self._validate_yrdoy(yrdoy)
        if not self._ready:
            self.refresh()
        daily = self._daily_by_yrdoy.get(key)
        if daily is None:
            available = sorted(self._daily_by_yrdoy)
            bounds = (
                f"{available[0]}..{available[-1]}"
                if available
                else "<empty>"
            )
            raise KeyError(
                f"No DSSAT daily state for {key}; cached range={bounds}."
            )
        self.cache_hit_count += 1
        return dict(daily)

    def season_summary(self) -> dict[str, Any]:
        """Return mature/seasonal Summary.OUT fields in a separate namespace."""
        if not self._ready:
            self.refresh()
        return dict(self._summary)

    def process_OUT_main(self, yrdoy: str) -> dict[str, Any]:
        """Legacy compatibility API; new code must not use this for observation."""
        return {**self.season_summary(), **self.daily_state(yrdoy)}

    def process_summary_out(self) -> dict[str, Any]:
        return self.season_summary()

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
