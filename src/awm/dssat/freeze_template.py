"""Freeze a known-good rendered DSSAT COX into an AWM irrigation template.

The source should come from a successful baseline/reset DSSAT worker. By
default this command refuses to strip existing explicit irrigation events
because they may contain non-policy management that requires human
classification before the AWM protocol is locked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .management import IRRIGATION_MARKER


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_cox_template(
    *,
    source_cox: str,
    output_template: str,
    report_path: str | None = None,
    allow_strip_existing_irrigation: bool = False,
) -> dict[str, object]:
    source = Path(source_cox)
    target = Path(output_template)
    if not source.is_file():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf-8")
    if IRRIGATION_MARKER in text:
        raise ValueError("source COX already contains AWM irrigation marker")

    lines = text.splitlines()
    header_index = _find_irrigation_table_header(lines)
    next_section = _find_next_section(lines, header_index + 1)

    body = lines[header_index + 1 : next_section]
    explicit_rows = [
        line
        for line in body
        if line.strip() and not line.lstrip().startswith(("!", "@"))
    ]
    if explicit_rows and not allow_strip_existing_irrigation:
        raise ValueError(
            "source COX contains explicit irrigation rows; freeze from a "
            "known-good no-policy baseline COX or pass "
            "--allow-strip-existing-irrigation after classifying them"
        )

    frozen = lines[: header_index + 1] + [IRRIGATION_MARKER] + lines[next_section:]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(frozen) + "\n", encoding="utf-8")

    report = {
        "source_cox": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output_template": str(target.resolve()),
        "template_sha256": sha256_file(target),
        "irrigation_header_line": header_index + 1,
        "stripped_explicit_irrigation_rows": explicit_rows,
        "stripped_explicit_irrigation_row_count": len(explicit_rows),
        "marker": IRRIGATION_MARKER,
        "allow_strip_existing_irrigation": bool(allow_strip_existing_irrigation),
        "status": "candidate_template_created",
        "warning": (
            "Candidate only until agronomic parameters and provenance are "
            "reviewed and locked in EXPERIMENT_PROTOCOL.md."
        ),
    }
    if report_path:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _find_irrigation_table_header(lines: list[str]) -> int:
    matches: list[int] = []
    for idx, line in enumerate(lines):
        tokens = line.strip().split()
        if len(tokens) >= 4 and tokens[:4] == ["@I", "IDATE", "IROP", "IRVAL"]:
            matches.append(idx)
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one '@I IDATE IROP IRVAL' irrigation table header"
        )
    return matches[0]


def _find_next_section(lines: list[str], start: int) -> int:
    for idx in range(start, len(lines)):
        stripped = lines[idx].lstrip()
        if stripped.startswith("*"):
            return idx
    raise ValueError("no DSSAT section found after irrigation table")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a successful rendered COX as an AWM template"
    )
    parser.add_argument("--source-cox", required=True)
    parser.add_argument("--output-template", required=True)
    parser.add_argument("--report")
    parser.add_argument(
        "--allow-strip-existing-irrigation",
        action="store_true",
        help=(
            "Explicitly acknowledge removal of existing irrigation rows. "
            "Prefer a clean baseline COX instead."
        ),
    )
    args = parser.parse_args()
    report = freeze_cox_template(
        source_cox=args.source_cox,
        output_template=args.output_template,
        report_path=args.report,
        allow_strip_existing_irrigation=args.allow_strip_existing_irrigation,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
