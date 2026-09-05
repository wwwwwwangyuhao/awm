from pathlib import Path

import pytest

from awm.dssat.freeze_template import freeze_cox_template
from awm.dssat.management import IRRIGATION_MARKER


COX = """*EXP.DETAILS: TEST
*IRRIGATION AND WATER MANAGEMENT
@I  EFIR  IDEP
 1   -99   -99
@I IDATE  IROP IRVAL

*FERTILIZERS (INORGANIC)
@F FDATE FMCD
 1 25119 FE010
"""

COX_WITH_EVENT = """*EXP.DETAILS: TEST
*IRRIGATION AND WATER MANAGEMENT
@I IDATE  IROP IRVAL
 1 25120 IR005 10.00
*FERTILIZERS (INORGANIC)
@F FDATE FMCD
"""


def test_freeze_clean_cox_inserts_one_marker(tmp_path: Path):
    source = tmp_path / "source.COX"
    output = tmp_path / "template.COX.in"
    report = tmp_path / "report.json"
    source.write_text(COX, encoding="utf-8")
    result = freeze_cox_template(
        source_cox=str(source),
        output_template=str(output),
        report_path=str(report),
    )
    text = output.read_text(encoding="utf-8")
    assert text.count(IRRIGATION_MARKER) == 1
    assert "*FERTILIZERS (INORGANIC)" in text
    assert "25119 FE010" in text
    assert result["stripped_explicit_irrigation_row_count"] == 0
    assert report.is_file()


def test_freeze_refuses_unclassified_existing_irrigation(tmp_path: Path):
    source = tmp_path / "source.COX"
    output = tmp_path / "template.COX.in"
    source.write_text(COX_WITH_EVENT, encoding="utf-8")
    with pytest.raises(ValueError, match="explicit irrigation rows"):
        freeze_cox_template(
            source_cox=str(source),
            output_template=str(output),
        )


def test_freeze_can_strip_only_with_explicit_acknowledgement(tmp_path: Path):
    source = tmp_path / "source.COX"
    output = tmp_path / "template.COX.in"
    source.write_text(COX_WITH_EVENT, encoding="utf-8")
    result = freeze_cox_template(
        source_cox=str(source),
        output_template=str(output),
        allow_strip_existing_irrigation=True,
    )
    text = output.read_text(encoding="utf-8")
    assert "25120 IR005" not in text
    assert IRRIGATION_MARKER in text
    assert result["stripped_explicit_irrigation_row_count"] == 1
