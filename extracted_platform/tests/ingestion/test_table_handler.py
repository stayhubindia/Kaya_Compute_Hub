"""
Unit tests for Table Handler and Markdown Table Conversion (Phase 3.3).
"""

import pytest
from src.ingestion.table_handler import TableHandler


def test_markdown_table_parsing():
    md_text = """
    | Constant | Symbol | Value | Units |
    |---|---|---|---|
    | Speed of Light | c | 299792458 | m/s |
    | Planck Constant | h | 6.626e-34 | J s |
    """
    handler = TableHandler(doc_id="tbl_doc")
    tbl = handler.parse_markdown_table(md_text)

    assert tbl is not None
    assert tbl.headers == ["Constant", "Symbol", "Value", "Units"]
    assert len(tbl.rows) == 2
    assert tbl.rows[0] == ["Speed of Light", "c", "299792458", "m/s"]
    assert tbl.rows[1] == ["Planck Constant", "h", "6.626e-34", "J s"]


def test_format_table_to_markdown():
    handler = TableHandler()
    headers = ["State", "Energy (eV)"]
    rows = [["n=1", "-13.6"], ["n=2", "-3.4"]]

    md = handler.format_table_to_markdown(headers, rows)
    lines = md.split("\n")
    assert lines[0] == "| State | Energy (eV) |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| n=1 | -13.6 |"
    assert lines[3] == "| n=2 | -3.4 |"


def test_extract_pipe_tables_from_text():
    text = """
    Below is the comparison of thermodynamic cycles:

    | Cycle | Working Fluid | Efficiency |
    |---|---|---|
    | Carnot | Ideal Gas | 1 - Tc/Th |
    | Rankine | Water/Steam | ~40% |

    These cycles are fundamental in power engineering.
    """
    handler = TableHandler(doc_id="pipe_doc")
    tables, mod_text = handler.extract_pipe_tables_from_text(text)

    assert len(tables) == 1
    assert tables[0].headers == ["Cycle", "Working Fluid", "Efficiency"]
    assert len(tables[0].rows) == 2
    assert "Carnot" in mod_text
