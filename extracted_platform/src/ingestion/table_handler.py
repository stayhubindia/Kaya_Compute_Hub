"""
Table Extraction and Formatting Handler (Phase 3.3).
Extracts, structures, and converts tabular data from documents into canonical Table models
and clean Markdown representations for downstream LLM training.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.ingestion.models import Table


class TableHandler:
    """Detects, parses, and formats structured tables."""

    def __init__(self, doc_id: str = "doc"):
        self.doc_id = doc_id
        self.total_tables = 0
        self.failed_tables = 0

    def parse_markdown_table(self, md_table_text: str, page_number: Optional[int] = None) -> Optional[Table]:
        """Parses a raw markdown table block into a structured Table object."""
        lines = [line.strip() for line in md_table_text.strip().split("\n") if line.strip()]
        if len(lines) < 2:
            return None

        # Headers from first line
        headers = [cell.strip() for cell in lines[0].strip("|").split("|")]

        # Determine delimiter line index
        row_start_idx = 1
        if len(lines) > 1 and re.match(r"^\|?[\s\-:]+(\|[\s\-:]+)+\|?$", lines[1]):
            row_start_idx = 2

        rows: List[List[str]] = []
        for line in lines[row_start_idx:]:
            row_cells = [cell.strip() for cell in line.strip("|").split("|")]
            # Pad or truncate row cells to match headers
            if len(row_cells) < len(headers):
                row_cells.extend([""] * (len(headers) - len(row_cells)))
            rows.append(row_cells[:len(headers)])

        self.total_tables += 1
        return Table(
            table_id=f"{self.doc_id[:8]}:tab_{self.total_tables}",
            headers=headers,
            rows=rows,
            markdown=md_table_text.strip(),
            page_number=page_number,
        )

    def format_table_to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        """Converts header and row arrays into a formatted GitHub Flavored Markdown table."""
        if not headers and not rows:
            return ""

        if not headers and rows:
            headers = [f"Col {i+1}" for i in range(len(rows[0]))]

        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for r in rows:
            padded_row = list(r)
            if len(padded_row) < len(headers):
                padded_row.extend([""] * (len(headers) - len(padded_row)))
            lines.append("| " + " | ".join(padded_row[:len(headers)]) + " |")

        return "\n".join(lines)

    def extract_pipe_tables_from_text(self, text: str, page_number: Optional[int] = None) -> Tuple[List[Table], str]:
        """
        Detects pipe-delimited tables embedded within raw text, extracts them into Table objects,
        and leaves normalized markdown in the text.
        """
        tables: List[Table] = []
        lines = text.split("\n")
        reconstructed_lines: List[str] = []
        in_table = False
        table_buffer: List[str] = []

        for line in lines:
            stripped = line.strip()
            is_pipe_line = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2

            if is_pipe_line:
                in_table = True
                table_buffer.append(stripped)
            else:
                if in_table:
                    # Flush table buffer
                    tbl_str = "\n".join(table_buffer)
                    tbl = self.parse_markdown_table(tbl_str, page_number=page_number)
                    if tbl:
                        tables.append(tbl)
                        reconstructed_lines.append(tbl.markdown or tbl_str)
                    else:
                        reconstructed_lines.extend(table_buffer)
                    table_buffer.clear()
                    in_table = False
                reconstructed_lines.append(line)

        if in_table and table_buffer:
            tbl_str = "\n".join(table_buffer)
            tbl = self.parse_markdown_table(tbl_str, page_number=page_number)
            if tbl:
                tables.append(tbl)
                reconstructed_lines.append(tbl.markdown or tbl_str)
            else:
                reconstructed_lines.extend(table_buffer)

        return tables, "\n".join(reconstructed_lines)
