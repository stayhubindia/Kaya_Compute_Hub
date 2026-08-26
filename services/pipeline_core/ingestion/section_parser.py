"""
Section and Hierarchy Parser (Phase 3.3).
Parses unsegmented or page-segmented text into a structured hierarchy of sections,
subsections, and paragraphs, recognizing academic and NPTEL course organizational patterns.
Also extracts citations/references into structured Reference models.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.ingestion.models import IngestionDocument, Reference, Section, Table


class SectionParser:
    """Detects and builds hierarchical sections from document text."""

    # Academic section patterns
    ACADEMIC_PATTERNS = [
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?abstract\b", "abstract"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?introduction\b", "introduction"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?(?:background|related\s+work)\b", "background"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?(?:methodology|methods|theory|theoretical\s+formulation|model)\b", "methodology"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?(?:experiments|experimental\s+setup|evaluation)\b", "experiments"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?(?:(?:experimental|simulation|empirical)\s+)?(?:results|findings)\b", "results"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?discussion\b", "discussion"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?(?:conclusion|conclusions|concluding\s+remarks)\b", "conclusion"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?references\b", "references"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?bibliography\b", "references"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?appendix\b", "appendix"),
    ]

    # NPTEL section patterns
    NPTEL_PATTERNS = [
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?(?:module\s+[0-9]+|module\s+[ivxlcdm]+)\s*[:\-–—]?\s*(.*)$", "module"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?(?:lecture\s+[0-9]+|lecture\s+[ivxlcdm]+)\s*[:\-–—]?\s*(.*)$", "lecture"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?(?:week\s+[0-9]+)\s*[:\-–—]?\s*(.*)$", "week"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?(?:lesson\s+[0-9]+|unit\s+[0-9]+)\s*[:\-–—]?\s*(.*)$", "lesson"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?(?:summary|recap|key\s+takeaways)\b", "summary"),
        (r"(?i)^(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?(?:assignment|quiz|practice\s+problems|tutorial)\b", "assignment"),
    ]

    # Generic numbered heading pattern: e.g. "1.2.3 Quantum States" or "1. Introduction"
    GENERIC_HEADING_PATTERN = r"^(?:(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z0-9\s,\-–—\(\)]{2,80}))$"

    # Citation/reference line pattern: e.g. "[1] Smith, J. ... (2020)" or "1. Landau and Lifshitz..."
    REFERENCE_LINE_PATTERN = re.compile(r"^(?:\[(\d+)\]|(\d+)\.)\s+(.+)$")

    def __init__(self, doc_id: str = "doc"):
        self.doc_id = doc_id

    def classify_heading_type(self, title: str) -> str:
        """Determines the section type (abstract, introduction, lecture, methodology, etc.)."""
        clean_title = title.strip()

        # Check NPTEL patterns
        for pat, sec_type in self.NPTEL_PATTERNS:
            if re.match(pat, clean_title):
                return sec_type

        # Check academic patterns
        for pat, sec_type in self.ACADEMIC_PATTERNS:
            if re.match(pat, clean_title):
                return sec_type

        # Check numbered heading
        if re.match(self.GENERIC_HEADING_PATTERN, clean_title):
            return "numbered_section"

        return "section"

    def is_likely_heading(self, line: str) -> bool:
        """Heuristic check to determine if a standalone line is a heading."""
        clean = line.strip()
        if not clean or len(clean) > 120 or len(clean) < 3:
            return False

        # Ends with period (except after heading number e.g. "1. Introduction")
        if clean.endswith(".") and not re.match(r"^\d+\.", clean):
            return False

        # Check all heading regex patterns
        for pat, _ in self.NPTEL_PATTERNS + self.ACADEMIC_PATTERNS:
            if re.match(pat, clean):
                return True

        if re.match(self.GENERIC_HEADING_PATTERN, clean):
            return True

        # All uppercase short title (e.g. "GOVERNING EQUATIONS OF FLUID MOTION")
        if clean.isupper() and len(clean.split()) <= 8 and len(clean) >= 4:
            return True

        return False

    def parse_text_into_sections(
        self,
        full_text: str,
        page_mappings: Optional[List[Tuple[int, str]]] = None,
    ) -> List[Section]:
        """
        Parses text into structured Section objects.
        If page_mappings is provided, associates sections with page start/end.
        """
        lines = full_text.split("\n")
        sections: List[Section] = []
        current_title = "Introduction"
        current_paragraphs: List[str] = []
        current_para_buffer: List[str] = []
        sec_idx = 0

        def flush_paragraph():
            if current_para_buffer:
                p_text = " ".join(current_para_buffer).strip()
                if p_text:
                    current_paragraphs.append(p_text)
                current_para_buffer.clear()

        def flush_section():
            nonlocal sec_idx
            flush_paragraph()
            if current_paragraphs:
                sec_type = self.classify_heading_type(current_title)
                sec = Section(
                    section_id=f"{self.doc_id[:12]}:sec_{sec_idx}",
                    title=current_title,
                    section_type=sec_type,
                    paragraphs=list(current_paragraphs),
                )
                sections.append(sec)
                sec_idx += 1
                current_paragraphs.clear()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush_paragraph()
                continue

            if self.is_likely_heading(stripped):
                flush_section()
                current_title = stripped
            else:
                current_para_buffer.append(stripped)

        flush_section()

        # Fallback if no sections were created
        if not sections and full_text.strip():
            sections.append(
                Section(
                    section_id=f"{self.doc_id[:12]}:sec_0",
                    title="Main Content",
                    section_type="main",
                    paragraphs=[p.strip() for p in full_text.split("\n\n") if p.strip()],
                )
            )

        return sections

    def parse_document_into_semantic_sections(
        self,
        raw_sections: List[Section],
    ) -> Tuple[List[Section], List[Reference]]:
        """
        Reorganizes raw page-based or fragmented sections into coherent semantic sections
        with appropriate hierarchy, page ranges, equations, and tables, extracting any references.
        """
        semantic_sections: List[Section] = []
        references: List[Reference] = []

        if not raw_sections:
            return semantic_sections, references

        current_title = raw_sections[0].title if raw_sections[0].title != "Page 1" else "Overview"
        current_paragraphs: List[str] = []
        current_tables: List[Table] = []
        current_page_start: Optional[int] = raw_sections[0].page_start or 1
        current_page_end: Optional[int] = raw_sections[0].page_end or 1
        sec_idx = 0
        in_references = False
        ref_counter = 0

        def flush_semantic_section():
            nonlocal sec_idx
            if current_paragraphs or current_tables:
                sec_type = self.classify_heading_type(current_title)
                sec = Section(
                    section_id=f"{self.doc_id[:12]}:sec_{sec_idx}",
                    title=current_title,
                    section_type=sec_type,
                    page_start=current_page_start,
                    page_end=current_page_end,
                    paragraphs=list(current_paragraphs),
                    tables=list(current_tables),
                )
                semantic_sections.append(sec)
                sec_idx += 1
                current_paragraphs.clear()
                current_tables.clear()

        for page_sec in raw_sections:
            p_num = page_sec.page_start or 1
            current_page_end = max(current_page_end or p_num, p_num)

            # Check if this page has tables
            if page_sec.tables:
                current_tables.extend(page_sec.tables)

            for p in page_sec.paragraphs:
                lines = [l.strip() for l in p.split("\n") if l.strip()]
                body_lines = []

                for line in lines:
                    # Check reference list item
                    if in_references:
                        ref_match = self.REFERENCE_LINE_PATTERN.match(line)
                        if ref_match:
                            ref_counter += 1
                            raw_ref = ref_match.group(3).strip()
                            references.append(
                                Reference(
                                    ref_id=f"{self.doc_id[:8]}:ref_{ref_counter}",
                                    raw_text=raw_ref,
                                )
                            )
                            continue

                    if self.is_likely_heading(line):
                        # Heading found!
                        if body_lines:
                            current_paragraphs.append(" ".join(body_lines).strip())
                            body_lines.clear()

                        flush_semantic_section()
                        current_title = line
                        current_page_start = p_num
                        current_page_end = p_num

                        if self.classify_heading_type(line) == "references":
                            in_references = True
                        else:
                            in_references = False
                    else:
                        body_lines.append(line)

                if body_lines:
                    para_text = " ".join(body_lines).strip()
                    if para_text:
                        current_paragraphs.append(para_text)

        flush_semantic_section()

        # If no semantic sections were detected (or fallback), keep original with updated types
        if not semantic_sections and raw_sections:
            return raw_sections, references

        return semantic_sections, references
