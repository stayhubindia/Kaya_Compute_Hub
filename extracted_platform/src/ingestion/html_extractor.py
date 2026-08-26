"""
HTML Document Extractor (Phase 3.3).
Extracts structured sections, metadata, mathematical formulas (MathML/KaTeX/MathJax),
and tables from HTML documents while cleanly stripping navigation, scripts, and boilerplate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from bs4 import BeautifulSoup, Comment, Tag
except ImportError:
    BeautifulSoup = None

from src.ingestion.models import (
    Equation,
    ExtractionStatus,
    ExtractionTelemetry,
    IngestionDocument,
    IngestionDocumentMetadata,
    Section,
    Table,
)
from src.training.utils import compute_file_sha256

logger = logging.getLogger(__name__)


class HTMLExtractor:
    """Extracts clean scientific text, headings, formulas, and tables from HTML documents."""

    def __init__(
        self,
        parser: str = "lxml",
        strip_tags: Optional[List[str]] = None,
    ):
        self.parser = parser
        self.strip_tags = strip_tags or [
            "nav", "header", "footer", "aside", "script", "style",
            "noscript", "iframe", "form", "button", "svg", "menu"
        ]

    def _extract_meta_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extracts standard and academic metadata tags from HTML head."""
        meta: Dict[str, Any] = {
            "title": None,
            "authors": [],
            "canonical_url": None,
            "description": None,
            "extra": {},
        }

        # Title
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            meta["title"] = title_tag.string.strip()

        # Headings / OpenGraph / Dublin Core / Highwire Press academic tags
        for m in soup.find_all("meta"):
            name = (m.get("name") or m.get("property") or "").lower()
            content = m.get("content") or ""
            if not content:
                continue

            if name in ["citation_title", "og:title", "dc.title", "twitter:title"] and not meta["title"]:
                meta["title"] = content.strip()
            elif name in ["citation_author", "author", "dc.creator"]:
                meta["authors"].append(content.strip())
            elif name in ["citation_abstract", "description", "og:description"]:
                meta["description"] = content.strip()
            elif name in ["citation_pdf_url", "citation_abstract_html_url"]:
                meta["extra"][name] = content.strip()

        # Canonical link
        canon = soup.find("link", rel="canonical")
        if canon and canon.get("href"):
            meta["canonical_url"] = canon["href"].strip()

        return meta

    def _convert_math_tags(self, soup: BeautifulSoup) -> None:
        """Converts MathML, KaTeX, and MathJax markup into inline/display LaTeX representations."""
        # 1. MathML
        for math_tag in soup.find_all("math"):
            alt_text = math_tag.get("alttext") or ""
            annotation = math_tag.find("annotation", attrs={"encoding": ["application/x-tex", "TeX"]})
            if annotation and annotation.string:
                math_tag.replace_with(f" ${annotation.string.strip()}$ ")
            elif alt_text:
                math_tag.replace_with(f" ${alt_text.strip()}$ ")
            else:
                # MathML text fallback
                math_tag.replace_with(f" ${math_tag.get_text().strip()}$ ")

        # 2. KaTeX / MathJax spans
        for span in soup.find_all("span", class_=lambda c: c and any(k in str(c) for k in ["katex", "math", "MathJax"])):
            tex_ann = span.find(attrs={"aria-label": True}) or span.find("annotation")
            if tex_ann and tex_ann.get("aria-label"):
                span.replace_with(f" ${tex_ann['aria-label']}$ ")
            elif tex_ann and tex_ann.string:
                span.replace_with(f" ${tex_ann.string}$ ")

    def _extract_table(self, table_tag: Tag, table_idx: int) -> Table:
        """Parses an HTML <table> element into a structured Table model."""
        headers: List[str] = []
        rows: List[List[str]] = []
        caption = None

        cap_tag = table_tag.find("caption")
        if cap_tag:
            caption = cap_tag.get_text().strip()

        # Extract headers
        for th in table_tag.find_all("th"):
            headers.append(th.get_text().strip())

        # Extract rows
        for tr in table_tag.find_all("tr"):
            row_cells = []
            for td in tr.find_all("td"):
                row_cells.append(td.get_text().strip())
            if row_cells:
                rows.append(row_cells)

        # Build markdown table representation
        md_lines = []
        if headers:
            md_lines.append("| " + " | ".join(headers) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            md_lines.append("| " + " | ".join(r) + " |")
        markdown_str = "\n".join(md_lines) if md_lines else None

        return Table(
            table_id=f"tab_{table_idx}",
            headers=headers,
            rows=rows,
            markdown=markdown_str,
            caption=caption,
        )

    def extract_html(
        self,
        html_path_or_str: Union[str, Path, bytes],
        source: str = "unknown",
        source_url: Optional[str] = None,
        default_domain: str = "science",
        default_topic: str = "physics",
    ) -> Tuple[Optional[IngestionDocument], ExtractionTelemetry]:
        """
        Parses an HTML document into an IngestionDocument with full telemetry.
        """
        telemetry = ExtractionTelemetry()

        if isinstance(html_path_or_str, bytes):
            html_bytes = html_path_or_str
            source_path = "<bytes>"
            file_hash = IngestionDocument.compute_document_id(html_bytes)
            html_content = html_bytes.decode("utf-8", errors="replace")
        elif isinstance(html_path_or_str, Path) or (isinstance(html_path_or_str, str) and Path(html_path_or_str).is_file()):
            p = Path(html_path_or_str)
            source_path = str(p.resolve())
            try:
                html_bytes = p.read_bytes()
                file_hash = compute_file_sha256(p)
                html_content = html_bytes.decode("utf-8", errors="replace")
            except Exception as e:
                telemetry.extraction_status = ExtractionStatus.FAILED
                telemetry.error_message = f"Failed to read HTML file: {str(e)}"
                return None, telemetry
        else:
            html_content = str(html_path_or_str)
            source_path = "<string>"
            file_hash = IngestionDocument.compute_document_id(html_content.encode("utf-8", errors="replace"))

        if BeautifulSoup is None:
            telemetry.extraction_status = ExtractionStatus.FAILED
            telemetry.error_message = "beautifulsoup4 library is not installed."
            return None, telemetry

        try:
            soup = BeautifulSoup(html_content, self.parser)
        except Exception:
            # Fallback to python html.parser if lxml fails
            try:
                soup = BeautifulSoup(html_content, "html.parser")
            except Exception as e:
                telemetry.extraction_status = ExtractionStatus.FAILED
                telemetry.error_message = f"HTML parse failed: {str(e)}"
                return None, telemetry

        # Extract metadata before stripping tags
        meta_dict = self._extract_meta_data(soup)

        # Remove HTML comments
        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()

        # Remove stripped tag elements (nav, header, footer, script, style, etc.)
        for tag_name in self.strip_tags:
            for el in soup.find_all(tag_name):
                el.decompose()

        # Convert math tags
        self._convert_math_tags(soup)

        # Parse sections based on heading hierarchy (h1, h2, h3) and content elements
        sections: List[Section] = []
        current_section: Optional[Section] = None
        sec_counter = 0

        # Scan through body content elements in document order
        body = soup.find("body") or soup
        for elem in body.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "pre", "blockquote", "table"]):
            tag_name = elem.name.lower()

            if tag_name in ["h1", "h2", "h3", "h4"]:
                heading_text = elem.get_text().strip()
                if not heading_text:
                    continue
                # Save previous section
                if current_section and (current_section.paragraphs or current_section.tables):
                    sections.append(current_section)
                sec_counter += 1
                current_section = Section(
                    section_id=f"{file_hash[:12]}:sec_{sec_counter}",
                    title=heading_text,
                    section_type="heading",
                    paragraphs=[],
                )
            elif tag_name == "table":
                tbl = self._extract_table(elem, telemetry.tables_detected + 1)
                telemetry.tables_detected += 1
                if current_section is None:
                    sec_counter += 1
                    current_section = Section(
                        section_id=f"{file_hash[:12]}:sec_{sec_counter}",
                        title=meta_dict.get("title") or "Content",
                        section_type="main",
                        paragraphs=[],
                    )
                current_section.tables.append(tbl)
            else:
                text = elem.get_text().strip()
                if not text:
                    continue
                if current_section is None:
                    sec_counter += 1
                    current_section = Section(
                        section_id=f"{file_hash[:12]}:sec_{sec_counter}",
                        title=meta_dict.get("title") or "Introduction",
                        section_type="main",
                        paragraphs=[],
                    )
                current_section.paragraphs.append(text)

        if current_section and (current_section.paragraphs or current_section.tables):
            sections.append(current_section)

        # Compute total character telemetry
        total_chars = sum(len(s.full_text()) for s in sections)
        telemetry.characters_extracted = total_chars
        telemetry.pages_total = 1
        telemetry.pages_processed = 1

        if total_chars == 0:
            telemetry.extraction_status = ExtractionStatus.FAILED
            telemetry.error_message = "No text content could be extracted from HTML."
        else:
            telemetry.extraction_status = ExtractionStatus.SUCCESS

        doc_meta = IngestionDocumentMetadata(
            title=meta_dict.get("title"),
            authors=meta_dict.get("authors", []),
            abstract=meta_dict.get("description"),
            source=source,
            source_url=source_url or meta_dict.get("canonical_url"),
            canonical_url=meta_dict.get("canonical_url"),
            domain=default_domain,
            topic=default_topic,
            extra=meta_dict.get("extra", {}),
        )

        doc = IngestionDocument(
            document_id=file_hash,
            source_path=source_path,
            source_file_hash=file_hash,
            format="html",
            metadata=doc_meta,
            sections=sections,
            telemetry=telemetry,
        )

        return doc, telemetry
