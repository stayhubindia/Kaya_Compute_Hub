"""
PDF Document Extractor (Phase 3.3).
Extracts structured text, metadata, page boundaries, equations, and tables from PDF documents.
Detects image-only/scanned documents and flags ocr_required without data fabrication.
Prefers PyMuPDF (fitz) with automatic table recognition and rich metadata parsing.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

from src.ingestion.models import (
    ExtractionStatus,
    ExtractionTelemetry,
    IngestionDocument,
    IngestionDocumentMetadata,
    Section,
    Table,
)
from src.training.utils import compute_file_sha256

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extracts text, metadata, tables, and structural elements from PDF files."""

    def __init__(
        self,
        ocr_threshold_chars_per_page: int = 50,
        max_pages: int = 2000,
    ):
        self.ocr_threshold_chars_per_page = ocr_threshold_chars_per_page
        self.max_pages = max_pages

    def parse_metadata_from_source(self, file_path_or_name: str, first_page_text: str = "") -> Dict[str, Any]:
        """Infers title, authors, institution, and course from filename and first page content."""
        stem = Path(file_path_or_name).stem
        meta: Dict[str, Any] = {
            "title": None,
            "authors": [],
            "institution": None,
            "course": None,
        }

        # 1. Filename pattern: "NOC: Course Title, Institution.pdf"
        noc_match = re.match(r"^NOC:\s*(.*?),\s*(IIT\s+[A-Za-z]+|IISc.*|.*)$", stem, re.IGNORECASE)
        if noc_match:
            meta["title"] = noc_match.group(1).strip()
            meta["institution"] = noc_match.group(2).strip()
            meta["course"] = noc_match.group(1).strip()
            return meta

        # 2. Filename pattern: "Lecture_01_Intro.pdf" or "Module_2.pdf"
        mod_lec_match = re.match(r"^(?:Module|Lecture|mod|lec)[_\s\-]*(\d+)[_\s\-]*(.*)$", stem, re.IGNORECASE)
        if mod_lec_match:
            suffix = mod_lec_match.group(2).replace("_", " ").strip()
            if suffix:
                meta["title"] = f"Module {mod_lec_match.group(1)}: {suffix.title()}"
            else:
                meta["title"] = f"Module {mod_lec_match.group(1)}"

        # 3. Check first page text for title/headings if title is still null or generic
        if (not meta["title"] or meta["title"].startswith("Module ")) and first_page_text:
            lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
            for line in lines[:10]:
                # Look for "Module X: Title" or "Lecture Y: Title" or "Course: Title"
                m_title = re.match(r"^(?:Module\s+\d+|Lecture\s+\d+|Course)\s*[:\-–—]\s*(.+)$", line, re.IGNORECASE)
                if m_title:
                    meta["title"] = m_title.group(1).strip()
                    break

                # Look for IIT / Institution affiliation
                inst_match = re.search(r"\b(IIT\s+[A-Za-z]+|IISc\s+[A-Za-z]+|Indian\s+Institute\s+of\s+Technology\s+[A-Za-z]+)\b", line, re.IGNORECASE)
                if inst_match and not meta["institution"]:
                    meta["institution"] = inst_match.group(1).strip()

                # Look for Professor / Instructor author
                prof_match = re.match(r"^(?:Prof\.|Dr\.|Instructor)\s+([A-Za-z\.\s]+)$", line)
                if prof_match and not meta["authors"]:
                    meta["authors"].append(prof_match.group(1).strip())

        if not meta["title"]:
            meta["title"] = stem.replace("_", " ").replace("-", " ").title()

        return meta

    def extract_pdf(
        self,
        pdf_path_or_bytes: Union[str, Path, bytes],
        source: str = "unknown",
        source_url: Optional[str] = None,
        default_domain: str = "science",
        default_topic: str = "physics",
    ) -> Tuple[Optional[IngestionDocument], ExtractionTelemetry]:
        """
        Parses a PDF document into an IngestionDocument with full telemetry.
        """
        telemetry = ExtractionTelemetry()

        if isinstance(pdf_path_or_bytes, bytes):
            content_bytes = pdf_path_or_bytes
            source_path = "<bytes>"
            file_hash = IngestionDocument.compute_document_id(content_bytes)
        else:
            p = Path(pdf_path_or_bytes)
            source_path = str(p.resolve())
            try:
                content_bytes = p.read_bytes()
                file_hash = compute_file_sha256(p)
            except Exception as e:
                telemetry.extraction_status = ExtractionStatus.FAILED
                telemetry.error_message = f"Failed to read file bytes: {str(e)}"
                return None, telemetry

        # Preferred engine: PyMuPDF
        if fitz is not None:
            return self._extract_with_pymupdf(
                content_bytes=content_bytes,
                source_path=source_path,
                file_hash=file_hash,
                source=source,
                source_url=source_url,
                default_domain=default_domain,
                default_topic=default_topic,
                telemetry=telemetry,
            )

        # Fallback engine: pypdf
        if PdfReader is not None:
            return self._extract_with_pypdf(
                content_bytes=content_bytes,
                source_path=source_path,
                file_hash=file_hash,
                source=source,
                source_url=source_url,
                default_domain=default_domain,
                default_topic=default_topic,
                telemetry=telemetry,
            )

        telemetry.extraction_status = ExtractionStatus.FAILED
        telemetry.error_message = "Neither PyMuPDF nor pypdf is installed."
        return None, telemetry

    def _extract_with_pymupdf(
        self,
        content_bytes: bytes,
        source_path: str,
        file_hash: str,
        source: str,
        source_url: Optional[str],
        default_domain: str,
        default_topic: str,
        telemetry: ExtractionTelemetry,
    ) -> Tuple[Optional[IngestionDocument], ExtractionTelemetry]:
        """Extraction using PyMuPDF (fitz) with native table and structure extraction."""
        try:
            doc_fitz = fitz.open(stream=content_bytes, filetype="pdf")
        except Exception as e:
            telemetry.extraction_status = ExtractionStatus.FAILED
            telemetry.error_message = f"Corrupted or invalid PDF format: {str(e)}"
            return None, telemetry

        if doc_fitz.is_encrypted:
            try:
                if not doc_fitz.authenticate(""):
                    telemetry.extraction_status = ExtractionStatus.FAILED
                    telemetry.error_message = "PDF is password-encrypted."
                    return None, telemetry
            except Exception as e:
                telemetry.extraction_status = ExtractionStatus.FAILED
                telemetry.error_message = f"Failed to decrypt PDF: {str(e)}"
                return None, telemetry

        num_pages = len(doc_fitz)
        telemetry.pages_total = num_pages
        pages_to_process = min(num_pages, self.max_pages)

        # Native PDF metadata
        pdf_meta = doc_fitz.metadata or {}
        title = pdf_meta.get("title") or None
        authors = [pdf_meta.get("author")] if pdf_meta.get("author") else []

        page_texts: List[Tuple[int, str, List[Table]]] = []
        total_extracted_chars = 0
        total_tables_found = 0

        first_page_text = ""

        for page_idx in range(pages_to_process):
            try:
                page = doc_fitz[page_idx]
                p_num = page_idx + 1
                p_text = page.get_text("text") or ""
                p_text = p_text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
                clean_p_text = p_text.strip()
                if page_idx == 0:
                    first_page_text = clean_p_text

                # Native table detection
                page_tables: List[Table] = []
                try:
                    tabs = page.find_tables()
                    for t_idx, t in enumerate(tabs):
                        df_rows = t.extract()
                        if df_rows and len(df_rows) >= 2:
                            # Clean up cells
                            cleaned_rows = [
                                [str(cell or "").strip().replace("\n", " ") for cell in row]
                                for row in df_rows
                            ]
                            headers = cleaned_rows[0]
                            body_rows = cleaned_rows[1:]
                            # Format markdown
                            md_lines = [
                                "| " + " | ".join(headers) + " |",
                                "| " + " | ".join(["---"] * len(headers)) + " |",
                            ]
                            for r in body_rows:
                                padded = list(r) + [""] * max(0, len(headers) - len(r))
                                md_lines.append("| " + " | ".join(padded[:len(headers)]) + " |")
                            table_md = "\n".join(md_lines)

                            table_obj = Table(
                                table_id=f"{file_hash[:8]}:tab_{total_tables_found + 1}",
                                headers=headers,
                                rows=body_rows,
                                markdown=table_md,
                                page_number=p_num,
                            )
                            page_tables.append(table_obj)
                            total_tables_found += 1
                except Exception as tbl_err:
                    logger.debug(f"Table extraction notice on page {p_num}: {tbl_err}")

                page_texts.append((p_num, clean_p_text, page_tables))
                total_extracted_chars += len(clean_p_text)
                telemetry.pages_processed += 1
            except Exception as e:
                logger.warning(f"Error extracting page {page_idx + 1} from {source_path}: {e}")
                telemetry.pages_failed += 1
                page_texts.append((page_idx + 1, "", []))

        doc_fitz.close()

        telemetry.characters_extracted = total_extracted_chars
        telemetry.tables_detected = total_tables_found

        # Detect image-only / scanned PDF
        avg_chars_per_page = total_extracted_chars / max(1, telemetry.pages_processed)
        if total_extracted_chars < 50 or avg_chars_per_page < self.ocr_threshold_chars_per_page:
            telemetry.ocr_required = True
            telemetry.extraction_status = ExtractionStatus.PARTIAL if total_extracted_chars > 0 else ExtractionStatus.FAILED
            if total_extracted_chars == 0:
                telemetry.error_message = "Scanned/image-only PDF detected (0 characters extracted). OCR required."
        elif telemetry.pages_failed > 0:
            telemetry.extraction_status = ExtractionStatus.PARTIAL
        else:
            telemetry.extraction_status = ExtractionStatus.SUCCESS

        # Inferred metadata
        inferred = self.parse_metadata_from_source(source_path, first_page_text=first_page_text)
        if not title:
            title = inferred.get("title")
        if not authors and inferred.get("authors"):
            authors = inferred.get("authors", [])

        # Construct initial sections
        sections: List[Section] = []
        for p_num, text, tbls in page_texts:
            if not text and not tbls:
                continue
            sec = Section(
                section_id=f"{file_hash[:12]}:page_{p_num}",
                title=f"Page {p_num}",
                section_type="page",
                page_start=p_num,
                page_end=p_num,
                paragraphs=[p for p in text.split("\n\n") if p.strip()],
                tables=tbls,
            )
            sections.append(sec)

        extra_meta = {}
        if inferred.get("institution"):
            extra_meta["institution"] = inferred["institution"]
        if inferred.get("course"):
            extra_meta["course"] = inferred["course"]

        metadata = IngestionDocumentMetadata(
            title=title,
            authors=authors,
            source=source,
            source_url=source_url,
            domain=default_domain,
            topic=default_topic,
            extra=extra_meta,
        )

        doc = IngestionDocument(
            document_id=file_hash,
            source_path=source_path,
            source_file_hash=file_hash,
            format="pdf",
            metadata=metadata,
            sections=sections,
            telemetry=telemetry,
        )

        return doc, telemetry

    def _extract_with_pypdf(
        self,
        content_bytes: bytes,
        source_path: str,
        file_hash: str,
        source: str,
        source_url: Optional[str],
        default_domain: str,
        default_topic: str,
        telemetry: ExtractionTelemetry,
    ) -> Tuple[Optional[IngestionDocument], ExtractionTelemetry]:
        """Fallback extraction using pypdf."""
        try:
            reader = PdfReader(io.BytesIO(content_bytes))
        except Exception as e:
            telemetry.extraction_status = ExtractionStatus.FAILED
            telemetry.error_message = f"Corrupted PDF format: {str(e)}"
            return None, telemetry

        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
                if decrypted == 0:
                    telemetry.extraction_status = ExtractionStatus.FAILED
                    telemetry.error_message = "PDF is password-encrypted."
                    return None, telemetry
            except Exception as e:
                telemetry.extraction_status = ExtractionStatus.FAILED
                telemetry.error_message = f"Failed to decrypt PDF: {str(e)}"
                return None, telemetry

        num_pages = len(reader.pages)
        telemetry.pages_total = num_pages
        pages_to_process = min(num_pages, self.max_pages)

        pdf_meta = reader.metadata or {}
        title = getattr(pdf_meta, "title", None)
        authors = [str(pdf_meta.author)] if getattr(pdf_meta, "author", None) else []

        page_texts: List[Tuple[int, str]] = []
        total_extracted_chars = 0
        first_page_text = ""

        for page_idx in range(pages_to_process):
            try:
                page = reader.pages[page_idx]
                p_text = page.extract_text() or ""
                clean_p_text = p_text.strip()
                if page_idx == 0:
                    first_page_text = clean_p_text
                page_texts.append((page_idx + 1, clean_p_text))
                total_extracted_chars += len(clean_p_text)
                telemetry.pages_processed += 1
            except Exception as e:
                logger.warning(f"Error extracting page {page_idx + 1} from {source_path}: {e}")
                telemetry.pages_failed += 1
                page_texts.append((page_idx + 1, ""))

        telemetry.characters_extracted = total_extracted_chars

        avg_chars_per_page = total_extracted_chars / max(1, telemetry.pages_processed)
        if total_extracted_chars < 50 or avg_chars_per_page < self.ocr_threshold_chars_per_page:
            telemetry.ocr_required = True
            telemetry.extraction_status = ExtractionStatus.PARTIAL if total_extracted_chars > 0 else ExtractionStatus.FAILED
            if total_extracted_chars == 0:
                telemetry.error_message = "Scanned/image-only PDF detected (0 characters extracted). OCR required."
        elif telemetry.pages_failed > 0:
            telemetry.extraction_status = ExtractionStatus.PARTIAL
        else:
            telemetry.extraction_status = ExtractionStatus.SUCCESS

        inferred = self.parse_metadata_from_source(source_path, first_page_text=first_page_text)
        if not title:
            title = inferred.get("title")
        if not authors and inferred.get("authors"):
            authors = inferred.get("authors", [])

        sections: List[Section] = []
        for p_num, text in page_texts:
            if not text:
                continue
            sec = Section(
                section_id=f"{file_hash[:12]}:page_{p_num}",
                title=f"Page {p_num}",
                section_type="page",
                page_start=p_num,
                page_end=p_num,
                paragraphs=[p for p in text.split("\n\n") if p.strip()],
            )
            sections.append(sec)

        metadata = IngestionDocumentMetadata(
            title=title,
            authors=authors,
            source=source,
            source_url=source_url,
            domain=default_domain,
            topic=default_topic,
            extra=inferred,
        )

        doc = IngestionDocument(
            document_id=file_hash,
            source_path=source_path,
            source_file_hash=file_hash,
            format="pdf",
            metadata=metadata,
            sections=sections,
            telemetry=telemetry,
        )

        return doc, telemetry
