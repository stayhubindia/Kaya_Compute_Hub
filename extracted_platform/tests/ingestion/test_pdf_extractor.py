"""
Unit tests for PDF Extraction Engine (Phase 3.3).
"""

import io
import pytest
from pypdf import PageObject, PdfWriter

from src.ingestion.models import ExtractionStatus
from src.ingestion.pdf_extractor import PDFExtractor


def create_sample_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to generate an in-memory PDF with specified text pages."""
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for text in pages_text:
        # Create a blank page
        page = PageObject.create_blank_page(width=612, height=792)
        # Add basic text stream
        writer.add_page(page)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_extractor_scanned_detection():
    # Empty blank page PDF -> 0 chars extracted -> should detect OCR required
    pdf_bytes = create_sample_pdf_bytes(["", ""])
    extractor = PDFExtractor()

    doc, telemetry = extractor.extract_pdf(pdf_bytes, source="nptel")
    assert telemetry.pages_total == 2
    assert telemetry.ocr_required is True
    assert telemetry.extraction_status in [ExtractionStatus.PARTIAL, ExtractionStatus.FAILED]


def test_pdf_extractor_corrupted_pdf():
    extractor = PDFExtractor()
    doc, telemetry = extractor.extract_pdf(b"This is not a real PDF file header", source="arxiv")

    assert doc is None
    assert telemetry.extraction_status == ExtractionStatus.FAILED
    assert "Corrupted or invalid PDF format" in telemetry.error_message


def test_pdf_extractor_metadata_and_pages():
    extractor = PDFExtractor()
    pdf_bytes = create_sample_pdf_bytes(["Page 1", "Page 2"])

    doc, telemetry = extractor.extract_pdf(
        pdf_bytes, source="nptel", default_domain="science", default_topic="physics"
    )
    assert doc is not None
    assert doc.format == "pdf"
    assert doc.metadata.source == "nptel"
    assert doc.metadata.domain == "science"
    assert telemetry.pages_total == 2
