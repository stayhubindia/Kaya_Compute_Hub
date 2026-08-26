"""
Canonical Intermediate Representation (IR) and Data Models for Knowledge Ingestion (Phase 3.3).
Defines strongly typed schemas for documents, sections, equations, tables, chunks, and metadata.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

from src.dataset.schema import ProvenanceInfo, SourceType


class ExtractionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class LicenseStatus(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    MISSING = "MISSING"


class QualityStatus(str, Enum):
    PASSED = "PASSED"
    WARNING = "WARNING"
    REJECTED = "REJECTED"


class ExtractionTelemetry(BaseModel):
    """Telemetry capturing physical and structural document extraction metrics."""
    pages_total: int = 0
    pages_processed: int = 0
    pages_failed: int = 0
    characters_extracted: int = 0
    tables_detected: int = 0
    equations_detected: int = 0
    ocr_required: bool = False
    extraction_status: ExtractionStatus = ExtractionStatus.SUCCESS
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages_total": self.pages_total,
            "pages_processed": self.pages_processed,
            "pages_failed": self.pages_failed,
            "characters_extracted": self.characters_extracted,
            "tables_detected": self.tables_detected,
            "equations_detected": self.equations_detected,
            "ocr_required": self.ocr_required,
            "extraction_status": self.extraction_status.value,
            "error_message": self.error_message,
        }


class Equation(BaseModel):
    """Mathematical equation extracted with formatting preservation."""
    equation_id: str
    latex_content: str
    mathml: Optional[str] = None
    raw_text: Optional[str] = None
    equation_type: str = "display"  # "inline" or "display"
    page_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equation_id": self.equation_id,
            "latex_content": self.latex_content,
            "mathml": self.mathml,
            "raw_text": self.raw_text,
            "equation_type": self.equation_type,
            "page_number": self.page_number,
        }


class Table(BaseModel):
    """Structured tabular data extracted with markdown formatting."""
    table_id: str
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    markdown: Optional[str] = None
    caption: Optional[str] = None
    page_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_id": self.table_id,
            "headers": self.headers,
            "rows": self.rows,
            "markdown": self.markdown,
            "caption": self.caption,
            "page_number": self.page_number,
        }


class Figure(BaseModel):
    """Extracted figure or graphic reference with caption."""
    figure_id: str
    caption: Optional[str] = None
    page_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "caption": self.caption,
            "page_number": self.page_number,
        }


class Reference(BaseModel):
    """Bibliographic citation or reference item."""
    ref_id: str
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    raw_text: str = ""
    doi: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "title": self.title,
            "authors": self.authors,
            "raw_text": self.raw_text,
            "doi": self.doi,
            "url": self.url,
        }


class Section(BaseModel):
    """Hierarchical document section or lecture module."""
    section_id: str
    title: str = "Untitled Section"
    section_type: str = "section"  # "abstract", "introduction", "lecture", "module", "results", etc.
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    subsections: List[Section] = Field(default_factory=list)
    paragraphs: List[str] = Field(default_factory=list)
    equations: List[Equation] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    figures: List[Figure] = Field(default_factory=list)

    def full_text(self) -> str:
        """Assembles the section's text including paragraphs and embedded tables/equations."""
        parts = []
        if self.title and self.title != "Untitled Section":
            parts.append(f"## {self.title}\n")
        parts.extend(self.paragraphs)
        for t in self.tables:
            if t.markdown:
                parts.append(t.markdown)
        for eq in self.equations:
            if eq.latex_content:
                parts.append(f"\n$${eq.latex_content}$$\n")
        for sub in self.subsections:
            parts.append(sub.full_text())
        return "\n\n".join(filter(None, parts))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "section_type": self.section_type,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "subsections": [sub.to_dict() for sub in self.subsections],
            "paragraphs": self.paragraphs,
            "equations": [eq.to_dict() for eq in self.equations],
            "tables": [t.to_dict() for t in self.tables],
            "figures": [f.to_dict() for f in self.figures],
        }


class IngestionDocumentMetadata(BaseModel):
    """Comprehensive document-level metadata and rights classification."""
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    source: str = "unknown"  # "nptel", "arxiv", "mit_ocw", etc.
    source_type: str = SourceType.DOCUMENTATION.value
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    canonical_url: Optional[str] = None
    license: Optional[str] = None
    license_status: LicenseStatus = LicenseStatus.UNKNOWN
    license_url: Optional[str] = None
    license_evidence: Optional[str] = None
    internal_only: bool = True
    domain: str = "science"
    topic: str = "physics"
    subtopic: Optional[str] = None
    classification_confidence: float = 1.0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    retrieved_at: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "source": self.source,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "canonical_url": self.canonical_url,
            "license": self.license,
            "license_status": self.license_status.value,
            "license_url": self.license_url,
            "license_evidence": self.license_evidence,
            "internal_only": self.internal_only,
            "domain": self.domain,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "classification_confidence": self.classification_confidence,
            "created_at": self.created_at,
            "retrieved_at": self.retrieved_at,
            "extra": self.extra,
        }


class IngestionDocument(BaseModel):
    """Canonical Intermediate Representation (IR) of a fully parsed document."""
    document_id: str
    source_path: str
    source_file_hash: str
    format: str  # "pdf", "html", "json", "mixed"
    metadata: IngestionDocumentMetadata
    sections: List[Section] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)
    telemetry: ExtractionTelemetry = Field(default_factory=ExtractionTelemetry)

    @classmethod
    def compute_document_id(cls, content_bytes: bytes) -> str:
        """Deterministic content identity via SHA-256."""
        return hashlib.sha256(content_bytes).hexdigest()

    def get_full_text(self) -> str:
        """Concatenates all section texts in document order."""
        return "\n\n".join([sec.full_text() for sec in self.sections])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "source_file_hash": self.source_file_hash,
            "format": self.format,
            "metadata": self.metadata.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "references": [r.to_dict() for r in self.references],
            "telemetry": self.telemetry.to_dict(),
        }

    def to_json(self) -> str:
        raw = json.dumps(self.to_dict(), ensure_ascii=False)
        return raw.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IngestionDocument:
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> IngestionDocument:
        return cls.from_dict(json.loads(json_str))


class KnowledgeChunk(BaseModel):
    """Atomic, semantically chunked knowledge segment ready for downstream processing."""
    chunk_id: str
    document_id: str
    section_id: str
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    token_estimate: int = 0
    domain: str
    topic: str
    subtopic: Optional[str] = None
    source: str
    source_type: str
    source_url: Optional[str] = None
    license: Optional[str] = None
    license_status: str = LicenseStatus.UNKNOWN.value
    internal_only: bool = True
    quality_score: float = 1.0
    provenance: ProvenanceInfo

    @classmethod
    def generate_chunk_id(cls, doc_id: str, section_id: str, chunk_idx: int, text: str) -> str:
        raw = f"{doc_id}:{section_id}:{chunk_idx}:{text.strip()}"
        clean_bytes = raw.encode("utf-8", errors="replace")
        return hashlib.sha256(clean_bytes).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "section_id": self.section_id,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "token_estimate": self.token_estimate,
            "domain": self.domain,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "source": self.source,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "license": self.license,
            "license_status": self.license_status,
            "internal_only": self.internal_only,
            "quality_score": self.quality_score,
            "provenance": self.provenance.to_dict(),
        }

    def to_json(self) -> str:
        raw = json.dumps(self.to_dict(), ensure_ascii=False)
        return raw.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeChunk:
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> KnowledgeChunk:
        return cls.from_dict(json.loads(json_str))
