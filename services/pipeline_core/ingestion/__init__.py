"""
Knowledge Ingestion Engine Package (Phase 3.3).
Provides production-quality document discovery, extraction, normalization, section parsing,
equation/table handling, chunking, license tracking, quality auditing, deduplication, and persistence.
"""

from src.ingestion.checkpoint import IngestionCheckpointManager
from src.ingestion.chunker import SemanticChunker
from src.ingestion.deduplicator import IngestionDeduplicator, IngestionDeduplicationReport
from src.ingestion.document_normalizer import DocumentNormalizer
from src.ingestion.equation_handler import EquationHandler
from src.ingestion.html_extractor import HTMLExtractor
from src.ingestion.license import LicenseEvaluationResult, LicenseHandler
from src.ingestion.manifest import IngestionManifestBuilder
from src.ingestion.metadata import MetadataClassifier
from src.ingestion.models import (
    Equation,
    ExtractionStatus,
    ExtractionTelemetry,
    Figure,
    IngestionDocument,
    IngestionDocumentMetadata,
    KnowledgeChunk,
    LicenseStatus,
    QualityStatus,
    Reference,
    Section,
    Table,
)
from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.pipeline import KnowledgeIngestionPipeline
from src.ingestion.provenance import ProvenanceTracker
from src.ingestion.quality import IngestionQualityValidator, QualityAuditResult
from src.ingestion.section_parser import SectionParser
from src.ingestion.statistics import IngestionStatistics
from src.ingestion.table_handler import TableHandler

__all__ = [
    "Equation",
    "EquationHandler",
    "ExtractionStatus",
    "ExtractionTelemetry",
    "Figure",
    "HTMLExtractor",
    "IngestionCheckpointManager",
    "IngestionDeduplicationReport",
    "IngestionDeduplicator",
    "IngestionDocument",
    "IngestionDocumentMetadata",
    "IngestionManifestBuilder",
    "IngestionQualityValidator",
    "IngestionStatistics",
    "KnowledgeChunk",
    "KnowledgeIngestionPipeline",
    "LicenseEvaluationResult",
    "LicenseHandler",
    "LicenseStatus",
    "MetadataClassifier",
    "PDFExtractor",
    "ProvenanceTracker",
    "QualityAuditResult",
    "QualityStatus",
    "Reference",
    "Section",
    "SectionParser",
    "Table",
    "TableHandler",
    "DocumentNormalizer",
    "SemanticChunker",
]
