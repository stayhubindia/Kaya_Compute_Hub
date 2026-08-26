"""
Knowledge Ingestion Pipeline (Phase 3.3).
Orchestrates recursive discovery, multi-format extraction, deterministic normalization,
hierarchical section parsing, equation/table extraction, domain classification,
licensing audit, semantic chunking, quality validation, deduplication, and atomic persistence.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.dataset.schema import SourceType
from src.ingestion.checkpoint import IngestionCheckpointManager
from src.ingestion.chunker import SemanticChunker
from src.ingestion.deduplicator import IngestionDeduplicator
from src.ingestion.document_normalizer import DocumentNormalizer
from src.ingestion.equation_handler import EquationHandler
from src.ingestion.html_extractor import HTMLExtractor
from src.ingestion.license import LicenseHandler
from src.ingestion.manifest import IngestionManifestBuilder
from src.ingestion.metadata import MetadataClassifier
from src.ingestion.models import (
    ExtractionStatus,
    IngestionDocument,
    IngestionDocumentMetadata,
    KnowledgeChunk,
    Section,
)
from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.provenance import ProvenanceTracker
from src.ingestion.quality import IngestionQualityValidator
from src.ingestion.section_parser import SectionParser
from src.ingestion.statistics import IngestionStatistics
from src.ingestion.table_handler import TableHandler
from src.training.utils import compute_file_sha256

logger = logging.getLogger(__name__)


class KnowledgeIngestionPipeline:
    """Master pipeline orchestrating end-to-end document knowledge ingestion."""

    SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt", ".json", ".jsonl"}
    IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules", ".cache"}
    IGNORED_EXTS = {".pyc", ".tmp", ".part", ".lock", ".metadata", ".ids", ".log", ".sqlite3"}

    def __init__(
        self,
        output_dir: Union[str, Path],
        source: str = "unknown",
        resume: bool = True,
        force: bool = False,
        seed: int = 42,
        max_documents: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.source = source
        self.resume = resume
        self.force = force
        self.seed = seed
        self.max_documents = max_documents

        # Initialize Subsystem Handlers
        self.pdf_extractor = PDFExtractor()
        self.html_extractor = HTMLExtractor()
        self.normalizer = DocumentNormalizer()
        self.metadata_classifier = MetadataClassifier()
        self.license_handler = LicenseHandler()
        self.provenance_tracker = ProvenanceTracker()
        self.provenance_tracker.register_default_sources()
        self.quality_validator = IngestionQualityValidator()
        self.deduplicator = IngestionDeduplicator()
        self.chunker = SemanticChunker()
        self.checkpoint_manager = IngestionCheckpointManager(self.output_dir)

        if self.resume and not self.force:
            self.checkpoint_manager.load()

    def discover_files(self, input_path: Union[str, Path]) -> List[Path]:
        """Recursively discovers valid PDF, HTML, MD, TXT, and JSON files in target directory."""
        path = Path(input_path).resolve()
        if path.is_file():
            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                return [path]
            return []

        discovered: List[Path] = []
        for item in sorted(path.rglob("*")):
            if not item.is_file():
                continue
            if any(part in self.IGNORED_DIRS for part in item.parts):
                continue
            if item.suffix.lower() in self.IGNORED_EXTS:
                continue
            if item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                discovered.append(item)

        return sorted(discovered)

    def process_single_file(self, file_path: Path) -> Tuple[Optional[IngestionDocument], List[KnowledgeChunk]]:
        """Processes an individual source file into an IngestionDocument and KnowledgeChunks."""
        ext = file_path.suffix.lower()
        doc: Optional[IngestionDocument] = None

        if ext == ".pdf":
            doc, _ = self.pdf_extractor.extract_pdf(file_path, source=self.source)
        elif ext in [".html", ".htm"]:
            doc, _ = self.html_extractor.extract_html(file_path, source=self.source)
        elif ext in [".md", ".markdown", ".txt"]:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                file_hash = compute_file_sha256(file_path)
                doc_title = file_path.stem.replace("_", " ").replace("-", " ").title()
                lines = content.splitlines()
                for line in lines[:10]:
                    if line.startswith("# "):
                        doc_title = line[2:].strip()
                        break
                meta = IngestionDocumentMetadata(
                    title=doc_title,
                    source=self.source,
                    source_url=None,
                    extra={},
                )
                sections = []
                curr_title = doc_title
                curr_paras = []
                sec_idx = 0
                for line in lines:
                    if line.startswith("#"):
                        if curr_paras:
                            sec_idx += 1
                            sections.append(Section(
                                section_id=f"{file_hash[:12]}:sec_{sec_idx}",
                                title=curr_title,
                                section_type="heading",
                                paragraphs=curr_paras,
                            ))
                            curr_paras = []
                        curr_title = line.lstrip("#").strip()
                    else:
                        if line.strip():
                            curr_paras.append(line.strip())
                if curr_paras:
                    sec_idx += 1
                    sections.append(Section(
                        section_id=f"{file_hash[:12]}:sec_{sec_idx}",
                        title=curr_title,
                        section_type="heading",
                        paragraphs=curr_paras,
                    ))
                if not sections:
                    sections.append(Section(
                        section_id=f"{file_hash[:12]}:sec_0",
                        title=doc_title,
                        section_type="main",
                        paragraphs=[content],
                    ))
                doc = IngestionDocument(
                    document_id=file_hash,
                    source_path=str(file_path.resolve()),
                    source_file_hash=file_hash,
                    format=ext.strip("."),
                    metadata=meta,
                    sections=sections,
                )
            except Exception as e:
                logger.warning(f"Error parsing Markdown/Text file {file_path}: {e}")
                return None, []
        elif ext in [".json", ".jsonl"]:
            # Parse JSON/JSONL document if formatted as metadata or document record
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                data = json.loads(content) if ext == ".json" else [json.loads(line) for line in content.splitlines() if line.strip()][0]
                text_body = data.get("text") or data.get("abstract") or data.get("content") or json.dumps(data)
                file_hash = compute_file_sha256(file_path)
                meta = IngestionDocumentMetadata(
                    title=data.get("title") or file_path.stem,
                    authors=data.get("authors") if isinstance(data.get("authors"), list) else ([data["authors"]] if "authors" in data else []),
                    abstract=data.get("abstract"),
                    source=self.source,
                    source_url=data.get("url") or data.get("source_url"),
                    extra=data,
                )
                sec = Section(
                    section_id=f"{file_hash[:12]}:sec_0",
                    title="Content",
                    section_type="main",
                    paragraphs=[text_body],
                )
                doc = IngestionDocument(
                    document_id=file_hash,
                    source_path=str(file_path.resolve()),
                    source_file_hash=file_hash,
                    format=ext.strip("."),
                    metadata=meta,
                    sections=[sec],
                )
            except Exception as e:
                logger.warning(f"Error parsing JSON file {file_path}: {e}")
                return None, []
        else:
            return None, []

        if doc is None:
            return None, []

        # 1. Normalize full text and sections (PUA glyph mapping, dehyphenation, header stripping)
        for sec in doc.sections:
            norm_paras = [self.normalizer.normalize(p) for p in sec.paragraphs]
            sec.paragraphs = [p for p in norm_paras if p.strip()]

        # 2. Reconstruct semantic sections and extract references
        section_parser = SectionParser(doc.document_id)
        semantic_sections, references = section_parser.parse_document_into_semantic_sections(doc.sections)
        if semantic_sections:
            doc.sections = semantic_sections
        if references:
            doc.references.extend(references)

        # 3. Extract and bind equations and tables to semantic sections
        eq_handler = EquationHandler(doc.document_id)
        tbl_handler = TableHandler(doc.document_id)
        for sec in doc.sections:
            sec_equations = []
            clean_paras = []
            for p in sec.paragraphs:
                # Detect and parse pipe tables in paragraphs
                pipe_tables, p_after_tables = tbl_handler.extract_pipe_tables_from_text(p, page_number=sec.page_start)
                if pipe_tables:
                    sec.tables.extend(pipe_tables)

                # Detect and normalize equations
                eqs, final_p = eq_handler.extract_equations(p_after_tables, page_number=sec.page_start)
                if eqs:
                    sec_equations.extend(eqs)
                clean_paras.append(final_p)

            sec.paragraphs = [p for p in clean_paras if p.strip()]
            sec.equations.extend(sec_equations)

        # 4. Synchronize extraction telemetry with final normalized artifact
        doc.telemetry.characters_extracted = sum(len(p) for s in doc.sections for p in s.paragraphs)
        doc.telemetry.equations_detected = sum(len(s.equations) for s in doc.sections)
        doc.telemetry.tables_detected = sum(len(s.tables) for s in doc.sections)

        full_doc_text = doc.get_full_text()

        # 5. Domain & Topic Classification
        categories = doc.metadata.extra.get("categories") if isinstance(doc.metadata.extra.get("categories"), list) else None
        domain, topic, subtopic, conf = self.metadata_classifier.classify(
            full_doc_text, title=doc.metadata.title, categories=categories, source=self.source
        )
        doc.metadata.domain = domain
        doc.metadata.topic = topic
        doc.metadata.subtopic = subtopic
        doc.metadata.classification_confidence = conf

        # 6. Licensing Audit
        lic_result = self.license_handler.evaluate_license(
            full_doc_text, declared_license=doc.metadata.license, source=self.source
        )
        doc.metadata.license = lic_result.license_name
        doc.metadata.license_status = lic_result.license_status
        doc.metadata.license_url = lic_result.license_url
        doc.metadata.license_evidence = lic_result.license_evidence
        doc.metadata.internal_only = lic_result.internal_only

        # 7. Semantic Chunking
        raw_chunks = self.chunker.chunk_document(doc)

        # 8. Quality Audit on Document and Chunks
        doc_quality = self.quality_validator.audit_document(doc)
        for ch in raw_chunks:
            self.quality_validator.audit_chunk(ch)

        return doc, raw_chunks

    def run(self, input_path: Union[str, Path]) -> IngestionStatistics:
        """Executes full knowledge ingestion on target input directory or file."""
        start_time = time.time()
        discovered_files = self.discover_files(input_path)
        total_files = len(discovered_files)
        stats = IngestionStatistics(documents_discovered=total_files)

        logger.info("Discovered %d candidate files. Starting ingestion...", total_files)
        logger.info("-" * 60)

        if self.max_documents:
            discovered_files = discovered_files[: self.max_documents]
            logger.info("Limited to first %d documents by --max-documents.", self.max_documents)

        docs_output_path = self.output_dir / "documents.jsonl"
        sections_output_path = self.output_dir / "sections.jsonl"
        chunks_output_path = self.output_dir / "chunks.jsonl"

        source_file_hashes: Dict[str, str] = {}
        all_unique_chunks: List[KnowledgeChunk] = []
        processed_count = 0
        skipped_count = 0

        file_mode = "a" if (self.resume and not self.force and docs_output_path.exists()) else "w"
        with open(docs_output_path, file_mode, encoding="utf-8", errors="replace") as f_docs, \
             open(sections_output_path, file_mode, encoding="utf-8", errors="replace") as f_secs, \
             open(chunks_output_path, file_mode, encoding="utf-8", errors="replace") as f_chunks:

            for idx, file_path in enumerate(discovered_files):
                doc_num = idx + 1
                file_size_kb = file_path.stat().st_size / 1024
                logger.info(
                    "[%d/%d] Processing: %-40s  (%.1f KB)",
                    doc_num, total_files, file_path.name[:40], file_size_kb,
                )

                t0 = time.time()
                file_hash = compute_file_sha256(file_path)
                source_file_hashes[str(file_path.name)] = file_hash

                # Check deduplication & checkpointing
                if not self.force:
                    if self.checkpoint_manager.is_completed(file_hash):
                        logger.info("  [SKIP] Already completed in previous run — skipping.")
                        skipped_count += 1
                        continue
                    if self.deduplicator.is_duplicate_document(file_hash):
                        logger.info("  [SKIP] Exact duplicate document — skipping.")
                        stats.documents_duplicate += 1
                        skipped_count += 1
                        continue

                self.checkpoint_manager.set_document_state(file_hash, "PROCESSING")

                # Extract
                logger.info("  [1/4] Extracting text, sections, equations...")
                t_extract = time.time()
                doc, chunks = self.process_single_file(file_path)
                if doc is None:
                    logger.warning("  [FAIL] Extraction failed — skipping file.")
                    self.checkpoint_manager.set_document_state(file_hash, "FAILED")
                    stats.documents_failed += 1
                    continue
                logger.info(
                    "  [1/4] Extracted %d sections, %d raw chunks in %.2fs",
                    len(doc.sections), len(chunks), time.time() - t_extract,
                )

                # Deduplication
                logger.info("  [2/4] Deduplicating %d chunks (MinHash LSH)...", len(chunks))
                t_dedup = time.time()
                unique_chunks, dedup_rep = self.deduplicator.deduplicate_chunks(chunks)
                logger.info(
                    "  [2/4] Dedup done in %.2fs: %d unique / %d exact-dup / %d near-dup",
                    time.time() - t_dedup,
                    dedup_rep.unique_chunks,
                    dedup_rep.exact_duplicate_chunks,
                    dedup_rep.near_duplicate_chunks,
                )
                all_unique_chunks.extend(unique_chunks)

                # Persist
                logger.info("  [3/4] Writing %d unique chunks to disk...", len(unique_chunks))
                f_docs.write(doc.to_json() + "\n")
                for s in doc.sections:
                    sec_json = json.dumps(s.to_dict(), ensure_ascii=False).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
                    f_secs.write(sec_json + "\n")
                for c in unique_chunks:
                    f_chunks.write(c.to_json() + "\n")

                # Checkpoint & stats
                stats.record_document(doc)
                status = "COMPLETED" if doc.telemetry.extraction_status == ExtractionStatus.SUCCESS else "PARTIAL"
                self.checkpoint_manager.set_document_state(file_hash, status)
                processed_count += 1

                elapsed_total = time.time() - start_time
                avg_per_doc = elapsed_total / max(1, processed_count)
                remaining = total_files - skipped_count - processed_count - stats.documents_failed
                eta_sec = avg_per_doc * remaining
                logger.info(
                    "  [4/4] Done [%s] in %.2fs | Cumulative: %d docs, %d chunks | ETA: %.0fs",
                    status, time.time() - t0,
                    processed_count, len(all_unique_chunks), eta_sec,
                )
                logger.info("  %s", "-" * 56)

                if doc_num % 10 == 0:
                    self.checkpoint_manager.save_atomic()
                    logger.info("  [CHECKPOINT] Saved at document %d.", doc_num)

        self.checkpoint_manager.save_atomic()
        stats.record_chunks(all_unique_chunks)
        stats.processing_duration_seconds = time.time() - start_time

        # Compute output file hashes
        output_file_hashes = {
            "documents.jsonl": compute_file_sha256(docs_output_path) if docs_output_path.exists() else "",
            "sections.jsonl": compute_file_sha256(sections_output_path) if sections_output_path.exists() else "",
            "chunks.jsonl": compute_file_sha256(chunks_output_path) if chunks_output_path.exists() else "",
        }

        # Build Manifest
        manifest_builder = IngestionManifestBuilder(
            execution_id=self.checkpoint_manager.execution_id,
            source=self.source,
            input_directory=str(Path(input_path).resolve()),
            seed=self.seed,
        )
        manifest_data = manifest_builder.build_manifest(
            doc_count=stats.documents_discovered,
            successful_count=stats.documents_successful,
            partial_count=stats.documents_partial,
            failed_count=stats.documents_failed,
            duplicate_count=stats.documents_duplicate,
            total_pages=stats.total_pages,
            total_characters=stats.total_characters,
            total_sections=stats.total_sections,
            total_chunks=stats.total_chunks,
            total_equations=stats.total_equations,
            total_tables=stats.total_tables,
            domain_dist=dict(stats.domain_distribution),
            license_dist=dict(stats.license_distribution),
            quality_dist=dict(stats.quality_distribution),
            source_file_hashes=source_file_hashes,
            output_file_hashes=output_file_hashes,
        )
        manifest_builder.write_manifest(manifest_data, self.output_dir / "manifest.json")

        # Write Reports
        report_json_path = self.reports_dir / "ingestion_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)

        report_md_path = self.reports_dir / "ingestion_report.md"
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(stats.generate_markdown_report(title=f"Ingestion Report: {self.source.upper()}"))

        return stats
