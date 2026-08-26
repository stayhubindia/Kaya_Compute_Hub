"""
Semantic Document Chunker (Phase 3.3).
Partitions structured IngestionDocuments into cohesive, self-contained KnowledgeChunks.
Preserves mathematical formulas, tables, and section hierarchies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.dataset.schema import ProvenanceInfo
from src.ingestion.models import (
    IngestionDocument,
    KnowledgeChunk,
    LicenseStatus,
    Section,
)

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Chunks documents into structured knowledge pieces conforming to token limits."""

    def __init__(
        self,
        min_chunk_tokens: int = 100,
        max_chunk_tokens: int = 1024,
        chunk_overlap_tokens: int = 50,
        token_estimation_ratio: float = 0.75,  # ~4 characters per token
    ):
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.token_estimation_ratio = token_estimation_ratio

    def estimate_tokens(self, text: str) -> int:
        """Estimates token count from word and character heuristics."""
        if not text:
            return 0
        words = len(text.split())
        chars = len(text)
        return max(words, int(chars * 0.25))

    def _chunk_section(
        self,
        doc: IngestionDocument,
        sec: Section,
        chunk_list: List[KnowledgeChunk],
    ) -> None:
        """Recursively chunks a section and its subsections."""
        meta = doc.metadata
        paragraphs = list(sec.paragraphs)

        # Include markdown tables as coherent paragraph blocks
        for t in sec.tables:
            if t.markdown:
                paragraphs.append(t.markdown)

        # Include display equations as coherent blocks
        for eq in sec.equations:
            if eq.latex_content and eq.equation_type == "display":
                paragraphs.append(f"\n$${eq.latex_content}$$\n")

        if not paragraphs and not sec.subsections:
            return

        current_paras: List[str] = []
        current_token_count = 0
        sec_chunk_idx = len(chunk_list)

        def create_chunk(text: str) -> KnowledgeChunk:
            tok_est = self.estimate_tokens(text)
            chunk_id = KnowledgeChunk.generate_chunk_id(
                doc.document_id, sec.section_id, len(chunk_list), text
            )

            prov = ProvenanceInfo(
                source_type=meta.source_type,
                source=meta.source,
                source_id=doc.document_id,
                license=meta.license,
                source_url=meta.source_url or meta.canonical_url,
            )

            return KnowledgeChunk(
                chunk_id=chunk_id,
                document_id=doc.document_id,
                section_id=sec.section_id,
                text=text,
                page_start=sec.page_start,
                page_end=sec.page_end,
                token_estimate=tok_est,
                domain=meta.domain,
                topic=meta.topic,
                subtopic=meta.subtopic,
                source=meta.source,
                source_type=meta.source_type,
                source_url=meta.source_url or meta.canonical_url,
                license=meta.license,
                license_status=meta.license_status.value if isinstance(meta.license_status, LicenseStatus) else str(meta.license_status),
                internal_only=meta.internal_only,
                quality_score=1.0,
                provenance=prov,
            )

        for para in paragraphs:
            p_clean = para.strip()
            if not p_clean:
                continue

            p_tokens = self.estimate_tokens(p_clean)

            # If adding this paragraph exceeds max_chunk_tokens, flush current chunk
            if current_paras and (current_token_count + p_tokens > self.max_chunk_tokens):
                chunk_text = "\n\n".join(current_paras)
                if current_token_count >= self.min_chunk_tokens or not chunk_list:
                    chunk_list.append(create_chunk(chunk_text))
                    current_paras = []
                    current_token_count = 0
                else:
                    # Current chunk is too small, but adding paragraph exceeds max; flush anyway
                    chunk_list.append(create_chunk(chunk_text))
                    current_paras = []
                    current_token_count = 0

            current_paras.append(p_clean)
            current_token_count += p_tokens

        # Flush remaining paragraphs
        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunk_list.append(create_chunk(chunk_text))

        # Recursively process child subsections
        for sub in sec.subsections:
            self._chunk_section(doc, sub, chunk_list)

    def chunk_document(self, doc: IngestionDocument) -> List[KnowledgeChunk]:
        """Splits an IngestionDocument into an ordered list of KnowledgeChunks."""
        chunks: List[KnowledgeChunk] = []

        if not doc.sections:
            # Create a single root section from full text if sections are empty
            full_txt = doc.get_full_text()
            if full_txt.strip():
                sec = Section(
                    section_id=f"{doc.document_id[:12]}:sec_root",
                    title=doc.metadata.title or "Root Section",
                    section_type="main",
                    paragraphs=[full_txt],
                )
                self._chunk_section(doc, sec, chunks)
            return chunks

        for sec in doc.sections:
            self._chunk_section(doc, sec, chunks)

        return chunks
