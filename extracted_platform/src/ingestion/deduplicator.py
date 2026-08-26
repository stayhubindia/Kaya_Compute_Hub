"""
Document and Chunk Deduplicator (Phase 3.3).
Performs two-stage deduplication:
1. Exact SHA-256 hash deduplication for documents and chunks.
2. Near-duplicate detection using MinHash fingerprint bucketing (O(n) per chunk,
   replacing the O(n^2) full Jaccard scan which freezes on large corpora).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ingestion.models import IngestionDocument, KnowledgeChunk

logger = logging.getLogger(__name__)

# 16 independent hash seeds for MinHash
_MINHASH_SEEDS = [
    0x9e3779b9, 0x517cc1b7, 0xbf58476d, 0x94d049bb,
    0x6c62272e, 0xc2b2ae35, 0x165667b1, 0xff51afd7,
    0x4be98134, 0xdbc97b49, 0xa54ff53a, 0x3c6ef372,
    0xabb58f65, 0x7fb5d329, 0xd76aa478, 0x02441453,
]
_MINHASH_NUM = len(_MINHASH_SEEDS)
# LSH banding: 4 bands x 4 rows
_BAND_SIZE = 4
_NUM_BANDS = _MINHASH_NUM // _BAND_SIZE


@dataclass
class IngestionDeduplicationReport:
    total_documents: int = 0
    unique_documents: int = 0
    duplicate_documents: int = 0
    total_chunks: int = 0
    unique_chunks: int = 0
    exact_duplicate_chunks: int = 0
    near_duplicate_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "unique_documents": self.unique_documents,
            "duplicate_documents": self.duplicate_documents,
            "total_chunks": self.total_chunks,
            "unique_chunks": self.unique_chunks,
            "exact_duplicate_chunks": self.exact_duplicate_chunks,
            "near_duplicate_chunks": self.near_duplicate_chunks,
        }


class IngestionDeduplicator:
    """Detects and filters duplicate documents and knowledge chunks.

    Near-dedup uses MinHash + LSH banding so cost is O(n) per chunk
    instead of O(n^2) full Jaccard scan. Safe for large corpora.
    """

    def __init__(
        self,
        enable_near_dedup: bool = True,
        near_duplicate_threshold: float = 0.85,
        ngram_size: int = 3,
    ):
        self.enable_near_dedup = enable_near_dedup
        self.near_duplicate_threshold = near_duplicate_threshold
        self.ngram_size = ngram_size

        self._seen_doc_hashes: Set[str] = set()
        self._seen_chunk_hashes: Set[str] = set()
        # LSH band buckets: band_idx -> {bucket_key -> chunk_id}
        self._lsh_buckets: List[Dict[str, str]] = [{} for _ in range(_NUM_BANDS)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_ngrams(self, text: str) -> Set[str]:
        clean = re.sub(r"\s+", " ", text.lower().strip())
        if len(clean) < self.ngram_size:
            return {clean}
        return {clean[i : i + self.ngram_size] for i in range(len(clean) - self.ngram_size + 1)}

    def _minhash(self, ngrams: Set[str]) -> List[int]:
        """Compute MinHash signature of length _MINHASH_NUM for a set of n-grams."""
        sig = [2**32] * _MINHASH_NUM
        for gram in ngrams:
            gram_int = int(hashlib.md5(gram.encode("utf-8", errors="replace")).hexdigest(), 16)
            for i, seed in enumerate(_MINHASH_SEEDS):
                h = (gram_int ^ seed) & 0xFFFFFFFF
                if h < sig[i]:
                    sig[i] = h
        return sig

    def _lsh_band_keys(self, sig: List[int]) -> List[str]:
        """Split signature into bands and return a bucket key per band."""
        keys = []
        for b in range(_NUM_BANDS):
            band = sig[b * _BAND_SIZE : (b + 1) * _BAND_SIZE]
            keys.append(f"b{b}:" + "-".join(str(v) for v in band))
        return keys

    def _is_near_duplicate_lsh(self, sig: List[int]) -> bool:
        """True if any LSH band bucket already contains a matching signature."""
        for b, key in enumerate(self._lsh_band_keys(sig)):
            if key in self._lsh_buckets[b]:
                return True
        return False

    def _register_lsh(self, chunk_id: str, sig: List[int]) -> None:
        """Insert the chunk's band keys into LSH buckets."""
        for b, key in enumerate(self._lsh_band_keys(sig)):
            self._lsh_buckets[b].setdefault(key, chunk_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_duplicate_document(self, doc_id: str) -> bool:
        """Returns True if the document has already been processed."""
        if doc_id in self._seen_doc_hashes:
            return True
        self._seen_doc_hashes.add(doc_id)
        return False

    def deduplicate_chunks(
        self, chunks: List[KnowledgeChunk]
    ) -> Tuple[List[KnowledgeChunk], IngestionDeduplicationReport]:
        """Filters exact and near-duplicate chunks via MinHash LSH — O(n) per chunk."""
        report = IngestionDeduplicationReport(total_chunks=len(chunks))
        unique_chunks: List[KnowledgeChunk] = []

        for chunk in chunks:
            # Stage 1: Exact hash deduplication
            text_hash = hashlib.sha256(chunk.text.strip().encode("utf-8", errors="replace")).hexdigest()
            if text_hash in self._seen_chunk_hashes:
                report.exact_duplicate_chunks += 1
                continue

            # Stage 2: Near-duplicate detection via MinHash + LSH banding
            if self.enable_near_dedup:
                ngrams = self._extract_ngrams(chunk.text)
                sig = self._minhash(ngrams)
                if self._is_near_duplicate_lsh(sig):
                    report.near_duplicate_chunks += 1
                    continue
                self._register_lsh(chunk.chunk_id, sig)

            self._seen_chunk_hashes.add(text_hash)
            unique_chunks.append(chunk)

        report.unique_chunks = len(unique_chunks)
        return unique_chunks, report
