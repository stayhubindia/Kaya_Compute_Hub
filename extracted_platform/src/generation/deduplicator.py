"""
Instruction Candidate Deduplicator (Phase 3.4).
Implements exact SHA-256 canonical conversation deduplication
and near-duplicate Jaccard n-gram similarity detection for instruction candidates.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.dataset.deduplicator import DeduplicationReport, DuplicateDetail
from src.dataset.schema import DatasetRecord, Role


class InstructionDeduplicator:
    """Performs exact and near-duplicate filtering across generated instruction candidates."""

    def __init__(
        self,
        enable_exact: bool = True,
        enable_near: bool = True,
        near_threshold: float = 0.85,
        ngram_size: int = 3,
    ):
        self.enable_exact = enable_exact
        self.enable_near = enable_near
        self.near_threshold = near_threshold
        self.ngram_size = ngram_size

    def _get_ngrams(self, text: str) -> Set[str]:
        clean = " ".join(re.findall(r"\w+", text.lower()))
        if len(clean) < self.ngram_size:
            return {clean} if clean else set()
        return {clean[i : i + self.ngram_size] for i in range(len(clean) - self.ngram_size + 1)}

    def _jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def deduplicate(self, records: List[DatasetRecord]) -> Tuple[List[DatasetRecord], DeduplicationReport]:
        """Deduplicates a list of DatasetRecord objects, returning unique records and detailed report."""
        report = DeduplicationReport(total_records=len(records))
        unique_records: List[DatasetRecord] = []
        seen_hashes: Dict[str, int] = {}
        unique_ngrams: List[Tuple[int, Set[str]]] = []

        for idx, record in enumerate(records):
            # 1. Exact Deduplication
            c_hash = record.canonical_content_hash()
            if self.enable_exact and c_hash in seen_hashes:
                report.exact_duplicates += 1
                report.duplicate_details.append(
                    DuplicateDetail(
                        duplicate_index=idx,
                        matched_index=seen_hashes[c_hash],
                        duplicate_type="exact",
                        similarity_score=1.0,
                        canonical_hash=c_hash,
                    )
                )
                continue

            # 2. Near Deduplication
            content_repr = " ".join(m.content for m in record.messages)
            record_ngrams = self._get_ngrams(content_repr)

            is_near_dup = False
            if self.enable_near and record_ngrams:
                for orig_idx, orig_ngrams in unique_ngrams:
                    sim = self._jaccard_similarity(record_ngrams, orig_ngrams)
                    if sim >= self.near_threshold:
                        is_near_dup = True
                        report.near_duplicates += 1
                        report.duplicate_details.append(
                            DuplicateDetail(
                                duplicate_index=idx,
                                matched_index=orig_idx,
                                duplicate_type="near",
                                similarity_score=sim,
                                canonical_hash=c_hash,
                            )
                        )
                        break

            if is_near_dup:
                continue

            # Passed deduplication
            seen_hashes[c_hash] = idx
            unique_ngrams.append((idx, record_ngrams))
            unique_records.append(record)

        report.unique_records = len(unique_records)
        return unique_records, report
