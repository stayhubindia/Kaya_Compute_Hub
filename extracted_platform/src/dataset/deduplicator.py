"""
Dataset Deduplicator.
Implements two-stage deduplication:
1. Exact deduplication via deterministic SHA-256 canonical conversation hashing.
2. Near-duplicate detection via character n-gram MinHash / Jaccard similarity.
Produces comprehensive, traceable deduplication telemetry.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.dataset.schema import DatasetRecord


@dataclass
class DuplicateDetail:
    duplicate_index: int
    matched_index: int
    duplicate_type: str  # "exact" or "near"
    similarity_score: float
    canonical_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duplicate_index": self.duplicate_index,
            "matched_index": self.matched_index,
            "duplicate_type": self.duplicate_type,
            "similarity_score": round(self.similarity_score, 4),
            "canonical_hash": self.canonical_hash,
        }


@dataclass
class DeduplicationReport:
    total_records: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    unique_records: int = 0
    duplicate_details: List[DuplicateDetail] = field(default_factory=list)

    @property
    def duplicate_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return (self.exact_duplicates + self.near_duplicates) / self.total_records

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "unique_records": self.unique_records,
            "duplicate_rate": round(self.duplicate_rate, 4),
            "duplicates_sample": [d.to_dict() for d in self.duplicate_details[:50]],
        }



class DatasetDeduplicator:
    """Detects and eliminates both exact duplicate and near-duplicate conversational records."""

    def __init__(
        self,
        enable_near_dedup: bool = False,
        near_duplicate_threshold: float = 0.85,
        ngram_size: int = 3,
    ):
        self.enable_near_dedup = enable_near_dedup
        self.near_duplicate_threshold = near_duplicate_threshold
        self.ngram_size = ngram_size

    def _extract_ngrams(self, text: str) -> Set[str]:
        """Extracts word n-grams from normalized text."""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < self.ngram_size:
            return {" ".join(words)} if words else {text.lower().strip()}
        return {" ".join(words[i : i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)}

    def _jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Computes Jaccard similarity index between two n-gram sets."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def deduplicate(
        self, records: List[DatasetRecord]
    ) -> Tuple[List[DatasetRecord], DeduplicationReport]:
        report = DeduplicationReport(total_records=len(records))
        unique_records: List[DatasetRecord] = []
        seen_exact_hashes: Dict[str, int] = {}  # hash -> index in unique_records
        retained_ngrams: List[Set[str]] = []
        ngram_index: Dict[str, Set[int]] = defaultdict(set)

        for rec_idx, record in enumerate(records):
            content_hash = record.canonical_content_hash()

            # Stage 1: Exact Hash Deduplication
            if content_hash in seen_exact_hashes:
                matched_idx = seen_exact_hashes[content_hash]
                report.exact_duplicates += 1
                report.duplicate_details.append(
                    DuplicateDetail(
                        duplicate_index=rec_idx,
                        matched_index=matched_idx,
                        duplicate_type="exact",
                        similarity_score=1.0,
                        canonical_hash=content_hash,
                    )
                )
                continue

            # Stage 2: Near-Duplicate Detection via Frequency-Ranked Candidate Index
            is_near_dup = False
            if self.enable_near_dedup:
                combined_text = " ".join([m.content for m in record.messages])
                rec_ngrams = self._extract_ngrams(combined_text)

                # Count shared n-grams across retained records (ignore high-frequency n-grams > 100 docs)
                candidate_counts: Dict[int, int] = defaultdict(int)
                for ng in rec_ngrams:
                    if ng in ngram_index and len(ngram_index[ng]) < 100:
                        for orig_idx in ngram_index[ng]:
                            candidate_counts[orig_idx] += 1

                # Check top-50 most overlapping candidate records
                if candidate_counts:
                    top_candidates = sorted(candidate_counts.keys(), key=lambda idx: candidate_counts[idx], reverse=True)[:50]
                    for orig_idx in top_candidates:
                        orig_ngrams = retained_ngrams[orig_idx]
                        sim = self._jaccard_similarity(rec_ngrams, orig_ngrams)
                        if sim >= self.near_duplicate_threshold:
                            is_near_dup = True
                            report.near_duplicates += 1
                            report.duplicate_details.append(
                                DuplicateDetail(
                                    duplicate_index=rec_idx,
                                    matched_index=orig_idx,
                                    duplicate_type="near",
                                    similarity_score=sim,
                                    canonical_hash=content_hash,
                                )
                            )
                            break

                if is_near_dup:
                    continue

                # Index retained record
                retained_idx = len(unique_records)
                retained_ngrams.append(rec_ngrams)
                for ng in rec_ngrams:
                    ngram_index[ng].add(retained_idx)

            # Record is unique
            seen_exact_hashes[content_hash] = len(unique_records)
            unique_records.append(record)

        report.unique_records = len(unique_records)
        return unique_records, report
