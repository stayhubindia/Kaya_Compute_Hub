"""
Source-Aware Dataset Splitter (Phase 3.4).
Prevents data leakage by partitioning dataset records strictly by source grouping:
Hierarchy: document_id -> section_id -> chunk_id
Ensures examples derived from the same chunk/section remain co-located in the same split.
Adheres to 90% train, 5% validation, 5% test with deterministic seed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from src.dataset.schema import DatasetRecord


@dataclass
class SourceAwareSplitResult:
    train: List[DatasetRecord]
    validation: List[DatasetRecord]
    test: List[DatasetRecord]
    leakage_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_count": len(self.train),
            "validation_count": len(self.validation),
            "test_count": len(self.test),
            "total_count": len(self.train) + len(self.validation) + len(self.test),
            "leakage_summary": self.leakage_summary,
        }


class SourceAwareSplitter:
    """Partitions dataset records into isolated splits preventing source-level and content leakage."""

    def __init__(
        self,
        train_ratio: float = 0.90,
        validation_ratio: float = 0.05,
        test_ratio: float = 0.05,
        random_seed: int = 42,
    ):
        total = train_ratio + validation_ratio + test_ratio
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split(self, records: List[DatasetRecord]) -> SourceAwareSplitResult:
        """Partitions records by clustering on source group (document/section/chunk)."""
        if not records:
            return SourceAwareSplitResult(
                train=[],
                validation=[],
                test=[],
                leakage_summary={"error": "Empty record list provided"},
            )

        rng = random.Random(self.random_seed)

        # 1. Cluster records by primary source grouping key
        # Priority: (document_id, section_id) -> chunk_id -> source_id
        clusters: Dict[str, List[DatasetRecord]] = defaultdict(list)
        for r in records:
            extra = getattr(r.metadata, "extra", {}) or {}
            doc_id = extra.get("document_id") or (r.metadata.provenance.source_id if r.metadata.provenance else None) or "doc_unknown"
            sec_id = extra.get("section_id") or "sec_unknown"
            cluster_key = f"{doc_id}::{sec_id}"
            clusters[cluster_key].append(r)

        # 2. Shuffle cluster keys deterministically
        sorted_cluster_keys = sorted(clusters.keys())
        rng.shuffle(sorted_cluster_keys)

        total_records = len(records)
        target_val = max(1, int(round(total_records * self.validation_ratio)))
        target_test = max(1, int(round(total_records * self.test_ratio)))

        train: List[DatasetRecord] = []
        val: List[DatasetRecord] = []
        test: List[DatasetRecord] = []

        curr_val_count = 0
        curr_test_count = 0

        # 3. Assign whole clusters to splits
        for c_key in sorted_cluster_keys:
            cluster_recs = clusters[c_key]
            c_size = len(cluster_recs)

            if curr_val_count + c_size <= target_val:
                val.extend(cluster_recs)
                curr_val_count += c_size
            elif curr_test_count + c_size <= target_test:
                test.extend(cluster_recs)
                curr_test_count += c_size
            else:
                train.extend(cluster_recs)

        # Fallback if val or test is empty due to large cluster granularity
        if not val and len(train) > 2:
            val.append(train.pop())
        if not test and len(train) > 2:
            test.append(train.pop())

        # Shuffle internally within each split
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        # 4. Leakage Verification
        train_hashes = {r.canonical_content_hash() for r in train}
        val_hashes = {r.canonical_content_hash() for r in val}
        test_hashes = {r.canonical_content_hash() for r in test}

        train_chunks = {getattr(r.metadata, "extra", {}).get("chunk_id") for r in train if hasattr(r, "metadata")}
        val_chunks = {getattr(r.metadata, "extra", {}).get("chunk_id") for r in val if hasattr(r, "metadata")}
        test_chunks = {getattr(r.metadata, "extra", {}).get("chunk_id") for r in test if hasattr(r, "metadata")}

        hash_overlap_tv = len(train_hashes.intersection(val_hashes))
        hash_overlap_tt = len(train_hashes.intersection(test_hashes))
        hash_overlap_vt = len(val_hashes.intersection(test_hashes))

        chunk_overlap_tv = len(train_chunks.intersection(val_chunks) - {None})
        chunk_overlap_tt = len(train_chunks.intersection(test_chunks) - {None})
        chunk_overlap_vt = len(val_chunks.intersection(test_chunks) - {None})

        leakage_detected = (
            hash_overlap_tv + hash_overlap_tt + hash_overlap_vt + chunk_overlap_tv + chunk_overlap_tt + chunk_overlap_vt
        ) > 0

        summary = {
            "total_records": total_records,
            "train_count": len(train),
            "train_pct": round(len(train) / total_records * 100, 2) if total_records else 0,
            "validation_count": len(val),
            "validation_pct": round(len(val) / total_records * 100, 2) if total_records else 0,
            "test_count": len(test),
            "test_pct": round(len(test) / total_records * 100, 2) if total_records else 0,
            "seed": self.random_seed,
            "leakage_detected": leakage_detected,
            "hash_overlaps": {
                "train_val": hash_overlap_tv,
                "train_test": hash_overlap_tt,
                "val_test": hash_overlap_vt,
            },
            "chunk_overlaps": {
                "train_val": chunk_overlap_tv,
                "train_test": chunk_overlap_tt,
                "val_test": chunk_overlap_vt,
            },
        }

        return SourceAwareSplitResult(train=train, validation=val, test=test, leakage_summary=summary)
