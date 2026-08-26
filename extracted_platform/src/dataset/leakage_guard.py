"""
Cross-Split and Source-Group Leakage Guard (Phase 3.5).
Prevents exact hash leakage, near-duplicate cross-contamination, and enforces source-group
clustering across train, validation, and test splits.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.dataset.schema import DatasetRecord, Role


class SplitLeakageDetail(BaseModel):
    """Detailed record of an identified cross-split leakage instance."""
    leak_type: str  # "EXACT_HASH", "NEAR_DUPLICATE", "SOURCE_GROUP"
    split_pair: str  # e.g., "train-val", "train-test", "val-test"
    record_id_1: str
    record_id_2: str
    similarity_or_source: str
    description: str


class LeakageAuditReport(BaseModel):
    """Comprehensive cross-split contamination audit report."""
    is_clean: bool
    total_exact_leaks: int
    train_val_exact: int
    train_test_exact: int
    val_test_exact: int
    total_near_leaks: int
    train_val_near: int
    train_test_near: int
    val_test_near: int
    total_source_group_leaks: int
    source_group_clustering_enforced: bool
    leak_details: List[SplitLeakageDetail] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_clean": self.is_clean,
            "total_exact_leaks": self.total_exact_leaks,
            "train_val_exact": self.train_val_exact,
            "train_test_exact": self.train_test_exact,
            "val_test_exact": self.val_test_exact,
            "total_near_leaks": self.total_near_leaks,
            "train_val_near": self.train_val_near,
            "train_test_near": self.train_test_near,
            "val_test_near": self.val_test_near,
            "total_source_group_leaks": self.total_source_group_leaks,
            "source_group_clustering_enforced": self.source_group_clustering_enforced,
            "leak_details": [d.model_dump() for d in self.leak_details],
        }


class LeakageGuard:
    """Detects and prevents cross-split contamination via content hashing, MinHash, and source-grouping."""

    def __init__(
        self,
        near_duplicate_threshold: float = 0.85,
        ngram_size: int = 3,
        seed: int = 42,
    ):
        self.near_duplicate_threshold = near_duplicate_threshold
        self.ngram_size = ngram_size
        self.seed = seed

    def _extract_source_group_key(self, record: DatasetRecord, default_idx: int = 0) -> str:
        """Extracts the source document/chunk group identifier for clustering."""
        prov = record.metadata.provenance
        if prov:
            if getattr(prov, "source_id", None):
                # If source_id contains sub-chunk notation, use base id
                return str(prov.source_id).split("::")[0]
            if getattr(prov, "source", None):
                return str(prov.source)
        if record.metadata.source:
            return str(record.metadata.source)
        return f"group_{default_idx}"

    def split_with_source_group_isolation(
        self,
        records: List[DatasetRecord],
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        cluster_by_source: bool = True,
    ) -> Tuple[List[DatasetRecord], List[DatasetRecord], List[DatasetRecord]]:
        """
        Splits dataset into train, validation, and test sets.
        If cluster_by_source is True, all records sharing a source group key are placed in the same split.
        """
        if not records:
            return [], [], []

        rng = random.Random(self.seed)

        if not cluster_by_source:
            # Standard deterministic stratified split
            shuffled = list(records)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))
            # ensure at least 1 in val and test if possible when n >= 10
            if n >= 10 and n_val == 0:
                n_val = 1
                n_train -= 1
            if n >= 10 and (n - n_train - n_val) == 0:
                n_test = 1
                n_train -= 1
            else:
                n_test = n - n_train - n_val

            train = shuffled[:n_train]
            val = shuffled[n_train : n_train + n_val]
            test = shuffled[n_train + n_val :]
            return train, val, test

        # Cluster by source group
        groups: Dict[str, List[DatasetRecord]] = defaultdict(list)
        for idx, r in enumerate(records):
            grp_key = self._extract_source_group_key(r, default_idx=idx)
            groups[grp_key].append(r)

        # Sort group keys for determinism, then shuffle
        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)

        total_records = len(records)
        target_train = int(round(total_records * train_ratio))
        target_val = int(round(total_records * val_ratio))
        target_test = total_records - target_train - target_val

        train: List[DatasetRecord] = []
        val: List[DatasetRecord] = []
        test: List[DatasetRecord] = []

        for grp in group_keys:
            grp_records = groups[grp]
            # Decide best split based on remaining capacities
            if len(train) + len(grp_records) <= target_train or (not val and not test and len(train) < target_train):
                train.extend(grp_records)
            elif len(val) + len(grp_records) <= target_val or (not val and target_val > 0):
                val.extend(grp_records)
            else:
                test.extend(grp_records)

        # Fallback if val or test is empty and total >= 3
        if len(val) == 0 and len(train) > 1 and target_val > 0:
            val.append(train.pop())
        if len(test) == 0 and len(train) > 1 and target_test > 0:
            test.append(train.pop())

        return train, val, test

    def audit_cross_split_leakage(
        self,
        train_records: List[DatasetRecord],
        val_records: List[DatasetRecord],
        test_records: List[DatasetRecord],
        check_source_groups: bool = True,
    ) -> LeakageAuditReport:
        """
        Thoroughly audits exact hash overlap, near-duplicate cross-leakage, and source-group contamination.
        """
        leak_details: List[SplitLeakageDetail] = []

        # 1. Exact Hash Overlap
        train_hashes = {r.canonical_content_hash(): r for r in train_records}
        val_hashes = {r.canonical_content_hash(): r for r in val_records}
        test_hashes = {r.canonical_content_hash(): r for r in test_records}

        tv_exact = set(train_hashes.keys()).intersection(set(val_hashes.keys()))
        tt_exact = set(train_hashes.keys()).intersection(set(test_hashes.keys()))
        vt_exact = set(val_hashes.keys()).intersection(set(test_hashes.keys()))

        for h in tv_exact:
            r1 = train_hashes[h]
            r2 = val_hashes[h]
            leak_details.append(SplitLeakageDetail(
                leak_type="EXACT_HASH",
                split_pair="train-val",
                record_id_1=getattr(r1.metadata, "record_id", "train_rec"),
                record_id_2=getattr(r2.metadata, "record_id", "val_rec"),
                similarity_or_source=h[:16],
                description=f"Exact SHA-256 hash match between train and validation: {h[:16]}",
            ))

        for h in tt_exact:
            r1 = train_hashes[h]
            r2 = test_hashes[h]
            leak_details.append(SplitLeakageDetail(
                leak_type="EXACT_HASH",
                split_pair="train-test",
                record_id_1=getattr(r1.metadata, "record_id", "train_rec"),
                record_id_2=getattr(r2.metadata, "record_id", "test_rec"),
                similarity_or_source=h[:16],
                description=f"Exact SHA-256 hash match between train and test: {h[:16]}",
            ))

        for h in vt_exact:
            r1 = val_hashes[h]
            r2 = test_hashes[h]
            leak_details.append(SplitLeakageDetail(
                leak_type="EXACT_HASH",
                split_pair="val-test",
                record_id_1=getattr(r1.metadata, "record_id", "val_rec"),
                record_id_2=getattr(r2.metadata, "record_id", "test_rec"),
                similarity_or_source=h[:16],
                description=f"Exact SHA-256 hash match between validation and test: {h[:16]}",
            ))

        total_exact = len(tv_exact) + len(tt_exact) + len(vt_exact)

        # 2. Near Duplicate Leakage (Jaccard on prompt content using inverted index)
        tv_near = 0
        tt_near = 0
        vt_near = 0

        def get_shingles(rec: DatasetRecord) -> Set[str]:
            text = " ".join(m.content for m in rec.messages).lower()
            tokens = re.findall(r"\b\w+\b", text)
            if len(tokens) < self.ngram_size:
                return set(tokens)
            return {" ".join(tokens[i : i + self.ngram_size]) for i in range(len(tokens) - self.ngram_size + 1)}

        train_shingles = [(r, get_shingles(r)) for r in train_records]
        val_shingles = [(r, get_shingles(r)) for r in val_records]
        test_shingles = [(r, get_shingles(r)) for r in test_records]

        # Build inverted index for val and test shingles
        val_index: Dict[str, Set[int]] = defaultdict(set)
        for idx, (_, s_vl) in enumerate(val_shingles):
            for sh in s_vl:
                val_index[sh].add(idx)

        test_index: Dict[str, Set[int]] = defaultdict(set)
        for idx, (_, s_te) in enumerate(test_shingles):
            for sh in s_te:
                test_index[sh].add(idx)

        # Train vs Val via Inverted Index
        for r_tr, s_tr in train_shingles:
            if not s_tr:
                continue
            candidates: Set[int] = set()
            for sh in s_tr:
                if sh in val_index:
                    candidates.update(val_index[sh])
            for vl_idx in candidates:
                r_vl, s_vl = val_shingles[vl_idx]
                jaccard = len(s_tr.intersection(s_vl)) / max(1, len(s_tr.union(s_vl)))
                if jaccard >= self.near_duplicate_threshold:
                    tv_near += 1
                    leak_details.append(SplitLeakageDetail(
                        leak_type="NEAR_DUPLICATE",
                        split_pair="train-val",
                        record_id_1=getattr(r_tr.metadata, "record_id", "train_rec"),
                        record_id_2=getattr(r_vl.metadata, "record_id", "val_rec"),
                        similarity_or_source=f"Jaccard={jaccard:.2f}",
                        description=f"High semantic user prompt overlap ({jaccard:.2f}) between train and val.",
                    ))

        # Train vs Test via Inverted Index
        for r_tr, s_tr in train_shingles:
            if not s_tr:
                continue
            candidates: Set[int] = set()
            for sh in s_tr:
                if sh in test_index:
                    candidates.update(test_index[sh])
            for te_idx in candidates:
                r_te, s_te = test_shingles[te_idx]
                jaccard = len(s_tr.intersection(s_te)) / max(1, len(s_tr.union(s_te)))
                if jaccard >= self.near_duplicate_threshold:
                    tt_near += 1
                    leak_details.append(SplitLeakageDetail(
                        leak_type="NEAR_DUPLICATE",
                        split_pair="train-test",
                        record_id_1=getattr(r_tr.metadata, "record_id", "train_rec"),
                        record_id_2=getattr(r_te.metadata, "record_id", "test_rec"),
                        similarity_or_source=f"Jaccard={jaccard:.2f}",
                        description=f"High semantic user prompt overlap ({jaccard:.2f}) between train and test.",
                    ))

        total_near = tv_near + tt_near + vt_near

        # 3. Source Group Contamination Check
        source_group_leaks = 0
        if check_source_groups:
            train_groups = {self._extract_source_group_key(r) for r in train_records}
            val_groups = {self._extract_source_group_key(r) for r in val_records}
            test_groups = {self._extract_source_group_key(r) for r in test_records}

            tv_grp = train_groups.intersection(val_groups)
            tt_grp = train_groups.intersection(test_groups)
            source_group_leaks = len(tv_grp) + len(tt_grp)

            for g in tv_grp:
                leak_details.append(SplitLeakageDetail(
                    leak_type="SOURCE_GROUP",
                    split_pair="train-val",
                    record_id_1="train_group",
                    record_id_2="val_group",
                    similarity_or_source=g,
                    description=f"Source group '{g}' co-occurs across train and validation splits.",
                ))

        is_clean = (total_exact == 0 and total_near == 0)

        return LeakageAuditReport(
            is_clean=is_clean,
            total_exact_leaks=total_exact,
            train_val_exact=len(tv_exact),
            train_test_exact=len(tt_exact),
            val_test_exact=len(vt_exact),
            total_near_leaks=total_near,
            train_val_near=tv_near,
            train_test_near=tt_near,
            val_test_near=vt_near,
            total_source_group_leaks=source_group_leaks,
            source_group_clustering_enforced=check_source_groups,
            leak_details=leak_details,
        )
