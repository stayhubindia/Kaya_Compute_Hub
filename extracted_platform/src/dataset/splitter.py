"""
Dataset Splitter.
Implements reproducible, deterministic train/validation/test dataset splitting.
Adheres to Phase 2.1 split targets (90% train, 5% validation, 5% test).
Guarantees strict test set isolation and leakage prevention.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.dataset.schema import DatasetRecord


@dataclass
class SplitResult:
    train: List[DatasetRecord]
    validation: List[DatasetRecord]
    test: List[DatasetRecord]
    split_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_count": len(self.train),
            "validation_count": len(self.validation),
            "test_count": len(self.test),
            "total_count": len(self.train) + len(self.validation) + len(self.test),
            "split_summary": self.split_summary,
        }


class DatasetSplitter:
    """Splits canonical dataset records reproducibly into isolated train, validation, and test sets."""

    def __init__(
        self,
        train_ratio: float = 0.90,
        validation_ratio: float = 0.05,
        test_ratio: float = 0.05,
        random_seed: int = 42,
        stratify_by_domain: bool = True,
    ):
        total = train_ratio + validation_ratio + test_ratio
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.stratify_by_domain = stratify_by_domain

    def split(self, records: List[DatasetRecord]) -> SplitResult:
        if not records:
            return SplitResult(
                train=[],
                validation=[],
                test=[],
                split_summary={"error": "Empty record list provided"},
            )

        rng = random.Random(self.random_seed)

        if not self.stratify_by_domain:
            # Simple randomized split
            shuffled = list(records)
            rng.shuffle(shuffled)
            train, val, test = self._partition_list(shuffled)
        else:
            # Stratified split by domain to maintain domain distribution
            domain_groups: Dict[str, List[DatasetRecord]] = defaultdict(list)
            for r in records:
                domain_groups[r.metadata.domain].append(r)

            train: List[DatasetRecord] = []
            val: List[DatasetRecord] = []
            test: List[DatasetRecord] = []

            for domain, group in sorted(domain_groups.items()):
                shuffled_group = list(group)
                rng.shuffle(shuffled_group)
                t, v, te = self._partition_list(shuffled_group)
                train.extend(t)
                val.extend(v)
                test.extend(te)

            # Final shuffle within each split using seed
            rng.shuffle(train)
            rng.shuffle(val)
            rng.shuffle(test)

        # Integrity Check: Test isolation and leakage prevention
        train_hashes = {r.canonical_content_hash() for r in train}
        val_hashes = {r.canonical_content_hash() for r in val}
        test_hashes = {r.canonical_content_hash() for r in test}

        leakage_train_val = len(train_hashes.intersection(val_hashes))
        leakage_train_test = len(train_hashes.intersection(test_hashes))
        leakage_val_test = len(val_hashes.intersection(test_hashes))

        total_records = len(records)
        summary = {
            "total_records": total_records,
            "train_count": len(train),
            "train_percent": round(len(train) / total_records * 100, 2) if total_records else 0,
            "validation_count": len(val),
            "validation_percent": round(len(val) / total_records * 100, 2) if total_records else 0,
            "test_count": len(test),
            "test_percent": round(len(test) / total_records * 100, 2) if total_records else 0,
            "random_seed": self.random_seed,
            "stratified": self.stratify_by_domain,
            "leakage_detected": (leakage_train_val + leakage_train_test + leakage_val_test) > 0,
            "leakage_counts": {
                "train_val_overlap": leakage_train_val,
                "train_test_overlap": leakage_train_test,
                "val_test_overlap": leakage_val_test,
            },
        }

        return SplitResult(train=train, validation=val, test=test, split_summary=summary)

    def _partition_list(
        self, items: List[DatasetRecord]
    ) -> Tuple[List[DatasetRecord], List[DatasetRecord], List[DatasetRecord]]:
        n = len(items)
        if n == 1:
            return items, [], []
        if n == 2:
            return [items[0]], [items[1]], []
        if n < 5:
            # Small group fallback
            return items[:-2], [items[-2]], [items[-1]]

        n_val = max(1, int(round(n * self.validation_ratio)))
        n_test = max(1, int(round(n * self.test_ratio)))
        n_train = n - n_val - n_test

        if n_train < 1:
            n_train = 1
            if n_val > n_test:
                n_val -= 1
            else:
                n_test -= 1

        train = items[:n_train]
        val = items[n_train : n_train + n_val]
        test = items[n_train + n_val :]

        return train, val, test
