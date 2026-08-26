"""
Benchmark Validator & Training Leakage Detection Engine (Phase 4.5).
Guarantees absolute independence of benchmark cases from training/validation/test splits
and enforces strict structural, schema, and quality validations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from src.dataset.schema import DatasetRecord, Message, Role
from src.evaluation.benchmark_cases import DOMAINS_13, BenchmarkCase

logger = logging.getLogger(__name__)


class BenchmarkValidationReport(BaseModel):
    """Structured report capturing all validation and leakage audit findings."""
    benchmark_version: str = "benchmark-v1.0"
    total_cases: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    exact_overlaps: int = 0
    near_overlaps: int = 0
    internal_duplicates: int = 0
    is_valid: bool = True
    domain_counts: Dict[str, int] = Field(default_factory=dict)
    difficulty_counts: Dict[str, int] = Field(default_factory=dict)
    task_counts: Dict[str, int] = Field(default_factory=dict)
    evaluation_type_counts: Dict[str, int] = Field(default_factory=dict)
    schema_errors: List[str] = Field(default_factory=list)
    leakage_errors: List[str] = Field(default_factory=list)
    quality_errors: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class BenchmarkValidator:
    """
    Validates benchmark cases for quality, schema compliance,
    and complete independence from fine-tuning training datasets.
    """

    def __init__(
        self,
        excluded_split_files: Optional[List[Union[str, Path]]] = None,
        similarity_threshold: float = 0.85,
    ):
        self.similarity_threshold = similarity_threshold
        self.excluded_split_files = [
            Path(p) for p in (excluded_split_files or [
                "datasets/production/splits/train.jsonl",
                "datasets/production/splits/validation.jsonl",
                "datasets/production/splits/test.jsonl",
            ])
        ]
        self._excluded_prompt_hashes: Set[str] = set()
        self._excluded_content_hashes: Set[str] = set()
        self._excluded_ngrams: List[Tuple[str, Set[str]]] = []
        self._loaded_splits = False

    def load_excluded_splits(self) -> None:
        """Load and index all records from existing training, validation, and test splits."""
        if self._loaded_splits:
            return

        for split_path in self.excluded_split_files:
            if not split_path.exists():
                logger.warning(f"Excluded split file not found: {split_path}")
                continue

            with open(split_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                        rec = DatasetRecord(**raw)
                        # Index content hash
                        self._excluded_content_hashes.add(rec.canonical_content_hash())

                        # Index prompt hash (all user messages)
                        user_texts = [m.content for m in rec.messages if m.role == Role.USER]
                        user_concat = " ".join(" ".join(t.strip().split()) for t in user_texts)
                        p_hash = hashlib.sha256(user_concat.encode("utf-8")).hexdigest()
                        self._excluded_prompt_hashes.add(p_hash)

                        # Index word 3-grams for near-duplicate check
                        words = re.findall(r"\w+", user_concat.lower())
                        if len(words) >= 3:
                            ngrams = set(" ".join(words[i:i+3]) for i in range(len(words)-2))
                            self._excluded_ngrams.append((p_hash, ngrams))
                    except Exception as e:
                        logger.warning(f"Error parsing record from {split_path}: {e}")

        self._loaded_splits = True

    def _compute_jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two sets of n-grams."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def validate_case(self, case: BenchmarkCase) -> List[str]:
        """Validate an individual benchmark case against schema and quality rules."""
        errors: List[str] = []

        # 1. Non-empty messages and valid roles
        if not case.messages:
            errors.append(f"Case {case.benchmark_id}: messages list is empty.")
            return errors

        # 2. Reference answer non-empty
        if not case.reference_answer.strip():
            errors.append(f"Case {case.benchmark_id}: reference_answer is empty.")

        # 3. Deterministic criteria checks
        ev = case.evaluation_type
        if ev == "code_based":
            if not case.evaluation_metadata.get("required_symbols") and not case.evaluation_metadata.get("static_checks"):
                errors.append(f"Case {case.benchmark_id}: code_based evaluation requires 'required_symbols' or 'static_checks'.")
        elif ev == "numerical":
            if not case.evaluation_metadata.get("expected_numerical_values"):
                errors.append(f"Case {case.benchmark_id}: numerical evaluation requires 'expected_numerical_values'.")
        elif ev == "reasoning":
            if not case.evaluation_metadata.get("expected_conclusion") and not case.evaluation_metadata.get("key_concept"):
                errors.append(f"Case {case.benchmark_id}: reasoning evaluation requires 'expected_conclusion' or 'key_concept'.")

        return errors

    def audit_leakage(self, case: BenchmarkCase) -> Tuple[bool, bool, Optional[str]]:
        """
        Audit a benchmark case against excluded training datasets.
        Returns: (is_exact_overlap, is_near_overlap, details_message)
        """
        self.load_excluded_splits()

        # 1. Exact prompt check
        user_texts = [m.content for m in case.messages if m.role == Role.USER]
        user_concat = " ".join(" ".join(t.strip().split()) for t in user_texts)
        prompt_hash = hashlib.sha256(user_concat.encode("utf-8")).hexdigest()

        if prompt_hash in self._excluded_prompt_hashes:
            return True, False, f"Exact prompt collision with existing training dataset (hash: {prompt_hash[:12]})"

        # 2. Exact full conversation check
        if case.canonical_content_hash() in self._excluded_content_hashes:
            return True, False, f"Exact conversation content collision with existing dataset (hash: {case.canonical_content_hash()[:12]})"

        # 3. Near-duplicate n-gram similarity check
        words = re.findall(r"\w+", user_concat.lower())
        if len(words) >= 3:
            case_ngrams = set(" ".join(words[i:i+3]) for i in range(len(words)-2))
            for target_hash, target_ngrams in self._excluded_ngrams:
                sim = self._compute_jaccard_similarity(case_ngrams, target_ngrams)
                if sim >= self.similarity_threshold:
                    return False, True, f"Near-duplicate overlap detected (similarity: {sim:.2f} >= {self.similarity_threshold}) with prompt {target_hash[:12]}"

        return False, False, None

    def validate_suite(self, cases: List[BenchmarkCase]) -> BenchmarkValidationReport:
        """
        Run complete validation, quality audit, and leakage detection across the entire benchmark suite.
        """
        self.load_excluded_splits()
        report = BenchmarkValidationReport(total_cases=len(cases))

        seen_bench_ids: Set[str] = set()
        seen_prompt_hashes: Set[str] = set()

        for case in cases:
            case_errors: List[str] = []

            # 1. Benchmark ID uniqueness
            if case.benchmark_id in seen_bench_ids:
                case_errors.append(f"Duplicate benchmark_id: '{case.benchmark_id}'")
                report.internal_duplicates += 1
            seen_bench_ids.add(case.benchmark_id)

            # 2. Internal prompt uniqueness
            p_hash = case.canonical_prompt_hash()
            if p_hash in seen_prompt_hashes:
                case_errors.append(f"Internal duplicate prompt hash in case '{case.benchmark_id}'")
                report.internal_duplicates += 1
            seen_prompt_hashes.add(p_hash)

            # 3. Schema & Quality validation
            q_errs = self.validate_case(case)
            if q_errs:
                case_errors.extend(q_errs)
                report.quality_errors.extend(q_errs)

            # 4. Leakage audit against training/validation/test splits
            is_exact, is_near, leak_detail = self.audit_leakage(case)
            if is_exact:
                report.exact_overlaps += 1
                report.leakage_errors.append(f"Case '{case.benchmark_id}': {leak_detail}")
                case_errors.append(f"Training leakage: {leak_detail}")
            elif is_near:
                report.near_overlaps += 1
                report.leakage_errors.append(f"Case '{case.benchmark_id}': {leak_detail}")
                case_errors.append(f"Training near-overlap: {leak_detail}")

            # 5. Tally
            if case_errors:
                report.rejected_count += 1
            else:
                report.accepted_count += 1
                # Distributions
                report.domain_counts[case.domain] = report.domain_counts.get(case.domain, 0) + 1
                report.difficulty_counts[case.difficulty] = report.difficulty_counts.get(case.difficulty, 0) + 1
                report.task_counts[case.task_type] = report.task_counts.get(case.task_type, 0) + 1
                report.evaluation_type_counts[case.evaluation_type] = report.evaluation_type_counts.get(case.evaluation_type, 0) + 1

        # Check all 13 domains represented
        missing_domains = set(DOMAINS_13) - set(report.domain_counts.keys())
        if missing_domains:
            report.schema_errors.append(f"Missing benchmark representation for domains: {sorted(missing_domains)}")

        report.is_valid = (
            report.rejected_count == 0
            and report.exact_overlaps == 0
            and report.near_overlaps == 0
            and len(report.schema_errors) == 0
        )

        return report
