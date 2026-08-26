"""
Dataset Mixing & Balancing Engine (Phase 2.3.4).
Combines heterogeneous dataset sources into a unified, balanced dataset with
configurable domain, difficulty, task, and source distributions.
Enforces deterministic selection, immutable provenance preservation,
explicit shortage tracking, and comprehensive telemetry reporting.
"""

from __future__ import annotations

import copy
import json
import math
import random
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from src.dataset.deduplicator import DatasetDeduplicator
from src.dataset.loader import DatasetLoader, RawRecord
from src.dataset.normalizer import DatasetNormalizer
from src.dataset.schema import (
    DatasetRecord,
    DifficultyLevel,
    ProvenanceInfo,
    RecordMetadata,
    Role,
    SourceType,
    TaskType,
)


# ============================================================================
# 1. DATA MODELS & REPORT CONTAINERS
# ============================================================================

class DistributionReport(BaseModel):
    """Distribution metrics for a specific dimension (domain, difficulty, task, source)."""
    dimension: str
    counts: Dict[str, int] = Field(default_factory=dict)
    percentages: Dict[str, float] = Field(default_factory=dict)
    targets: Dict[str, float] = Field(default_factory=dict)
    deviations: Dict[str, float] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "counts": self.counts,
            "percentages": self.percentages,
            "targets": self.targets,
            "deviations": self.deviations,
        }


class ShortageDetail(BaseModel):
    """Detailed record of candidate shortages for a specific stratum."""
    category: str
    dimension: str
    requested: int
    available: int
    selected: int
    shortage: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "dimension": self.dimension,
            "requested": self.requested,
            "available": self.available,
            "selected": self.selected,
            "shortage": self.shortage,
        }


class OversamplingDetail(BaseModel):
    """Telemetry regarding oversampling operations."""
    oversampled_records: int = 0
    unique_records_oversampled: int = 0
    oversampling_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "oversampled_records": self.oversampled_records,
            "unique_records_oversampled": self.unique_records_oversampled,
            "oversampling_ratio": self.oversampling_ratio,
        }


class MixingRequest(BaseModel):
    """Strongly-typed configuration request for dataset mixing."""
    input_sources: List[Union[str, Path]] = Field(
        default_factory=list,
        description="List of file or directory paths containing candidate dataset records.",
    )
    target_count: int = Field(gt=0, description="Target total count of examples in unified dataset.")
    strategy: str = Field(default="proportional", description="Mixing strategy: 'proportional' or 'balanced'.")
    seed: int = Field(default=42, description="Deterministic random seed.")
    allow_oversampling: bool = Field(default=False, description="Whether to allow repeating records on shortage.")
    allow_undersampling: bool = Field(default=True, description="Whether to allow subsampling excess candidate records.")
    preserve_source_provenance: bool = Field(default=True, description="Preserve original provenance on all records.")
    enforce_domain_targets: bool = Field(default=True, description="Enforce domain distribution weights.")
    enforce_difficulty_targets: bool = Field(default=True, description="Enforce difficulty distribution weights.")
    enforce_task_targets: bool = Field(default=False, description="Enforce optional task distribution weights.")
    enforce_source_targets: bool = Field(default=False, description="Enforce optional source distribution weights.")
    domain_targets: Optional[Dict[str, float]] = None
    difficulty_targets: Optional[Dict[str, float]] = None
    task_targets: Optional[Dict[str, float]] = None
    source_targets: Optional[Dict[str, float]] = None
    batch_id: Optional[str] = None
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("strategy")
    @classmethod
    def validate_strategy_name(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in ["proportional", "balanced"]:
            raise ValueError(f"Unsupported mixing strategy '{v}'. Allowed: ['proportional', 'balanced'].")
        return s

    def get_effective_batch_id(self) -> str:
        if self.batch_id and self.batch_id.strip():
            return self.batch_id.strip()
        uid = uuid.uuid4().hex[:8]
        return f"mix_{self.strategy}_n{self.target_count}_s{self.seed}_{uid}"


class MixingResult(BaseModel):
    """Complete output and audit report from dataset mixing engine."""
    records: List[DatasetRecord] = Field(default_factory=list)
    requested_count: int
    selected_count: int
    total_candidates: int
    strategy: str
    seed: int
    batch_id: str
    domain_distribution: DistributionReport
    difficulty_distribution: DistributionReport
    task_distribution: DistributionReport
    source_distribution: DistributionReport
    shortages: List[ShortageDetail] = Field(default_factory=list)
    oversampling: Optional[OversamplingDetail] = None
    discarded_count: int = 0
    errors: List[str] = Field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        return len(self.errors) == 0 and len(self.records) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_count": self.requested_count,
            "selected_count": self.selected_count,
            "total_candidates": self.total_candidates,
            "discarded_count": self.discarded_count,
            "strategy": self.strategy,
            "seed": self.seed,
            "batch_id": self.batch_id,
            "domain_distribution": self.domain_distribution.to_dict(),
            "difficulty_distribution": self.difficulty_distribution.to_dict(),
            "task_distribution": self.task_distribution.to_dict(),
            "source_distribution": self.source_distribution.to_dict(),
            "shortages": [s.to_dict() for s in self.shortages],
            "oversampling": self.oversampling.to_dict() if self.oversampling else None,
            "errors": self.errors,
        }

    def save_jsonl(self, output_path: Union[str, Path], overwrite: bool = False) -> int:
        path = Path(output_path).resolve()
        if path.is_file() and not overwrite:
            raise FileExistsError(f"Output file '{path}' exists. Pass overwrite=True to replace.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(r.to_json() + "\n")
        return len(self.records)

    def generate_markdown_report(self) -> str:
        """Generates comprehensive GitHub Flavored Markdown mixing audit report."""
        lines = [
            "# Dataset Mixing & Balancing Engine Audit Report",
            "",
            "## Executive Summary",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Mixing Batch ID** | `{self.batch_id}` |",
            f"| **Mixing Strategy** | `{self.strategy}` |",
            f"| **Deterministic Seed** | `{self.seed}` |",
            f"| **Total Candidates Ingested** | `{self.total_candidates}` |",
            f"| **Target Requested Count** | `{self.requested_count}` |",
            f"| **Final Selected Count** | `{self.selected_count}` |",
            f"| **Undersampled Discarded** | `{self.discarded_count}` |",
        ]

        if self.oversampling:
            lines.append(f"| **Oversampled Copies** | `{self.oversampling.oversampled_records}` |")
            lines.append(f"| **Oversampling Ratio** | `{self.oversampling.oversampling_ratio:.3f}` |")

        lines.extend([
            "",
            "---",
            "",
            "## 1. Domain Distribution & Target Deviations",
            "",
            "| Domain | Target % | Actual Count | Actual % | Deviation |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for dom, count in sorted(self.domain_distribution.counts.items()):
            actual_pct = self.domain_distribution.percentages.get(dom, 0.0)
            target_pct = self.domain_distribution.targets.get(dom, 0.0)
            dev = self.domain_distribution.deviations.get(dom, 0.0)
            dev_str = f"{dev:+.2f}%" if target_pct > 0 else "N/A"
            lines.append(
                f"| `{dom}` | {target_pct:.1f}% | `{count}` | {actual_pct:.1f}% | {dev_str} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 2. Difficulty Distribution",
            "",
            "| Difficulty Level | Target % | Actual Count | Actual % | Deviation |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for diff, count in sorted(self.difficulty_distribution.counts.items()):
            actual_pct = self.difficulty_distribution.percentages.get(diff, 0.0)
            target_pct = self.difficulty_distribution.targets.get(diff, 0.0)
            dev = self.difficulty_distribution.deviations.get(diff, 0.0)
            dev_str = f"{dev:+.2f}%" if target_pct > 0 else "N/A"
            lines.append(
                f"| `{diff}` | {target_pct:.1f}% | `{count}` | {actual_pct:.1f}% | {dev_str} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. Task Type Distribution",
            "",
            "| Task Type | Count | Actual % |",
            "| :--- | :--- | :--- |",
        ])

        for task, count in sorted(self.task_distribution.counts.items()):
            actual_pct = self.task_distribution.percentages.get(task, 0.0)
            lines.append(f"| `{task}` | `{count}` | {actual_pct:.1f}% |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Source & Provenance Distribution",
            "",
            "| Source Type | Target % | Count | Actual % |",
            "| :--- | :--- | :--- | :--- |",
        ])

        for src, count in sorted(self.source_distribution.counts.items()):
            actual_pct = self.source_distribution.percentages.get(src, 0.0)
            target_pct = self.source_distribution.targets.get(src, 0.0)
            target_str = f"{target_pct:.1f}%" if target_pct > 0 else "N/A"
            lines.append(f"| `{src}` | {target_str} | `{count}` | {actual_pct:.1f}% |")

        lines.extend([
            "",
            "---",
            "",
            "## 5. Candidate Shortages & Deficit Tracking",
            "",
        ])

        if not self.shortages:
            lines.append("✅ **No candidate shortages recorded.** All requested quotas were completely satisfied.")
        else:
            lines.extend([
                "| Stratum Category | Dimension | Requested Quota | Available Candidates | Selected | Shortage Deficit |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for s in self.shortages:
                lines.append(
                    f"| `{s.category}` | `{s.dimension}` | `{s.requested}` | `{s.available}` | `{s.selected}` | **`{s.shortage}`** |"
                )

        lines.append("")
        return "\n".join(lines)

    def save_reports(self, output_dir: Union[str, Path]) -> Tuple[Path, Path]:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "dataset_mix_report.json"
        md_path = out_dir / "dataset_mix_report.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown_report())

        return json_path, md_path


# ============================================================================
# 2. MIXING STRATEGIES (BASE, PROPORTIONAL, BALANCED)
# ============================================================================

class BaseMixingStrategy(ABC):
    """Abstract base class for dataset mixing strategies."""

    @abstractmethod
    def select(
        self,
        candidates: List[DatasetRecord],
        request: MixingRequest,
        domain_targets: Dict[str, float],
        difficulty_targets: Dict[str, float],
        task_targets: Optional[Dict[str, float]] = None,
        source_targets: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[DatasetRecord], List[ShortageDetail], Optional[OversamplingDetail], int]:
        """
        Executes selection algorithm returning (selected_records, shortages, oversampling, discarded_count).
        """
        pass


class ProportionalMixingStrategy(BaseMixingStrategy):
    """
    Proportional Stratified Selection Strategy.
    Allocates records to match exact target distribution percentages.
    Partitions candidates into domain x difficulty strata, computes integer quotas
    via Hare-Niemeyer (Largest Remainder) allocation, deterministically samples candidates,
    and accurately tracks shortages, oversampling, and undersampling.
    """

    def select(
        self,
        candidates: List[DatasetRecord],
        request: MixingRequest,
        domain_targets: Dict[str, float],
        difficulty_targets: Dict[str, float],
        task_targets: Optional[Dict[str, float]] = None,
        source_targets: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[DatasetRecord], List[ShortageDetail], Optional[OversamplingDetail], int]:
        rng = random.Random(request.seed)
        target_n = request.target_count

        # 1. Allocate Domain Quotas using Largest Remainder Method
        domain_quotas = self._allocate_quotas(target_n, domain_targets)

        # 2. Partition candidates by domain -> difficulty
        pool: Dict[str, Dict[str, List[DatasetRecord]]] = defaultdict(lambda: defaultdict(list))
        for r in candidates:
            dom = r.metadata.domain
            diff = r.metadata.difficulty
            pool[dom][diff].append(r)

        selected_records: List[DatasetRecord] = []
        shortages: List[ShortageDetail] = []
        oversampled_records_count = 0
        unique_oversampled_set: Set[str] = set()

        total_candidates_count = len(candidates)

        # 3. Stratified Sub-allocation & Deterministic Sampling
        for dom, dom_quota in sorted(domain_quotas.items()):
            if dom_quota <= 0:
                continue

            # Sub-allocate domain quota to difficulty levels
            diff_quotas = self._allocate_quotas(dom_quota, difficulty_targets)
            dom_candidates = [r for diff_list in pool[dom].values() for r in diff_list]
            dom_available = len(dom_candidates)

            # Check domain-level candidate availability
            if dom_available < dom_quota and not request.allow_oversampling:
                shortages.append(
                    ShortageDetail(
                        category=dom,
                        dimension="domain",
                        requested=dom_quota,
                        available=dom_available,
                        selected=dom_available,
                        shortage=dom_quota - dom_available,
                    )
                )

            for diff, diff_quota in sorted(diff_quotas.items()):
                if diff_quota <= 0:
                    continue

                available_in_stratum = list(pool[dom][diff])
                available_count = len(available_in_stratum)

                # Deterministically shuffle stratum candidates
                # Use a combined stratum seed for stability
                stratum_seed = request.seed + hash(f"{dom}_{diff}") % 1000000
                stratum_rng = random.Random(stratum_seed)
                stratum_rng.shuffle(available_in_stratum)

                if available_count >= diff_quota:
                    # Sufficient data: select exact quota (undersampling excess)
                    chosen = available_in_stratum[:diff_quota]
                    for r in chosen:
                        rec_copy = copy.deepcopy(r)
                        rec_copy.metadata.mixing = {
                            "strategy": request.strategy,
                            "seed": request.seed,
                            "batch_id": request.get_effective_batch_id(),
                            "oversampled": False,
                        }
                        selected_records.append(rec_copy)
                else:
                    # Shortage in stratum
                    if request.allow_oversampling and available_count > 0:
                        # Controlled Oversampling
                        chosen = list(available_in_stratum)
                        for r in chosen:
                            rec_copy = copy.deepcopy(r)
                            rec_copy.metadata.mixing = {
                                "strategy": request.strategy,
                                "seed": request.seed,
                                "batch_id": request.get_effective_batch_id(),
                                "oversampled": False,
                            }
                            selected_records.append(rec_copy)

                        needed = diff_quota - available_count
                        for k in range(needed):
                            # Deterministic cycle over available
                            template_rec = available_in_stratum[k % available_count]
                            rec_copy = copy.deepcopy(template_rec)
                            rec_copy.metadata.mixing = {
                                "strategy": request.strategy,
                                "seed": request.seed,
                                "batch_id": request.get_effective_batch_id(),
                                "oversampled": True,
                                "copy_index": k + 1,
                            }
                            selected_records.append(rec_copy)
                            oversampled_records_count += 1
                            unique_oversampled_set.add(template_rec.canonical_content_hash())
                    else:
                        # No oversampling: take all available and record shortage
                        for r in available_in_stratum:
                            rec_copy = copy.deepcopy(r)
                            rec_copy.metadata.mixing = {
                                "strategy": request.strategy,
                                "seed": request.seed,
                                "batch_id": request.get_effective_batch_id(),
                                "oversampled": False,
                            }
                            selected_records.append(rec_copy)

                        shortages.append(
                            ShortageDetail(
                                category=f"{dom}:{diff}",
                                dimension="difficulty",
                                requested=diff_quota,
                                available=available_count,
                                selected=available_count,
                                shortage=diff_quota - available_count,
                            )
                        )

        # 4. Final deterministic shuffle of selected records
        rng.shuffle(selected_records)

        oversampling_info = None
        if oversampled_records_count > 0:
            oversampling_info = OversamplingDetail(
                oversampled_records=oversampled_records_count,
                unique_records_oversampled=len(unique_oversampled_set),
                oversampling_ratio=round(oversampled_records_count / len(selected_records), 4) if selected_records else 0.0,
            )

        discarded_count = max(0, total_candidates_count - len(selected_records))
        return selected_records, shortages, oversampling_info, discarded_count

    @staticmethod
    def _allocate_quotas(total: int, weights: Dict[str, float]) -> Dict[str, int]:
        """Hare-Niemeyer / Largest Remainder algorithm for exact integer quota distribution."""
        if total <= 0 or not weights:
            return {k: 0 for k in weights}

        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            raise ValueError(f"Weight sum must be positive, got {weight_sum}")

        exact_quotas = {k: (w / weight_sum) * total for k, w in weights.items()}
        int_quotas = {k: int(math.floor(q)) for k, q in exact_quotas.items()}
        remainders = {k: exact_quotas[k] - int_quotas[k] for k in weights}

        deficit = total - sum(int_quotas.values())
        # Distribute remainder to largest remainder keys deterministically
        sorted_keys = sorted(remainders.keys(), key=lambda k: (-remainders[k], k))
        for i in range(deficit):
            int_quotas[sorted_keys[i % len(sorted_keys)]] += 1

        return int_quotas


class BalancedMixingStrategy(BaseMixingStrategy):
    """
    Balanced Mixing Strategy.
    Aims to maximize diversity and equalize representation across available domains and classes,
    reducing distribution skew when data availability is heterogeneous.
    """

    def select(
        self,
        candidates: List[DatasetRecord],
        request: MixingRequest,
        domain_targets: Dict[str, float],
        difficulty_targets: Dict[str, float],
        task_targets: Optional[Dict[str, float]] = None,
        source_targets: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[DatasetRecord], List[ShortageDetail], Optional[OversamplingDetail], int]:
        rng = random.Random(request.seed)
        target_n = request.target_count

        # Group candidates by domain
        domain_pools: Dict[str, List[DatasetRecord]] = defaultdict(list)
        for r in candidates:
            domain_pools[r.metadata.domain].append(r)

        # Equal share allocation across recognized domains
        all_domains = sorted(domain_targets.keys())
        num_domains = len(all_domains)
        base_quota = target_n // num_domains
        remainder = target_n % num_domains

        quotas: Dict[str, int] = {}
        for i, dom in enumerate(all_domains):
            quotas[dom] = base_quota + (1 if i < remainder else 0)

        selected_records: List[DatasetRecord] = []
        shortages: List[ShortageDetail] = []
        oversampled_records_count = 0
        unique_oversampled_set: Set[str] = set()

        for dom, quota in quotas.items():
            pool = list(domain_pools[dom])
            avail = len(pool)

            dom_seed = request.seed + hash(dom) % 1000000
            dom_rng = random.Random(dom_seed)
            dom_rng.shuffle(pool)

            if avail >= quota:
                chosen = pool[:quota]
                for r in chosen:
                    rec_copy = copy.deepcopy(r)
                    rec_copy.metadata.mixing = {
                        "strategy": "balanced",
                        "seed": request.seed,
                        "batch_id": request.get_effective_batch_id(),
                        "oversampled": False,
                    }
                    selected_records.append(rec_copy)
            else:
                if request.allow_oversampling and avail > 0:
                    for r in pool:
                        rec_copy = copy.deepcopy(r)
                        rec_copy.metadata.mixing = {
                            "strategy": "balanced",
                            "seed": request.seed,
                            "batch_id": request.get_effective_batch_id(),
                            "oversampled": False,
                        }
                        selected_records.append(rec_copy)

                    needed = quota - avail
                    for k in range(needed):
                        template_rec = pool[k % avail]
                        rec_copy = copy.deepcopy(template_rec)
                        rec_copy.metadata.mixing = {
                            "strategy": "balanced",
                            "seed": request.seed,
                            "batch_id": request.get_effective_batch_id(),
                            "oversampled": True,
                            "copy_index": k + 1,
                        }
                        selected_records.append(rec_copy)
                        oversampled_records_count += 1
                        unique_oversampled_set.add(template_rec.canonical_content_hash())
                else:
                    for r in pool:
                        rec_copy = copy.deepcopy(r)
                        rec_copy.metadata.mixing = {
                            "strategy": "balanced",
                            "seed": request.seed,
                            "batch_id": request.get_effective_batch_id(),
                            "oversampled": False,
                        }
                        selected_records.append(rec_copy)

                    shortages.append(
                        ShortageDetail(
                            category=dom,
                            dimension="domain",
                            requested=quota,
                            available=avail,
                            selected=avail,
                            shortage=quota - avail,
                        )
                    )

        rng.shuffle(selected_records)

        oversampling_info = None
        if oversampled_records_count > 0:
            oversampling_info = OversamplingDetail(
                oversampled_records=oversampled_records_count,
                unique_records_oversampled=len(unique_oversampled_set),
                oversampling_ratio=round(oversampled_records_count / len(selected_records), 4) if selected_records else 0.0,
            )

        discarded_count = max(0, len(candidates) - len(selected_records))
        return selected_records, shortages, oversampling_info, discarded_count


# ============================================================================
# 3. DATASET MIXER ENGINE
# ============================================================================

class DatasetMixer:
    """
    Main Dataset Mixing & Balancing Engine.
    Orchestrates ingestion across source pools, enforces distribution targets,
    applies selected mixing strategies, preserves provenance, and outputs rich reports.
    """

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/dataset.yaml",
        deduplicate_before_mix: bool = False,
    ):
        self.config_path = Path(config_path).resolve()
        self.config = self._load_config()
        self.deduplicate_before_mix = deduplicate_before_mix

        # 1. Authoritative Domain & Difficulty Targets
        self.domain_targets: Dict[str, float] = self.config.get("domain_targets", {})
        self.difficulty_targets: Dict[str, float] = self.config.get("difficulty", {}).get("targets", {
            "beginner": 0.25,
            "intermediate": 0.40,
            "advanced": 0.25,
            "expert": 0.10,
        })
        self.mixing_cfg: Dict[str, Any] = self.config.get("mixing", {})

        # 2. Strict Target Validations
        self._validate_targets()

        # 3. Strategy Registry
        self.strategies: Dict[str, BaseMixingStrategy] = {
            "proportional": ProportionalMixingStrategy(),
            "balanced": BalancedMixingStrategy(),
        }

        # 4. Helper components
        self.loader = DatasetLoader(continue_on_error=True)
        self.normalizer = DatasetNormalizer()
        self.deduplicator = DatasetDeduplicator(enable_near_dedup=False)

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Dataset config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _validate_targets(self) -> None:
        """Validates that domain and difficulty target distributions sum to 1.0."""
        if not self.domain_targets:
            raise ValueError("domain_targets configuration is missing or empty in dataset config.")
        
        dom_sum = sum(self.domain_targets.values())
        if abs(dom_sum - 1.0) > 1e-4:
            raise ValueError(
                f"Authoritative domain_targets must sum to 1.000, got sum = {dom_sum:.5f}. "
                "Check configs/dataset.yaml."
            )

        diff_sum = sum(self.difficulty_targets.values())
        if abs(diff_sum - 1.0) > 1e-4:
            raise ValueError(
                f"Difficulty targets must sum to 1.000, got sum = {diff_sum:.5f}. "
                "Check configs/dataset.yaml."
            )

    def register_strategy(self, name: str, strategy: BaseMixingStrategy) -> None:
        """Registers a custom or extended mixing strategy."""
        self.strategies[name.strip().lower()] = strategy

    def ingest_sources(
        self, sources: Sequence[Union[str, Path, DatasetRecord, List[DatasetRecord]]]
    ) -> List[DatasetRecord]:
        """
        Ingests and normalizes records from file paths, directories, or pre-loaded record lists.
        """
        candidates: List[DatasetRecord] = []

        for src in sources:
            if isinstance(src, list):
                for item in src:
                    if isinstance(item, DatasetRecord):
                        candidates.append(item)
                    elif isinstance(item, dict):
                        candidates.append(DatasetRecord.from_dict(item))
            elif isinstance(src, DatasetRecord):
                candidates.append(src)
            elif isinstance(src, (str, Path)):
                p = Path(src).resolve()
                if not p.exists():
                    raise FileNotFoundError(f"Source dataset path does not exist: {p}")
                raw_records, _ = self.loader.load_path(p)
                for r in raw_records:
                    norm = self.normalizer.normalize_record(r)
                    payload = {
                        "messages": norm.get("messages", []),
                        "metadata": norm.get("metadata", {}),
                    }
                    candidates.append(DatasetRecord.from_dict(payload))
            else:
                raise TypeError(f"Unsupported source type: {type(src)}")

        if self.deduplicate_before_mix and candidates:
            candidates, _ = self.deduplicator.deduplicate(candidates)

        return candidates

    def mix(
        self,
        request: MixingRequest,
        candidate_records: Optional[List[DatasetRecord]] = None,
    ) -> MixingResult:
        """
        Executes deterministic dataset mixing according to request specifications.
        """
        # 1. Ingest candidate pool
        if candidate_records is not None:
            candidates = list(candidate_records)
        elif request.input_sources:
            candidates = self.ingest_sources(request.input_sources)
        else:
            candidates = []

        batch_id = request.get_effective_batch_id()
        strategy_name = request.strategy.strip().lower()

        if strategy_name not in self.strategies:
            raise ValueError(
                f"Mixing strategy '{request.strategy}' is not registered. "
                f"Available: {list(self.strategies.keys())}"
            )

        strategy = self.strategies[strategy_name]

        # 2. Resolve target dictionaries
        effective_domain_targets = request.domain_targets or self.domain_targets
        effective_diff_targets = request.difficulty_targets or self.difficulty_targets

        # Validate domain targets sum
        d_sum = sum(effective_domain_targets.values())
        if abs(d_sum - 1.0) > 1e-4:
            raise ValueError(f"Domain target weights must sum to 1.000, got {d_sum:.4f}")

        # Validate difficulty targets sum
        diff_sum = sum(effective_diff_targets.values())
        if abs(diff_sum - 1.0) > 1e-4:
            raise ValueError(f"Difficulty target weights must sum to 1.000, got {diff_sum:.4f}")

        # 3. Execute Strategy Selection
        selected, shortages, oversampling, discarded_count = strategy.select(
            candidates=candidates,
            request=request,
            domain_targets=effective_domain_targets,
            difficulty_targets=effective_diff_targets,
            task_targets=request.task_targets,
            source_targets=request.source_targets,
        )

        # 4. Compute Comprehensive Distributions & Deviations
        domain_dist = self._compute_distribution(
            selected,
            dimension="domain",
            key_fn=lambda r: r.metadata.domain,
            targets=effective_domain_targets,
        )

        diff_dist = self._compute_distribution(
            selected,
            dimension="difficulty",
            key_fn=lambda r: r.metadata.difficulty,
            targets=effective_diff_targets,
        )

        task_dist = self._compute_distribution(
            selected,
            dimension="task_type",
            key_fn=lambda r: r.metadata.task_type,
            targets=request.task_targets or {},
        )

        source_dist = self._compute_distribution(
            selected,
            dimension="source_type",
            key_fn=lambda r: r.metadata.source_type,
            targets=request.source_targets or {},
        )

        # 5. Assemble Result
        return MixingResult(
            records=selected,
            requested_count=request.target_count,
            selected_count=len(selected),
            total_candidates=len(candidates),
            strategy=strategy_name,
            seed=request.seed,
            batch_id=batch_id,
            domain_distribution=domain_dist,
            difficulty_distribution=diff_dist,
            task_distribution=task_dist,
            source_distribution=source_dist,
            shortages=shortages,
            oversampling=oversampling,
            discarded_count=discarded_count,
            errors=[],
        )

    @staticmethod
    def _compute_distribution(
        records: List[DatasetRecord],
        dimension: str,
        key_fn: Any,
        targets: Dict[str, float],
    ) -> DistributionReport:
        total = len(records)
        counts: Dict[str, int] = defaultdict(int)

        # Initialize known targets
        for k in targets.keys():
            counts[k] = 0

        for r in records:
            k = key_fn(r)
            counts[k] += 1

        percentages: Dict[str, float] = {}
        deviations: Dict[str, float] = {}
        target_pcts: Dict[str, float] = {}

        for k, count in counts.items():
            pct = round((count / total) * 100, 2) if total > 0 else 0.0
            percentages[k] = pct

            t_val = targets.get(k, 0.0)
            t_pct = round(t_val * 100, 2)
            target_pcts[k] = t_pct

            if t_val > 0:
                deviations[k] = round(pct - t_pct, 2)
            else:
                deviations[k] = 0.0

        return DistributionReport(
            dimension=dimension,
            counts=dict(counts),
            percentages=percentages,
            targets=target_pcts,
            deviations=deviations,
        )
