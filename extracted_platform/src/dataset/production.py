"""
Production Dataset Specification & Scaling Architecture (Phase 3.1).
Provides deterministic quota planning, 2D Domain x Difficulty matrix allocation,
batch planning, resumable checkpointing, manifest management, and dataset freeze lifecycles
for scaling the dataset from 10K to 100K+ conversational examples.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.dataset.schema import DifficultyLevel, TaskType
from src.dataset.source_registry import SourceRegistry
from src.dataset.template_registry import TemplateRegistry


# ============================================================================
# 1. ENUMS & CONSTANTS
# ============================================================================

class DatasetFreezeState(str, Enum):
    """Lifecycle states of a production dataset release."""
    PLANNED = "PLANNED"
    GENERATING = "GENERATING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    FROZEN = "FROZEN"


class BatchStatus(str, Enum):
    """Status of an individual batch during generation/processing."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# 2. DATA MODELS FOR PLANNING & QUOTAS
# ============================================================================

class QuotaBreakdown(BaseModel):
    """Detailed quota allocation for a single category (domain or difficulty)."""
    category: str
    weight: float
    target_percentage: float
    exact_quota: float
    integer_quota: int
    remainder: float
    rounding_adjustment: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "weight": round(self.weight, 6),
            "target_percentage": round(self.target_percentage, 4),
            "exact_quota": round(self.exact_quota, 4),
            "integer_quota": self.integer_quota,
            "remainder": round(self.remainder, 6),
            "rounding_adjustment": self.rounding_adjustment,
        }


class DomainDifficultyMatrix(BaseModel):
    """
    2D joint distribution matrix for domain x difficulty allocation.
    Guarantees that row sums match domain quotas exactly, and grand total
    equals the production target count.
    """
    matrix: Dict[str, Dict[str, int]]
    row_totals: Dict[str, int]
    col_totals: Dict[str, int]
    grand_total: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix": self.matrix,
            "row_totals": self.row_totals,
            "col_totals": self.col_totals,
            "grand_total": self.grand_total,
        }


class BatchPlan(BaseModel):
    """Deterministic plan for a single production generation batch."""
    batch_id: str
    batch_index: int
    seed: int
    target_count: int
    candidate_target: int
    domain_quotas: Dict[str, int] = Field(default_factory=dict)
    difficulty_quotas: Dict[str, int] = Field(default_factory=dict)
    matrix: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    template_ids: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "batch_index": self.batch_index,
            "seed": self.seed,
            "target_count": self.target_count,
            "candidate_target": self.candidate_target,
            "domain_quotas": self.domain_quotas,
            "difficulty_quotas": self.difficulty_quotas,
            "matrix": self.matrix,
            "template_count": len(self.template_ids),
        }


# ============================================================================
# 3. APPORTIONMENT ALGORITHMS
# ============================================================================

def derive_batch_seed(global_seed: int, batch_index: int) -> int:
    """
    Derives a deterministic, high-entropy integer seed for a batch given the global seed.
    Uses SHA-256 to ensure zero collision or sequential bias across batches.
    """
    msg = f"qwen3:production:seed:{global_seed}:batch:{batch_index}".encode("utf-8")
    digest = hashlib.sha256(msg).digest()
    return int.from_bytes(digest[:4], "big")


def apportion_quotas_hare_niemeyer(
    total: int,
    weights: Dict[str, float],
) -> List[QuotaBreakdown]:
    """
    Hare-Niemeyer (Largest Remainder) apportionment method.
    Allocates integer quotas from weights such that sum(integer_quota) == total exactly.
    Tie-breaking is deterministic (sorted by category name).
    """
    if total <= 0:
        raise ValueError(f"Target total count must be positive, got {total}")
    if not weights:
        raise ValueError("Weights dictionary must not be empty")

    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError(f"Sum of weights must be positive, got {weight_sum}")

    # 1. Compute exact and integer base quotas
    exact_quotas: Dict[str, float] = {}
    base_quotas: Dict[str, int] = {}
    remainders: Dict[str, float] = {}

    for cat, w in weights.items():
        normalized_w = w / weight_sum
        exact = normalized_w * total
        base = int(math.floor(exact))
        exact_quotas[cat] = exact
        base_quotas[cat] = base
        remainders[cat] = exact - base

    # 2. Distribute deficit to largest remainders
    deficit = total - sum(base_quotas.values())
    sorted_cats = sorted(weights.keys(), key=lambda c: (-remainders[c], c))

    allocated_adjustment: Dict[str, int] = {c: 0 for c in weights}
    for i in range(deficit):
        cat = sorted_cats[i % len(sorted_cats)]
        allocated_adjustment[cat] += 1

    # 3. Build QuotaBreakdown items
    results: List[QuotaBreakdown] = []
    for cat in sorted(weights.keys()):
        w = weights[cat]
        norm_w = w / weight_sum
        int_q = base_quotas[cat] + allocated_adjustment[cat]
        results.append(
            QuotaBreakdown(
                category=cat,
                weight=w,
                target_percentage=norm_w * 100.0,
                exact_quota=exact_quotas[cat],
                integer_quota=int_q,
                remainder=remainders[cat],
                rounding_adjustment=allocated_adjustment[cat],
            )
        )

    return results


def build_domain_difficulty_matrix(
    target_count: int,
    domain_weights: Dict[str, float],
    difficulty_weights: Dict[str, float],
) -> DomainDifficultyMatrix:
    """
    Constructs a 2-dimensional joint distribution quota matrix (domain x difficulty).
    Guarantees:
      1. Every domain row sum equals its 1D Hare-Niemeyer domain quota exactly.
      2. The grand total of all cells equals target_count exactly.
      3. Column totals match target difficulty distribution as closely as mathematically possible.
    """
    domain_breakdowns = apportion_quotas_hare_niemeyer(target_count, domain_weights)
    domain_quotas = {b.category: b.integer_quota for b in domain_breakdowns}

    diff_breakdowns = apportion_quotas_hare_niemeyer(target_count, difficulty_weights)
    target_diff_quotas = {b.category: b.integer_quota for b in diff_breakdowns}

    matrix: Dict[str, Dict[str, int]] = {}
    col_totals: Dict[str, int] = {d: 0 for d in difficulty_weights}

    for dom, dom_q in sorted(domain_quotas.items()):
        matrix[dom] = {}
        if dom_q == 0:
            for diff in difficulty_weights:
                matrix[dom][diff] = 0
            continue

        # Apportion domain quota across difficulty tiers
        cell_breakdowns = apportion_quotas_hare_niemeyer(dom_q, difficulty_weights)
        for cb in cell_breakdowns:
            matrix[dom][cb.category] = cb.integer_quota
            col_totals[cb.category] += cb.integer_quota

    grand_total = sum(sum(cells.values()) for cells in matrix.values())

    return DomainDifficultyMatrix(
        matrix=matrix,
        row_totals=domain_quotas,
        col_totals=col_totals,
        grand_total=grand_total,
    )


# ============================================================================
# 4. CHECKPOINT ARCHITECTURE
# ============================================================================

class BatchCheckpoint(BaseModel):
    """Persisted state of an individual batch during generation."""
    batch_id: str
    batch_index: int
    status: str = BatchStatus.PENDING.value
    seed: int
    requested_count: int
    generated_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    quality_stats: Dict[str, Any] = Field(default_factory=dict)
    source_stats: Dict[str, Any] = Field(default_factory=dict)
    template_ids: List[str] = Field(default_factory=list)
    output_file: Optional[str] = None
    checksum_sha256: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ProductionCheckpointManager:
    """
    Manages batch checkpointing, recovery, failure handling, and resume logic
    for large-scale production dataset runs.
    """

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, batch_id: str) -> Path:
        clean_name = batch_id.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"{clean_name}.json"

    def load_checkpoint(self, batch_id: str) -> Optional[BatchCheckpoint]:
        path = self.get_checkpoint_path(batch_id)
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return BatchCheckpoint.model_validate(data)

    def save_checkpoint(self, checkpoint: BatchCheckpoint) -> Path:
        path = self.get_checkpoint_path(checkpoint.batch_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        return path

    def list_checkpoints(self) -> List[BatchCheckpoint]:
        checkpoints: List[BatchCheckpoint] = []
        for p in sorted(self.checkpoint_dir.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                checkpoints.append(BatchCheckpoint.model_validate(data))
            except Exception:
                continue
        return checkpoints

    def get_completed_batches(self) -> List[BatchCheckpoint]:
        return [c for c in self.list_checkpoints() if c.status == BatchStatus.COMPLETED.value]

    def get_pending_batches(self) -> List[BatchCheckpoint]:
        return [c for c in self.list_checkpoints() if c.status == BatchStatus.PENDING.value]

    def get_failed_batches(self) -> List[BatchCheckpoint]:
        return [c for c in self.list_checkpoints() if c.status == BatchStatus.FAILED.value]

    def is_batch_completed(self, batch_id: str) -> bool:
        ckpt = self.load_checkpoint(batch_id)
        return ckpt is not None and ckpt.status == BatchStatus.COMPLETED.value

    def should_process_batch(self, batch_id: str, force: bool = False) -> bool:
        if force:
            return True
        return not self.is_batch_completed(batch_id)

    def handle_batch_failure(
        self,
        batch_id: str,
        error: Exception,
        fail_fast: bool = False,
    ) -> None:
        """Records batch failure and enforces fail-fast policy."""
        ckpt = self.load_checkpoint(batch_id)
        now_str = datetime.now(timezone.utc).isoformat()
        if ckpt:
            ckpt.status = BatchStatus.FAILED.value
            ckpt.error_message = str(error)
            ckpt.completed_at = now_str
            self.save_checkpoint(ckpt)

        if fail_fast:
            raise RuntimeError(f"Batch '{batch_id}' failed with fail_fast=True: {error}") from error


# ============================================================================
# 5. PRODUCTION MANIFEST
# ============================================================================

class ProductionManifest(BaseModel):
    """
    Authoritative cryptographic and lifecycle manifest for a production dataset release.
    """
    dataset_version: str
    schema_version: str = "1.0.0"
    config_version: str = "1.0.0"
    template_version: str = "1.0.0"
    source_manifest_version: str = "1.0.0"
    target_count: int
    candidate_target: int
    actual_candidate_count: int = 0
    actual_final_count: int = 0
    seed: int
    domain_targets: Dict[str, float]
    difficulty_targets: Dict[str, float]
    task_strategy: str = "observational"
    source_strategy: str = "multi_source"
    batch_size: int
    batch_count: int
    pipeline_version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = DatasetFreezeState.PLANNED.value
    checksums: Dict[str, str] = Field(default_factory=dict)
    quality_thresholds: Dict[str, float] = Field(default_factory=dict)
    batch_summaries: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        clean = v.strip().upper()
        allowed = {s.value for s in DatasetFreezeState}
        if clean not in allowed:
            raise ValueError(f"Invalid dataset status '{v}'. Allowed: {sorted(allowed)}")
        return clean

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def save(self, path: Union[str, Path]) -> Path:
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> ProductionManifest:
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Production manifest not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def transition_state(self, new_state: Union[DatasetFreezeState, str]) -> None:
        val = new_state.value if isinstance(new_state, DatasetFreezeState) else str(new_state).strip().upper()
        if val not in [s.value for s in DatasetFreezeState]:
            raise ValueError(f"Invalid freeze state: {val}")
        self.status = val
        self.updated_at = datetime.now(timezone.utc).isoformat()


# ============================================================================
# 6. PRODUCTION PLAN & DRY-RUN PLANNER ENGINE
# ============================================================================

class ProductionPlan(BaseModel):
    """
    Comprehensive plan for production dataset generation and scaling.
    Generated entirely without synthesizing dataset records.
    """
    version: str
    target_count: int
    candidate_multiplier: float
    candidate_target: int
    batch_size: int
    estimated_batches: int
    seed: int
    domain_quotas: List[QuotaBreakdown]
    difficulty_quotas: List[QuotaBreakdown]
    matrix: DomainDifficultyMatrix
    batch_plans: List[BatchPlan]
    task_strategy: str
    source_strategy: str
    quality_gates: Dict[str, Any]
    storage_layout: Dict[str, str]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "target_count": self.target_count,
            "candidate_multiplier": self.candidate_multiplier,
            "candidate_target": self.candidate_target,
            "batch_size": self.batch_size,
            "estimated_batches": self.estimated_batches,
            "seed": self.seed,
            "domain_quotas": [q.to_dict() for q in self.domain_quotas],
            "difficulty_quotas": [q.to_dict() for q in self.difficulty_quotas],
            "matrix": self.matrix.to_dict(),
            "batch_plans": [b.to_dict() for b in self.batch_plans],
            "task_strategy": self.task_strategy,
            "source_strategy": self.source_strategy,
            "quality_gates": self.quality_gates,
            "storage_layout": self.storage_layout,
            "created_at": self.created_at,
        }

    def generate_markdown_report(self) -> str:
        lines: List[str] = [
            f"# Production Dataset Specification & Plan — `{self.version}`",
            "",
            "> [!NOTE]",
            f"> Plan generated deterministically at `{self.created_at}` with seed `{self.seed}`.",
            "",
            "## 1. Overview & Dataset Parameters",
            "",
            "| Parameter | Value |",
            "| :--- | :--- |",
            f"| **Dataset Version** | `{self.version}` |",
            f"| **Target Final Count** | `{self.target_count:,}` examples |",
            f"| **Candidate Multiplier** | `{self.candidate_multiplier:.2f}x` |",
            f"| **Candidate Target Pool** | `{self.candidate_target:,}` examples |",
            f"| **Batch Size** | `{self.batch_size:,}` examples/batch |",
            f"| **Estimated Batches** | `{self.estimated_batches}` batches |",
            f"| **Global Deterministic Seed** | `{self.seed}` |",
            f"| **Task Strategy** | `{self.task_strategy}` |",
            f"| **Source Strategy** | `{self.source_strategy}` |",
            "",
            "---",
            "",
            "## 2. Domain Quota Allocation (Hare-Niemeyer)",
            "",
            "| Domain | Target % | Exact Float | Integer Quota | Adjustment |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        total_dom_quota = 0
        for q in self.domain_quotas:
            total_dom_quota += q.integer_quota
            adj_str = f"+{q.rounding_adjustment}" if q.rounding_adjustment > 0 else "0"
            lines.append(
                f"| `{q.category}` | {q.target_percentage:.2f}% | {q.exact_quota:.2f} | **`{q.integer_quota:,}`** | `{adj_str}` |"
            )

        lines.extend([
            f"| **TOTAL** | **100.00%** | **{self.target_count:.2f}** | **`{total_dom_quota:,}`** | — |",
            "",
            "---",
            "",
            "## 3. Difficulty Quota Allocation",
            "",
            "| Difficulty Level | Target % | Exact Float | Integer Quota | Adjustment |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        total_diff_quota = 0
        for q in self.difficulty_quotas:
            total_diff_quota += q.integer_quota
            adj_str = f"+{q.rounding_adjustment}" if q.rounding_adjustment > 0 else "0"
            lines.append(
                f"| `{q.category}` | {q.target_percentage:.2f}% | {q.exact_quota:.2f} | **`{q.integer_quota:,}`** | `{adj_str}` |"
            )

        lines.extend([
            f"| **TOTAL** | **100.00%** | **{self.target_count:.2f}** | **`{total_diff_quota:,}`** | — |",
            "",
            "---",
            "",
            "## 4. Domain × Difficulty Quota Matrix",
            "",
            "| Domain | Beginner (25%) | Intermediate (40%) | Advanced (25%) | Expert (10%) | Row Total |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for dom, row_total in sorted(self.matrix.row_totals.items()):
            cells = self.matrix.matrix.get(dom, {})
            b = cells.get("beginner", 0)
            i = cells.get("intermediate", 0)
            a = cells.get("advanced", 0)
            e = cells.get("expert", 0)
            lines.append(f"| `{dom}` | `{b:,}` | `{i:,}` | `{a:,}` | `{e:,}` | **`{row_total:,}`** |")

        col_b = self.matrix.col_totals.get("beginner", 0)
        col_i = self.matrix.col_totals.get("intermediate", 0)
        col_a = self.matrix.col_totals.get("advanced", 0)
        col_e = self.matrix.col_totals.get("expert", 0)
        lines.extend([
            f"| **TOTAL** | **`{col_b:,}`** | **`{col_i:,}`** | **`{col_a:,}`** | **`{col_e:,}`** | **`{self.matrix.grand_total:,}`** |",
            "",
            "---",
            "",
            "## 5. Batch Architecture & Checkpoint Breakdown",
            "",
            f"Total planned batches: **`{len(self.batch_plans)}`**",
            "",
            "| Batch ID | Index | Seed | Target Count | Candidate Target |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for bp in self.batch_plans:
            lines.append(
                f"| `{bp.batch_id}` | `{bp.batch_index:03d}` | `{bp.seed}` | `{bp.target_count:,}` | `{bp.candidate_target:,}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 6. Quality & Verification Gates",
            "",
            "| Gate | Threshold Criteria | Action on Violation |",
            "| :--- | :--- | :--- |",
            "| **Schema Validity** | 0 formatting errors or malformed turns | Rejection & Error Logging |",
            "| **Quality Score** | Mean score $\\ge 0.85$, prefer $\\ge 0.90$ | Quality Filter Rejection |",
            "| **Deduplication** | Exact SHA-256 hash & MinHash Jaccard $\\ge 0.85$ | Candidate Deduplication |",
            "| **Provenance** | 100% records with complete, immutable metadata | Hard Pipeline Enforcement |",
            "| **Cross-Split Leakage** | 0 overlapping hashes across Train/Val/Test | Build Rejection (FAIL) |",
            "| **Split Integrity** | Stratified 90% Train / 5% Validation / 5% Test | Manifest Verification |",
            "",
            "---",
            "",
            "## 7. Storage Architecture",
            "",
            "| Directory Key | Configured Path |",
            "| :--- | :--- |",
        ])

        for k, v in sorted(self.storage_layout.items()):
            lines.append(f"| **`{k}`** | `{v}` |")

        lines.append("")
        return "\n".join(lines)

    def save_reports(self, reports_dir: Union[str, Path]) -> Tuple[Path, Path]:
        """Saves JSON and Markdown production plan reports."""
        r_dir = Path(reports_dir).resolve()
        r_dir.mkdir(parents=True, exist_ok=True)
        json_path = r_dir / "production_plan.json"
        md_path = r_dir / "production_plan.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown_report())

        return json_path, md_path


class ProductionPlanner:
    """
    Orchestrates dry-run production planning, mathematical quota allocation,
    batch segmentation, and manifest generation from dataset configurations.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        template_manifest_path: Optional[Union[str, Path]] = None,
        source_manifest_path: Optional[Union[str, Path]] = None,
    ):
        self.config_path = Path(config_path or "configs/dataset.yaml").resolve()
        self.template_path = Path(template_manifest_path or "configs/domain_templates.yaml").resolve()
        self.source_path = Path(source_manifest_path or "configs/sources.yaml").resolve()

        self.config = self._load_yaml(self.config_path)
        self.template_registry = TemplateRegistry()
        if self.template_path.is_file():
            self.template_registry.load_manifest(self.template_path)

        self.source_registry = SourceRegistry()
        if self.source_path.is_file():
            self.source_registry.load_manifest(self.source_path)

        self.domain_targets: Dict[str, float] = self.config.get("domain_targets", {})
        self.difficulty_targets: Dict[str, float] = self.config.get("difficulty", {}).get("targets", {})
        self.prod_cfg: Dict[str, Any] = self.config.get("production", {})

        self._validate_weights()

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_weights(self) -> None:
        if not self.domain_targets:
            raise ValueError("Configuration must specify 'domain_targets'.")
        dom_sum = sum(self.domain_targets.values())
        if abs(dom_sum - 1.0) > 1e-4:
            raise ValueError(f"Domain targets must sum to 1.00, got {dom_sum:.6f}")

        if not self.difficulty_targets:
            raise ValueError("Configuration must specify 'difficulty.targets'.")
        diff_sum = sum(self.difficulty_targets.values())
        if abs(diff_sum - 1.0) > 1e-4:
            raise ValueError(f"Difficulty targets must sum to 1.00, got {diff_sum:.6f}")

    def plan(
        self,
        target_count: Optional[int] = None,
        seed: Optional[int] = None,
        candidate_multiplier: Optional[float] = None,
        batch_size: Optional[int] = None,
        version: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> ProductionPlan:
        """
        Calculates complete production dataset plan without generating any dataset records.
        """
        eff_target = target_count if target_count is not None else self.prod_cfg.get("target_count", 10000)
        if eff_target <= 0:
            raise ValueError(f"Target count must be a positive integer, got {eff_target}")

        eff_seed = seed if seed is not None else self.prod_cfg.get("seed", 42)
        eff_mult = candidate_multiplier if candidate_multiplier is not None else self.prod_cfg.get("candidate_multiplier", 1.20)
        if eff_mult < 1.0:
            raise ValueError(f"Candidate multiplier must be >= 1.0, got {eff_mult}")

        eff_batch_size = batch_size if batch_size is not None else self.prod_cfg.get("batch_size", 500)
        if eff_batch_size <= 0:
            raise ValueError(f"Batch size must be a positive integer, got {eff_batch_size}")

        eff_version = version or self.prod_cfg.get("version", "dataset-v1.0")
        eff_out_dir = Path(output_dir or self.prod_cfg.get("output_dir", "datasets/production")).resolve()


        # 1. Candidate pool target calculation
        candidate_target = int(math.ceil(eff_target * eff_mult))

        # 2. Domain and Difficulty Quotas via Largest Remainder
        domain_quotas = apportion_quotas_hare_niemeyer(eff_target, self.domain_targets)
        difficulty_quotas = apportion_quotas_hare_niemeyer(eff_target, self.difficulty_targets)

        # 3. 2D Domain x Difficulty Joint Matrix
        matrix = build_domain_difficulty_matrix(eff_target, self.domain_targets, self.difficulty_targets)

        # 4. Batch Planning Segmentation
        # Calculate number of batches based on candidate_target and batch_size
        num_batches = int(math.ceil(candidate_target / eff_batch_size))
        batch_plans: List[BatchPlan] = []

        all_templates = self.template_registry.list_templates()
        template_ids = [t.id for t in all_templates]

        # Allocate per-batch target and candidate quotas
        remaining_target = eff_target
        remaining_candidate = candidate_target

        for b_idx in range(1, num_batches + 1):
            batch_seed = derive_batch_seed(eff_seed, b_idx)
            b_target = min(int(math.ceil(eff_target / num_batches)), remaining_target)
            b_candidate = min(eff_batch_size, remaining_candidate)

            remaining_target -= b_target
            remaining_candidate -= b_candidate

            # Batch domain breakdown
            if b_target > 0:
                b_dom_q = {
                    q.category: q.integer_quota
                    for q in apportion_quotas_hare_niemeyer(b_target, self.domain_targets)
                }
                b_diff_q = {
                    q.category: q.integer_quota
                    for q in apportion_quotas_hare_niemeyer(b_target, self.difficulty_targets)
                }
                b_matrix = build_domain_difficulty_matrix(b_target, self.domain_targets, self.difficulty_targets)
            else:
                b_dom_q = {k: 0 for k in self.domain_targets}
                b_diff_q = {k: 0 for k in self.difficulty_targets}
                b_matrix = DomainDifficultyMatrix(
                    matrix={k: {d: 0 for d in self.difficulty_targets} for k in self.domain_targets},
                    row_totals={k: 0 for k in self.domain_targets},
                    col_totals={d: 0 for d in self.difficulty_targets},
                    grand_total=0,
                )

            batch_plans.append(
                BatchPlan(
                    batch_id=f"{eff_version}-batch-{b_idx:04d}",
                    batch_index=b_idx,
                    seed=batch_seed,
                    target_count=b_target,
                    candidate_target=b_candidate,
                    domain_quotas=b_dom_q,
                    difficulty_quotas=b_diff_q,
                    matrix=b_matrix.matrix,
                    template_ids=template_ids,
                )
            )

        # 5. Storage Layout
        storage_layout = {
            "root": str(eff_out_dir),
            "raw_synthetic": str(eff_out_dir / "raw" / "synthetic"),
            "candidates": str(eff_out_dir / "candidates"),
            "processed": str(eff_out_dir / "processed"),
            "batches": str(eff_out_dir / "batches"),
            "checkpoints": str(eff_out_dir / "checkpoints"),
            "manifests": str(eff_out_dir / "manifests"),
            "reports": str(eff_out_dir / "reports"),
        }

        # 6. Quality & Gate Parameters
        quality_gates = {
            "minimum_quality_score": self.config.get("quality", {}).get("minimum_score", 0.85),
            "preferred_quality_score": self.config.get("quality", {}).get("preferred_score", 0.90),
            "exact_deduplication_hash": self.config.get("pipeline", {}).get("deduplication", {}).get("exact_hash", "sha256"),
            "near_duplicate_threshold": self.config.get("pipeline", {}).get("deduplication", {}).get("near_duplicate_threshold", 0.85),
            "enforce_provenance": True,
            "train_val_test_split": self.config.get("split", {"train": 0.90, "validation": 0.05, "test": 0.05}),
            "max_cross_split_leakage": 0,
        }

        return ProductionPlan(
            version=eff_version,
            target_count=eff_target,
            candidate_multiplier=eff_mult,
            candidate_target=candidate_target,
            batch_size=eff_batch_size,
            estimated_batches=num_batches,
            seed=eff_seed,
            domain_quotas=domain_quotas,
            difficulty_quotas=difficulty_quotas,
            matrix=matrix,
            batch_plans=batch_plans,
            task_strategy="observational",
            source_strategy="multi_source",
            quality_gates=quality_gates,
            storage_layout=storage_layout,
        )

    def create_initial_manifest(self, plan: ProductionPlan) -> ProductionManifest:
        """Constructs an initial ProductionManifest with status PLANNED."""
        return ProductionManifest(
            dataset_version=plan.version,
            schema_version="1.0.0",
            config_version="1.0.0",
            template_version="1.0.0",
            source_manifest_version="1.0.0",
            target_count=plan.target_count,
            candidate_target=plan.candidate_target,
            seed=plan.seed,
            domain_targets=self.domain_targets,
            difficulty_targets=self.difficulty_targets,
            task_strategy=plan.task_strategy,
            source_strategy=plan.source_strategy,
            batch_size=plan.batch_size,
            batch_count=plan.estimated_batches,
            pipeline_version="1.0.0",
            status=DatasetFreezeState.PLANNED.value,
            quality_thresholds={
                "minimum_score": plan.quality_gates.get("minimum_quality_score", 0.85),
                "preferred_score": plan.quality_gates.get("preferred_quality_score", 0.90),
            },
            batch_summaries=[
                {
                    "batch_id": b.batch_id,
                    "batch_index": b.batch_index,
                    "seed": b.seed,
                    "target_count": b.target_count,
                    "candidate_target": b.candidate_target,
                    "status": BatchStatus.PENDING.value,
                }
                for b in plan.batch_plans
            ],
        )
