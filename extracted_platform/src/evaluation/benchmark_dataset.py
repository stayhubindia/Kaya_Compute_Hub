"""
Benchmark Dataset Manager, Manifest & Statistics Engine (Phase 4.5).
Manages serialization, SHA-256 integrity verification, token distributions,
and immutable lifecycle management for benchmark-v1.0.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from src.dataset.schema import Role
from src.evaluation.benchmark_cases import BenchmarkCase
from src.training.utils import compute_file_sha256


class BenchmarkManifest(BaseModel):
    """Manifest tracking benchmark metadata, distributions, integrity, and locked lifecycle state."""
    benchmark_version: str = "benchmark-v1.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    case_count: int = 0
    lifecycle_status: str = "FROZEN"  # 'DRAFT', 'VALIDATING', 'RELEASED', 'FROZEN'
    dataset_versions_excluded: List[str] = Field(default_factory=lambda: ["dataset-v1.0"])
    domain_distribution: Dict[str, int] = Field(default_factory=dict)
    difficulty_distribution: Dict[str, int] = Field(default_factory=dict)
    task_distribution: Dict[str, int] = Field(default_factory=dict)
    evaluation_type_distribution: Dict[str, int] = Field(default_factory=dict)
    turn_type_distribution: Dict[str, int] = Field(default_factory=dict)
    config_hash: str = ""
    benchmark_sha256: str = ""
    generation_config: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def save(self, path: Union[str, Path]) -> Path:
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> BenchmarkManifest:
        p = Path(path).resolve()
        with open(p, "r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


class TokenDistribution(BaseModel):
    """Token length statistical percentiles and extrema."""
    mean: float = 0.0
    min: int = 0
    max: int = 0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class BenchmarkStatistics(BaseModel):
    """Multi-dimensional benchmark statistics and sequence length telemetry."""
    benchmark_version: str = "benchmark-v1.0"
    total_cases: int = 0
    prompt_tokens: TokenDistribution = Field(default_factory=TokenDistribution)
    reference_tokens: TokenDistribution = Field(default_factory=TokenDistribution)
    domains: Dict[str, int] = Field(default_factory=dict)
    difficulties: Dict[str, int] = Field(default_factory=dict)
    task_types: Dict[str, int] = Field(default_factory=dict)
    evaluation_types: Dict[str, int] = Field(default_factory=dict)
    turn_types: Dict[str, int] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def save(self, path: Union[str, Path]) -> Path:
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> BenchmarkStatistics:
        p = Path(path).resolve()
        with open(p, "r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


class BenchmarkDatasetManager:
    """Orchestrates serialization, verification, statistics calculation, and immutability checks."""

    @staticmethod
    def _approximate_tokens(text: str) -> int:
        """Heuristic word/token estimation (approx 1.3 tokens per whitespace-separated word)."""
        words = text.strip().split()
        return max(1, int(math.ceil(len(words) * 1.3))) if words else 0

    @classmethod
    def compute_statistics(cls, cases: List[BenchmarkCase]) -> BenchmarkStatistics:
        """Compute full statistical distributions across benchmark cases."""
        if not cases:
            return BenchmarkStatistics()

        prompt_lens: List[int] = []
        ref_lens: List[int] = []
        domains: Dict[str, int] = {}
        difficulties: Dict[str, int] = {}
        task_types: Dict[str, int] = {}
        eval_types: Dict[str, int] = {}
        turn_types: Dict[str, int] = {"single_turn": 0, "multi_turn": 0}

        for c in cases:
            # Prompt token length (all user and system prompt turns)
            prompts = [m.content for m in c.get_prompt_messages()]
            prompt_str = " ".join(prompts)
            p_tok = cls._approximate_tokens(prompt_str)
            r_tok = cls._approximate_tokens(c.reference_answer)

            prompt_lens.append(p_tok)
            ref_lens.append(r_tok)

            # Categorical distributions
            domains[c.domain] = domains.get(c.domain, 0) + 1
            difficulties[c.difficulty] = difficulties.get(c.difficulty, 0) + 1
            task_types[c.task_type] = task_types.get(c.task_type, 0) + 1
            eval_types[c.evaluation_type] = eval_types.get(c.evaluation_type, 0) + 1

            # Turn classification
            user_msg_count = sum(1 for m in c.messages if m.role == Role.USER)
            if user_msg_count > 1:
                turn_types["multi_turn"] += 1
            else:
                turn_types["single_turn"] += 1

        def _calc_dist(lens: List[int]) -> TokenDistribution:
            sorted_lens = sorted(lens)
            n = len(sorted_lens)
            mean_v = round(sum(sorted_lens) / n, 2)
            p50_idx = int(n * 0.50)
            p90_idx = min(int(n * 0.90), n - 1)
            p95_idx = min(int(n * 0.95), n - 1)
            p99_idx = min(int(n * 0.99), n - 1)
            return TokenDistribution(
                mean=mean_v,
                min=sorted_lens[0],
                max=sorted_lens[-1],
                p50=float(sorted_lens[p50_idx]),
                p90=float(sorted_lens[p90_idx]),
                p95=float(sorted_lens[p95_idx]),
                p99=float(sorted_lens[p99_idx]),
            )

        return BenchmarkStatistics(
            total_cases=len(cases),
            prompt_tokens=_calc_dist(prompt_lens),
            reference_tokens=_calc_dist(ref_lens),
            domains=domains,
            difficulties=difficulties,
            task_types=task_types,
            evaluation_types=eval_types,
            turn_types=turn_types,
        )

    @classmethod
    def generate_readme(cls, manifest: BenchmarkManifest, stats: BenchmarkStatistics) -> str:
        """Generate human-readable markdown documentation for the benchmark suite."""
        lines = [
            f"# Independent Evaluation Benchmark Suite — `{manifest.benchmark_version}`",
            "",
            "## 1. Overview",
            "",
            f"- **Version:** `{manifest.benchmark_version}`",
            f"- **Lifecycle Status:** `{manifest.lifecycle_status}` (Immutable)",
            f"- **Total Benchmark Cases:** `{manifest.case_count}`",
            f"- **Benchmark SHA-256:** `{manifest.benchmark_sha256}`",
            f"- **Excluded Datasets:** `{', '.join(manifest.dataset_versions_excluded)}`",
            f"- **Created Timestamp:** `{manifest.created_at}`",
            "",
            "## 2. Sequence Length Distribution",
            "",
            "| Metric | Prompt Tokens | Reference Tokens |",
            "| :--- | :--- | :--- |",
            f"| `Mean` | {stats.prompt_tokens.mean:.1f} | {stats.reference_tokens.mean:.1f} |",
            f"| `Min / Max` | {stats.prompt_tokens.min} / {stats.prompt_tokens.max} | {stats.reference_tokens.min} / {stats.reference_tokens.max} |",
            f"| `P50 (Median)` | {stats.prompt_tokens.p50:.1f} | {stats.reference_tokens.p50:.1f} |",
            f"| `P90` | {stats.prompt_tokens.p90:.1f} | {stats.reference_tokens.p90:.1f} |",
            f"| `P95` | {stats.prompt_tokens.p95:.1f} | {stats.reference_tokens.p95:.1f} |",
            f"| `P99` | {stats.prompt_tokens.p99:.1f} | {stats.reference_tokens.p99:.1f} |",
            "",
            "## 3. Domain Distribution",
            "",
            "| Domain | Case Count | Percentage |",
            "| :--- | :--- | :--- |",
        ]

        total = stats.total_cases if stats.total_cases > 0 else 1
        for dom, cnt in sorted(stats.domains.items()):
            pct = (cnt / total) * 100
            lines.append(f"| `{dom}` | {cnt} | {pct:.1f}% |")

        lines.extend([
            "",
            "## 4. Difficulty Breakdown",
            "",
            "| Difficulty | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ])
        for diff, cnt in sorted(stats.difficulties.items()):
            pct = (cnt / total) * 100
            lines.append(f"| `{diff}` | {cnt} | {pct:.1f}% |")

        lines.extend([
            "",
            "## 5. Evaluation Types",
            "",
            "| Evaluation Mode | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ])
        for et, cnt in sorted(stats.evaluation_types.items()):
            pct = (cnt / total) * 100
            lines.append(f"| `{et}` | {cnt} | {pct:.1f}% |")

        lines.extend([
            "",
            "## 6. Conversation Turn Distribution",
            "",
            f"- **Single-Turn Cases:** `{stats.turn_types.get('single_turn', 0)}`",
            f"- **Multi-Turn Cases:** `{stats.turn_types.get('multi_turn', 0)}`",
        ])

        return "\n".join(lines)

    @classmethod
    def save_benchmark(
        cls,
        cases: List[BenchmarkCase],
        base_dir: Union[str, Path],
        manifest: Optional[BenchmarkManifest] = None,
        config_hash: str = "",
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, BenchmarkManifest, BenchmarkStatistics]:
        """Save full benchmark bundle (benchmark.jsonl, manifest.json, statistics.json, README.md)."""
        b_dir = Path(base_dir)
        b_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = b_dir / "benchmark.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c.to_dict()) + "\n")

        sha256_hash = compute_file_sha256(jsonl_path)
        stats = cls.compute_statistics(cases)

        if manifest is None:
            manifest = BenchmarkManifest(
                benchmark_version="benchmark-v1.0",
                case_count=len(cases),
                lifecycle_status="FROZEN",
                domain_distribution=stats.domains,
                difficulty_distribution=stats.difficulties,
                task_distribution=stats.task_types,
                evaluation_type_distribution=stats.evaluation_types,
                turn_type_distribution=stats.turn_types,
                config_hash=config_hash,
                benchmark_sha256=sha256_hash,
                generation_config=generation_config or {},
            )
        else:
            manifest.benchmark_sha256 = sha256_hash
            manifest.case_count = len(cases)

        manifest.save(b_dir / "manifest.json")
        stats.save(b_dir / "statistics.json")

        readme_path = b_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(cls.generate_readme(manifest, stats))

        return jsonl_path, manifest, stats

    @classmethod
    def load_benchmark(cls, base_dir: Union[str, Path]) -> Tuple[List[BenchmarkCase], BenchmarkManifest, BenchmarkStatistics]:
        """Load and verify an existing benchmark suite."""
        b_dir = Path(base_dir)
        manifest_path = b_dir / "manifest.json"
        jsonl_path = b_dir / "benchmark.jsonl"
        stats_path = b_dir / "statistics.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {jsonl_path}")

        manifest = BenchmarkManifest.load(manifest_path)
        actual_sha = compute_file_sha256(jsonl_path)

        if manifest.benchmark_sha256 and actual_sha.lower() != manifest.benchmark_sha256.lower():
            raise ValueError(
                f"Benchmark SHA-256 mismatch! Manifest expected {manifest.benchmark_sha256}, got {actual_sha}"
            )

        cases: List[BenchmarkCase] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cases.append(BenchmarkCase.model_validate(json.loads(line)))

        stats = BenchmarkStatistics.load(stats_path) if stats_path.exists() else cls.compute_statistics(cases)
        return cases, manifest, stats
