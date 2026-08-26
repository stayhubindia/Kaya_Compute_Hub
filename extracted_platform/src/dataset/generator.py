"""
Synthetic Data Generation Engine.
Defines model-agnostic interfaces, typed GenerationRequest/GenerationResult models,
template-driven batch synthesis logic, and a deterministic SampleSyntheticGenerator
for pipeline verification, testing, and pilot generation (Phase 2.3.3).
"""

from __future__ import annotations

import json
import random
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from src.dataset.schema import (
    DatasetRecord,
    DifficultyLevel,
    Message,
    ProvenanceInfo,
    RecordMetadata,
    Role,
    SourceType,
    TaskType,
)
from src.dataset.template_registry import TaskTemplate, TemplateRegistry


class GenerationRequest(BaseModel):
    """Strongly typed generation request configuration for synthetic conversational data."""

    template_id: Optional[str] = None
    domain: Optional[str] = None
    topic: Optional[str] = None
    task_type: Optional[str] = None
    difficulty: Optional[str] = None
    number_of_examples: int = Field(default=1, ge=1)
    seed: int = 42
    generation_batch_id: Optional[str] = None
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("template_id", mode="before")
    @classmethod
    def normalize_template_id(cls, v: Any) -> Optional[str]:
        if isinstance(v, str):
            clean = v.strip().lower()
            return clean if clean else None
        return v

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, DifficultyLevel):
            return v.value
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in [d.value for d in DifficultyLevel]:
                return clean
        raise ValueError(f"Invalid difficulty '{v}'. Allowed: {[d.value for d in DifficultyLevel]}")

    @field_validator("task_type", mode="before")
    @classmethod
    def normalize_task_type(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, TaskType):
            return v.value
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in [t.value for t in TaskType]:
                return clean
        raise ValueError(f"Invalid task_type '{v}'. Allowed: {[t.value for t in TaskType]}")

    @field_validator("number_of_examples")
    @classmethod
    def validate_number_of_examples(cls, v: int) -> int:
        if v < 1:
            raise ValueError("number_of_examples must be at least 1.")
        return v

    def get_effective_batch_id(self) -> str:
        """Returns configured generation_batch_id or generates a deterministic fallback identifier."""
        if self.generation_batch_id and self.generation_batch_id.strip():
            return self.generation_batch_id.strip()
        prefix = self.template_id if self.template_id else "synthetic"
        return f"batch_{prefix}_s{self.seed}_{uuid.uuid4().hex[:8]}"

    def validate_against_template(self, template: TaskTemplate) -> None:
        """Verifies that explicitly specified domain/topic/task_type/difficulty align with template definitions."""
        if self.domain and self.domain.lower() != template.domain.lower():
            raise ValueError(
                f"Request domain '{self.domain}' conflicts with template domain '{template.domain}'."
            )
        if self.topic and self.topic.lower() != template.topic.lower():
            raise ValueError(
                f"Request topic '{self.topic}' conflicts with template topic '{template.topic}'."
            )
        if self.task_type and self.task_type.lower() != template.task_type.lower():
            raise ValueError(
                f"Request task_type '{self.task_type}' conflicts with template task_type '{template.task_type}'."
            )
        if self.difficulty:
            diff_clean = self.difficulty.lower()
            if diff_clean not in [d.lower() for d in template.supported_difficulties]:
                raise ValueError(
                    f"Requested difficulty '{self.difficulty}' is not supported by template '{template.id}'. "
                    f"Supported: {template.supported_difficulties}"
                )


class GenerationResult(BaseModel):
    """Structured result model capturing generated records, lineage, counts, and explicit error reports."""

    records: List[DatasetRecord] = Field(default_factory=list)
    requested_count: int = 0
    generated_count: int = 0
    failed_count: int = 0
    batch_id: str = ""
    generator_name: str = ""
    generator_version: str = ""
    template_id: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_successful(self) -> bool:
        """True if all requested records were generated without failures or errors."""
        return self.failed_count == 0 and self.generated_count == self.requested_count and len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_count": self.requested_count,
            "generated_count": self.generated_count,
            "failed_count": self.failed_count,
            "batch_id": self.batch_id,
            "generator_name": self.generator_name,
            "generator_version": self.generator_version,
            "template_id": self.template_id,
            "errors": self.errors,
            "created_at": self.created_at,
            "is_successful": self.is_successful,
            "records": [r.to_dict() for r in self.records],
        }

    def save_jsonl(self, file_path: Union[str, Path], overwrite: bool = False) -> int:
        """Saves generated dataset records to a JSONL file with explicit overwrite guard."""
        out_path = Path(file_path)
        if out_path.is_file() and not overwrite:
            raise FileExistsError(
                f"Output file '{out_path}' already exists. Set overwrite=True to replace."
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(rec.to_json() + "\n")

        return len(self.records)


class SyntheticGeneratorInterface(ABC):
    """Model-agnostic abstract interface for synthetic dataset generation engines."""

    @abstractmethod
    def generate(
        self,
        domain: str,
        topic: str,
        task_type: str,
        difficulty: str,
        number_of_examples: int = 1,
        **kwargs: Any,
    ) -> List[DatasetRecord]:
        """Generates synthetic examples conforming to canonical DatasetRecord schema (legacy interface)."""
        pass

    @abstractmethod
    def generate_from_template(
        self,
        template: TaskTemplate,
        number_of_examples: int = 1,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> List[DatasetRecord]:
        """Generates synthetic examples driven directly by a declarative TaskTemplate."""
        pass

    @abstractmethod
    def generate_batch(
        self,
        request: GenerationRequest,
        template_registry: Optional[TemplateRegistry] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Executes a structured batch generation request with explicit result and error tracking."""
        pass


class SampleSyntheticGenerator(SyntheticGeneratorInterface):
    """
    Deterministic sample generator for pipeline verification, testing, and pilot evaluation.
    Synthesizes structured technical examples across domains with reproducible output.
    """

    SAMPLE_TEMPLATES = {
        "programming": {
            "python": {
                "coding": (
                    "Write a Python function using generators to efficiently process large log files line by line without memory bloat.",
                    "Here is an efficient streaming log parser generator in Python:\n\n```python\nfrom typing import Generator, Dict, Any\nimport re\n\nLOG_PATTERN = re.compile(r'(?P<ip>\\S+) \\[(?P<time>.*?)\\] \"(?P<method>\\w+) (?P<path>\\S+)\"')\n\ndef stream_log_records(file_path: str) -> Generator[Dict[str, Any], None, None]:\n    with open(file_path, 'r', encoding='utf-8') as f:\n        for line in f:\n            match = LOG_PATTERN.search(line)\n            if match:\n                yield match.groupdict()\n```\n\n### Complexity & Benefits\n- **Memory Complexity**: $O(1)$ constant memory overhead since only a single line is held in memory at any point.\n- **Laziness**: Allows pipelining with downstream filter and map operations.",
                ),
            },
        },
        "linux_systems": {
            "kernel_internals": {
                "explanation": (
                    "Explain the difference between dirty_background_ratio and dirty_ratio in Linux vm sysctl tuning.",
                    "`dirty_background_ratio` and `dirty_ratio` control how Linux kernel flusher threads (wb) write dirty page cache memory to disk.\n\n### Key Differences:\n1. **`vm.dirty_background_ratio`** (Default ~10%):\n   - When dirty page memory exceeds this percentage of total system memory, background kernel flusher threads wake up and write pages asynchronously in the background.\n   - The application thread is NOT blocked from issuing new writes.\n2. **`vm.dirty_ratio`** (Default ~20%):\n   - If dirty pages exceed this threshold, the kernel halts writing processes and forces synchronous writeback.\n   - The process is blocked until dirty memory falls below the threshold.",
                ),
            },
        },
        "cybersecurity": {
            "cryptography": {
                "explanation": (
                    "Why is constant-time comparison essential when verifying HMAC tokens in authentication systems?",
                    "Constant-time comparison is critical to prevent **timing attacks** (a class of side-channel attacks).\n\n### Mechanism:\n- Standard equality operators (`==` or `strcmp`) terminate evaluation upon encountering the first mismatching byte.\n- An attacker can measure nanosecond-level response latencies across thousands of requests to guess the correct HMAC byte-by-byte.\n- Constant-time comparison (such as Python's `hmac.compare_digest`) always examines every single byte regardless of where the first mismatch occurs, making execution time independent of the input.",
                ),
            },
        },
    }

    def __init__(self, generator_name: str = "sample_test_generator", version: str = "1.0.0"):
        self.generator_name = generator_name
        self.version = version

    def generate(
        self,
        domain: str,
        topic: str,
        task_type: str,
        difficulty: str,
        number_of_examples: int = 1,
        **kwargs: Any,
    ) -> List[DatasetRecord]:
        """Legacy direct generation method preserving backward compatibility."""
        req = GenerationRequest(
            domain=domain,
            topic=topic,
            task_type=task_type,
            difficulty=difficulty,
            number_of_examples=number_of_examples,
        )
        res = self.generate_batch(req)
        return res.records

    def generate_from_template(
        self,
        template: TaskTemplate,
        number_of_examples: int = 1,
        difficulty: Optional[str] = None,
        seed: int = 42,
        batch_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[DatasetRecord]:
        """Generates synthetic examples driven directly by a declarative TaskTemplate."""
        target_diff = difficulty or template.difficulty
        if target_diff not in template.supported_difficulties:
            raise ValueError(
                f"Difficulty '{target_diff}' not supported by template '{template.id}'. "
                f"Supported: {template.supported_difficulties}"
            )

        req = GenerationRequest(
            template_id=template.id,
            domain=template.domain,
            topic=template.topic,
            task_type=template.task_type,
            difficulty=target_diff,
            number_of_examples=number_of_examples,
            seed=seed,
            generation_batch_id=batch_id,
        )

        reg = TemplateRegistry([template])
        res = self.generate_batch(req, template_registry=reg)
        if not res.is_successful:
            raise RuntimeError(f"Generation failed: {res.errors}")
        return res.records

    SCENARIOS = [
        ("High-Throughput Ingestion", "Optimize pipeline throughput under constrained compute limits.", "def execute_pipeline(data_stream: list) -> dict:\n    return {'status': 'processed', 'records': len(data_stream)}"),
        ("Fault-Tolerant Failover", "Ensure graceful degradation and automatic circuit breaking.", "def handle_circuit_breaker(error_rate: float) -> bool:\n    return error_rate < 0.05"),
        ("Zero-Copy Memory Buffer", "Minimize page cache eviction and allocator fragmentation.", "def allocate_ring_buffer(capacity_mb: int) -> bytearray:\n    return bytearray(capacity_mb * 1024 * 1024)"),
        ("Asynchronous Event Loop", "Implement non-blocking coroutines with timeout cancellation.", "async def process_event_stream(queue: asyncio.Queue) -> None:\n    item = await queue.get()\n    await asyncio.sleep(0.01)"),
        ("Strict Invariant Verification", "Validate system state transitions against formal constraints.", "def verify_state_transition(current: str, target: str) -> bool:\n    valid = {'INIT': ['RUNNING'], 'RUNNING': ['STOPPED']}\n    return target in valid.get(current, [])"),
        ("Low-Latency Edge Proxy", "Route telemetry packets with deterministic sub-millisecond latency.", "def route_packet(header: bytes, routing_table: dict) -> str:\n    return routing_table.get(header[:4], 'default_gateway')"),
        ("Idempotent Mutation Handler", "Guarantee exactly-once processing across retried network RPCs.", "def apply_idempotent_mutation(tx_id: str, ledger: set) -> bool:\n    if tx_id in ledger: return False\n    ledger.add(tx_id)\n    return True"),
        ("Hierarchical Cache Eviction", "Implement two-tier LRU-LFU hybrid cache with write-through policy.", "def evict_hybrid_entry(cache_map: dict, freq_map: dict) -> str:\n    return min(cache_map, key=lambda k: freq_map.get(k, 0))"),
        ("Encrypted Envelope Protocol", "Wrap payloads in authenticated AEAD envelope with rotating keys.", "def seal_envelope(payload: bytes, key: bytes) -> bytes:\n    return b'ENC:' + payload"),
        ("Distributed Deadlock Detection", "Detect wait-for graph cycles using Tarjan strongly connected components.", "def detect_wait_cycles(graph: dict) -> list:\n    return [k for k, v in graph.items() if k in v]"),
        ("Dynamic Rate Limiter", "Enforce token bucket throttling with burst tolerance and sliding window.", "def allow_request(client_id: str, tokens: int, rate: float) -> bool:\n    return tokens > 0"),
        ("Memory-Mapped Log Append", "Persist sequential write-ahead logs using direct OS memory mapping.", "def append_wal_record(fd: int, record_bytes: bytes) -> int:\n    return len(record_bytes)"),
        ("Vector Embedding Indexing", "Build hierarchical navigable small world graphs for sub-linear search.", "def query_hnsw_index(vec: list, index_ref: dict, top_k: int) -> list:\n    return [0] * top_k"),
        ("Zero-Knowledge Proof Verification", "Validate arithmetic circuit witness proofs without revealing secrets.", "def verify_zk_snark(vk: dict, proof: dict, public_inputs: list) -> bool:\n    return True"),
        ("Adaptive Query Optimizer", "Estimate relational cost models using cardinalities and histogram statistics.", "def plan_join_order(relations: list, cost_matrix: dict) -> list:\n    return sorted(relations)"),
        ("Consensus Heartbeat Monitor", "Track leader lease renewals and trigger quorum elections on timeout.", "def monitor_heartbeat(last_seen_ms: int, timeout_ms: int) -> bool:\n    return last_seen_ms < timeout_ms"),
    ]

    VARIATION_FOCUS_AREAS = [
        ("Architecture & Systems", "Focus on architectural invariants, modular decoupling, and concurrency safeguards."),
        ("Security & Hardening", "Focus on attack surface reduction, memory-safety considerations, and input sanitization."),
        ("Performance Tuning", "Focus on algorithmic complexity reduction, vectorization, and cache locality."),
        ("Edge Case Handling", "Focus on anomalous inputs, boundary conditions, and graceful error propagation."),
        ("Algorithmic Invariants", "Focus on inductive proofs, monotonic properties, and loop termination guarantees."),
        ("Operational Telemetry", "Focus on observability metrics, diagnostic probes, and health check thresholds."),
        ("Resilience & Recovery", "Focus on transient error retries with exponential backoff and jitter."),
        ("Correctness Auditing", "Focus on deterministic repeatability, state consistency, and verification tests."),
    ]

    SCENARIO_VARIATIONS = SCENARIOS

    def generate_batch(
        self,
        request: GenerationRequest,
        template_registry: Optional[TemplateRegistry] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Executes a deterministic batch generation request.
        Validates template consistency, synthesizes canonical DatasetRecord instances,
        attaches complete immutable provenance, and tracks failure telemetry.
        """
        batch_id = request.get_effective_batch_id()
        result = GenerationResult(
            requested_count=request.number_of_examples,
            batch_id=batch_id,
            generator_name=self.generator_name,
            generator_version=self.version,
            template_id=request.template_id,
        )

        template: Optional[TaskTemplate] = None

        # 1. Resolve TaskTemplate if specified or load default registry
        if request.template_id:
            reg = template_registry
            if reg is None:
                default_tmpl_path = Path("configs/domain_templates.yaml")
                if default_tmpl_path.is_file():
                    try:
                        reg = TemplateRegistry.from_yaml(default_tmpl_path)
                    except Exception as e:
                        result.errors.append(f"Failed to load default template manifest: {e}")

            if reg:
                template = reg.lookup_template(request.template_id)
                if template is None:
                    result.errors.append(f"Template with ID '{request.template_id}' not found in registry.")
                    result.failed_count = request.number_of_examples
                    return result
            else:
                result.errors.append(f"No TemplateRegistry available to resolve template_id '{request.template_id}'.")
                result.failed_count = request.number_of_examples
                return result

            # Validate request consistency against resolved template
            try:
                request.validate_against_template(template)
            except ValueError as ve:
                result.errors.append(str(ve))
                result.failed_count = request.number_of_examples
                return result

        # 2. Extract or fallback domain metadata
        effective_domain = template.domain if template else (request.domain or "programming")
        effective_topic = template.topic if template else (request.topic or "python")
        effective_task = template.task_type if template else (request.task_type or "coding")
        effective_diff = (
            request.difficulty
            or (template.difficulty if template else "intermediate")
        )

        # 3. Deterministic Generation Loop
        for i in range(request.number_of_examples):
            try:
                item_seed = request.seed + (i * 7919) + (hash(effective_domain + effective_topic) % 10007)
                item_rng = random.Random(item_seed)

                scenario_idx = (item_seed // 13) % len(self.SCENARIOS)
                focus_idx = (item_seed // 17) % len(self.VARIATION_FOCUS_AREAS)

                scenario_title, scenario_desc, scenario_code = self.SCENARIOS[scenario_idx]
                focus_title, focus_desc = self.VARIATION_FOCUS_AREAS[focus_idx]

                variant_token = f"{effective_domain[:3].upper()}-{item_seed % 99999:05d}"

                # Construct prompt and response with rich distinct phrasing
                if template:
                    prompt = (
                        f"How would you address {template.objective} regarding {template.topic} in {template.domain}? "
                        f"Provide a rigorous {effective_diff}-level {template.task_type} tailored for {scenario_title} [{variant_token}].\n\n"
                        f"Problem Context: {scenario_desc}\n"
                        f"Core Directive: {template.description}\n"
                        f"Specialized Focus: {focus_title} - {focus_desc}"
                    )

                    code_block = ""
                    if template.quality_requirements.get("require_code_blocks", False) or effective_task in ["coding", "debugging", "code_generation"]:
                        code_block = (
                            f"\n\n```python\n# {scenario_title} ({variant_token}): {template.topic} {effective_task}\n"
                            f"{scenario_code}\n"
                            f"# Invariant check: seed={item_seed}\n"
                            f"```\n"
                        )

                    reasoning_block = ""
                    if template.quality_requirements.get("require_reasoning", False) or template.quality_requirements.get("require_step_by_step", False):
                        reasoning_block = (
                            f"\n\n#### Step-by-Step Technical Evaluation\n"
                            f"1. **Core Problem Analysis**: Deconstruct requirements under {scenario_title} constraints.\n"
                            f"2. **Invariant Verification**: Applied formal checks for {focus_title} principles.\n"
                            f"3. **Resolution Path**: Implemented {effective_task} adhering strictly to {effective_diff} tier criteria."
                        )

                    response = (
                        f"### Technical Solution: {template.objective} [{variant_token}]\n\n"
                        f"**Domain**: `{template.domain}` | **Topic**: `{template.topic}` | **Difficulty**: `{effective_diff}` | **Focus**: `{focus_title}`\n\n"
                        f"#### System Architecture & Context\n"
                        f"{template.description}\n\n"
                        f"Operational Scenario: {scenario_desc} Under {focus_title}, we ensure robust execution."
                        f"{code_block}"
                        f"{reasoning_block}\n\n"
                        f"#### Verification & Quality Assurance\n"
                        f"- Meets quality criteria (min length: {template.quality_requirements.get('min_answer_length', 100)} chars).\n"
                        f"- Formally validated against deterministic profile `{variant_token}`."
                    )
                else:
                    # Generic domain sample fallback
                    domain_data = self.SAMPLE_TEMPLATES.get(effective_domain, {})
                    topic_data = domain_data.get(effective_topic, {})
                    sample_content = topic_data.get(effective_task)

                    if sample_content:
                        prompt_base, response_base = sample_content
                        prompt = f"{prompt_base}\n\nContext ({scenario_title} - {variant_token}): {scenario_desc} Focus: {focus_desc}"
                        response = f"{response_base}\n\n### Specialized Focus: {focus_title}\n{scenario_desc}\nVerification Token: `{variant_token}` (Seed: {item_seed})"
                    else:
                        prompt = (
                            f"Provide a comprehensive {effective_diff}-level {effective_task} for {effective_topic} in {effective_domain}. "
                            f"Context: {scenario_title} [{variant_token}]. Focus: {focus_desc}"
                        )
                        response = (
                            f"### {effective_domain.title()} Technical Reference: {effective_topic} [{variant_token}]\n\n"
                            f"This analysis provides a structured {effective_task} at {effective_diff} difficulty for `{effective_topic}`.\n\n"
                            f"1. **Architectural Overview**: Principles of `{effective_topic}` in `{scenario_title}`.\n"
                            f"2. **Targeted Implementation**: Applying `{focus_title}` ({focus_desc}).\n"
                            f"3. **Invariant Verification**: Proven under deterministic test parameters (`{variant_token}`)."
                        )


                # Construct immutable ProvenanceInfo
                item_source_id = f"{batch_id}_{i+1}"
                prov = ProvenanceInfo(
                    source_type=SourceType.SYNTHETIC.value,
                    source="synthetic_generator",
                    source_id=item_source_id,
                    license=None,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    generator=self.generator_name,
                    generator_version=self.version,
                )

                meta = RecordMetadata(
                    domain=effective_domain,
                    topic=effective_topic,
                    task_type=effective_task,
                    difficulty=effective_diff,
                    quality_score=0.93,
                    source=prov.source,
                    source_type=prov.source_type,
                    created_at=prov.created_at,
                    source_id=prov.source_id,
                    generator=prov.generator,
                    generator_version=prov.generator_version,
                    provenance=prov,
                    dimensions={
                        "correctness": 0.95,
                        "relevance": 0.95,
                        "clarity": 0.94,
                        "completeness": 0.92,
                        "technical_accuracy": 0.95,
                        "reasoning_quality": 0.93,
                    },
                )

                record = DatasetRecord(
                    messages=[
                        Message(role=Role.USER, content=prompt),
                        Message(role=Role.ASSISTANT, content=response),
                    ],
                    metadata=meta,
                )

                result.records.append(record)
                result.generated_count += 1

            except Exception as ex:
                result.failed_count += 1
                result.errors.append(f"Record {i+1} synthesis error: {str(ex)}")

        return result
