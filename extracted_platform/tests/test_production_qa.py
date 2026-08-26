"""
Unit and Integration Tests for Phase 3.3 Production Dataset QA, Token Budget & Freeze Engine.
"""

import json
import pytest
from pathlib import Path
from typing import List

from src.dataset.schema import (
    DatasetRecord,
    Message,
    RecordMetadata,
    ProvenanceInfo,
    Role,
    DifficultyLevel,
    TaskType,
    SourceType,
)
from src.dataset.production import (
    DatasetFreezeState,
    ProductionManifest,
    ProductionPlanner,
)
from src.dataset.production_qa import (
    ProductionQAEngine,
    ReadinessStatus,
    SchemaQAResult,
    ProvenanceQAResult,
    DomainQAResult,
    DifficultyQAResult,
    QualityQAResult,
    YieldLossAttribution,
    DuplicateQAResult,
    LeakageQAResult,
    TokenQAResult,
    TrainingEstimateResult,
    FreezeQAResult,
    ProductionQAReport,
)
from src.dataset.production_generator import ProductionGenerationEngine, atomic_write_jsonl


# Helper Mock Tokenizer
class MockQwenTokenizer:
    """Deterministic mock tokenizer for testing token statistics and truncation detection."""
    name_or_path = "MockQwen3Tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # Approximate 1 token per word + 2 special tokens
        words = text.split()
        return [100 + i for i in range(len(words))]


def _create_sample_record(
    idx: int,
    domain: str = "programming",
    difficulty: str = "intermediate",
    task_type: str = "coding",
    quality_score: float = 0.95,
    user_text: str = "Explain how quicksort works in Python with code examples.",
    asst_text: str = "Quicksort is a divide-and-conquer algorithm with average O(n log n) time complexity.",
    has_provenance: bool = True,
    generator: str = "template_engine_v1",
) -> DatasetRecord:
    prov = None
    if has_provenance:
        prov = ProvenanceInfo(
            source_type=SourceType.SYNTHETIC.value,
            source="synthetic_generator",
            source_id=f"synth_{idx}",
            generator=generator,
            generator_version="1.0.0",
            created_at="2026-08-12T00:00:00Z",
            license="MIT",
        )
    return DatasetRecord(
        messages=[
            Message(role=Role.USER, content=user_text),
            Message(role=Role.ASSISTANT, content=asst_text),
        ],
        metadata=RecordMetadata(
            domain=domain,
            topic="general",
            difficulty=difficulty,
            task_type=task_type,
            source_type=SourceType.SYNTHETIC.value,
            source="synthetic_generator",
            quality_score=quality_score,
            provenance=prov,
        ),
    )


def _create_full_domain_records(count_per_domain: int = 4) -> List[DatasetRecord]:
    prompts_pool = {
        "programming": [
            ("How does memory allocation work on the heap in C++?", "Heap memory is managed via dynamic allocation operators like new and delete."),
            ("Explain Python GIL multithreading limitations.", "The GIL enforces single-thread bytecode execution, making multiprocessing preferable for CPU tasks."),
            ("What is tail call optimization in functional languages?", "TCO reuses the current stack frame for recursive calls in tail position to avoid stack overflow."),
            ("Describe the borrow checker mechanics in Rust.", "Rust checks ownership and lifetime annotations at compile time to ensure reference validity."),
        ],
        "cybersecurity": [
            ("What is the difference between asymmetric and symmetric encryption?", "Symmetric uses the same key for encryption/decryption while asymmetric uses public/private pairs."),
            ("How do cross-site request forgery attacks exploit browser cookies?", "CSRF tricks authenticated browsers into executing unwanted actions using ambient session cookies."),
            ("Explain the concept of zero-trust architecture in enterprise security.", "Zero trust enforces continuous verification for every access request regardless of perimeter origin."),
            ("How does ASLR protect against return-oriented programming?", "ASLR randomizes memory addresses of program segments to hinder predictable payload execution."),
        ],
        "software_engineering": [
            ("Explain the single responsibility principle with an example.", "SRP states that a software module should have one, and only one, reason to change."),
            ("What are the advantages of event sourcing over CRUD state mutations?", "Event sourcing records all state transitions as immutable events, enabling full audit trails."),
            ("How does continuous integration improve deployment velocity?", "CI automates build verification and test execution upon code commits, catching defects early."),
            ("What is circuit breaker pattern in microservices resilience?", "Circuit breaker detects failing downstream calls and trips open to prevent cascading failure."),
        ],
        "linux_systems": [
            ("How does systemd manage service dependencies during boot?", "systemd uses unit dependencies and socket activation to parallelize service initialization."),
            ("Explain inode structures in ext4 filesystems.", "Inodes store file metadata including permissions, ownership, timestamps, and block pointers."),
            ("What is virtual memory swapping and thrashing?", "Thrashing occurs when excessive page faults force the system to spend more time swapping than executing."),
            ("How do Linux cgroups limit container resource usage?", "Control groups enforce hierarchical limits on CPU, memory, and I/O consumption per process group."),
        ],
        "networking": [
            ("How does the TCP three-way handshake establish a reliable connection?", "Handshake exchanges SYN, SYN-ACK, and ACK packets to synchronize initial sequence numbers."),
            ("Explain the role of DNS recursive resolvers and root servers.", "Resolvers query root, TLD, and authoritative nameservers to translate hostnames into IP addresses."),
            ("What is the purpose of BGP in Internet routing?", "Border Gateway Protocol exchanges reachability information between autonomous systems globally."),
            ("How does QUIC reduce connection establishment latency over TLS/UDP?", "QUIC merges cryptographic and transport handshakes into a single round trip over UDP."),
        ],
        "ai_ml": [
            ("Explain multi-head attention mechanism in Transformer models.", "Multi-head attention projects queries, keys, and values into subspaces to attend to diverse features."),
            ("What is the difference between batch normalization and layer normalization?", "LayerNorm normalizes across features per example, making it ideal for sequential transformer inputs."),
            ("How does QLoRA achieve 4-bit parameter-efficient fine-tuning?", "QLoRA combines NF4 quantization with double quantization and paged optimizers to reduce VRAM."),
            ("Explain the bias-variance tradeoff in supervised learning.", "High bias causes underfitting while high variance causes overfitting to training noise."),
        ],
        "general_knowledge": [
            ("What was the impact of the Industrial Revolution on urbanization?", "Mechanization concentrated factories in cities, driving mass migration from rural areas."),
            ("Explain the structure and function of the United Nations Security Council.", "The Security Council maintains international peace, featuring 5 permanent and 10 rotating members."),
            ("What were the primary achievements of the Apollo space program?", "Apollo landed twelve astronauts on the Moon and returned extensive lunar geological samples."),
            ("Describe the global economic effects of the Bretton Woods conference.", "Bretton Woods established the IMF, World Bank, and pegged currencies to the gold-backed US dollar."),
        ],
        "reasoning": [
            ("Solve this logic scenario: If all A are B, and some B are C...", "It does not necessarily follow that any A are C, since the intersection between B and C may exclude A."),
            ("Evaluate the fallacious reasoning in an ad hominem argument.", "Ad hominem attacks the speaker rather than evaluating the substantive validity of the proposition."),
            ("How does Bayesian inference update probabilities with new evidence?", "Posterior probability is proportional to the prior probability multiplied by the evidence likelihood."),
            ("What is the difference between deductive and inductive reasoning?", "Deductive reasoning guarantees truth if premises hold; inductive reasoning infers probabilistic patterns."),
        ],
        "mathematics": [
            ("Derive the Taylor series expansion of e^x around zero.", "The derivative of e^x is e^x; evaluating at zero yields coefficients 1/k!, so sum x^k / k!."),
            ("What is the fundamental theorem of calculus?", "It links differentiation and integration, stating that definite integrals compute antiderivative differences."),
            ("Explain singular value decomposition for rectangular matrices.", "SVD factors matrix A into U * Sigma * V^T, decomposing linear transformations into rotations and scales."),
            ("How does Lagrange multiplier optimization handle equality constraints?", "It solves grad(f) = lambda * grad(g) to find constrained extrema along level curves."),
        ],
        "science": [
            ("Describe the mechanism of cellular respiration in mitochondria.", "Respiration comprises glycolysis, Krebs cycle, and electron transport chain yielding ATP."),
            ("How does CRISPR-Cas9 locate and cleave target DNA?", "Cas9 guided by crRNA scans DNA for PAM motifs and unwinds the duplex to cleave matching target strands."),
            ("Explain general relativity concept of gravitational spacetime curvature.", "Mass-energy curves spacetime, and free-falling objects follow geodesic paths within this geometry."),
            ("What is the thermodynamic second law and entropy increase?", "Isolated systems evolve towards maximum thermodynamic entropy and thermodynamic equilibrium."),
        ],
        "psychology": [
            ("How does working memory capacity constrain cognitive load?", "Working memory holds roughly 4-7 chunks of information; exceeding this degrades problem-solving."),
            ("Explain classical versus operant conditioning paradigms.", "Classical pairs involuntary reflexes with stimuli; operant pairs voluntary actions with reinforcements."),
            ("What is the Yerkes-Dodson law relating arousal to performance?", "Performance increases with physiological arousal up to an optimal peak, beyond which it declines."),
            ("Describe the psychological mechanisms behind cognitive dissonance.", "Discomfort from conflicting cognitions motivates attitude change or rationalization to restore harmony."),
        ],
        "human_behavior": [
            ("How does the bystander effect reduce intervention probability?", "Diffusion of responsibility and pluralistic ignorance lead individuals to assume others will act."),
            ("Analyze the effect of loss aversion on consumer purchasing decisions.", "Kahneman and Tversky showed that losses are perceived as roughly twice as psychologically painful as gains."),
            ("Explain how social proof guides collective group dynamics.", "People look to peer actions and normative consensus when navigating ambiguous social situations."),
            ("What is the sunk cost fallacy in human decision-making?", "Continuing an endeavor due to past resource investment rather than evaluating prospective net utility."),
        ],
        "technology": [
            ("How do SSD NVMe controllers achieve high throughput over PCIe?", "NVMe uses parallel command queues supporting up to 64K entries, bypassing legacy SATA bottlenecks."),
            ("Explain the architecture of distributed consensus algorithms like Raft.", "Raft elects a leader, replicates log entries sequentially, and commits entries upon majority consensus."),
            ("What are the advantages of WebAssembly for browser applications?", "Wasm provides a near-native binary format with deterministic memory execution for compute-heavy tasks."),
            ("How does 5G beamforming improve cellular spectrum efficiency?", "Phased array antennas direct wireless signals toward specific devices rather than broadcasting omnidirectionally."),
        ],
    }

    difficulties = ["beginner", "intermediate", "advanced", "expert"]
    records = []
    idx = 0
    for d, prompts in prompts_pool.items():
        for i in range(count_per_domain):
            q_idx = i % len(prompts)
            diff = difficulties[i % len(difficulties)]
            q_text, a_text = prompts[q_idx]
            records.append(_create_sample_record(
                idx=idx,
                domain=d,
                difficulty=diff,
                quality_score=0.92,
                user_text=q_text,
                asst_text=a_text,
            ))
            idx += 1
    return records


# ============================================================================
# 1. SCHEMA QA TESTS
# ============================================================================

def test_schema_qa_valid_records():
    engine = ProductionQAEngine()
    records = _create_full_domain_records(count_per_domain=2)
    res = engine.validate_schema(records)
    assert res.total_records == len(records)
    assert res.valid_records == len(records)
    assert res.invalid_records == 0
    assert res.schema_error_count == 0


def test_schema_qa_invalid_records():
    engine = ProductionQAEngine()
    valid_rec = _create_sample_record(0)
    invalid_dict = {"messages": "not_a_list", "metadata": {}}
    res = engine.validate_schema([valid_rec, invalid_dict])
    assert res.total_records == 2
    assert res.valid_records == 1
    assert res.invalid_records == 1
    assert res.schema_error_count == 1


# ============================================================================
# 2. PROVENANCE QA TESTS
# ============================================================================

def test_provenance_qa_complete():
    engine = ProductionQAEngine()
    records = _create_full_domain_records(count_per_domain=2)
    res = engine.validate_provenance(records)
    assert res.provenance_completeness == 1.0
    assert res.records_with_provenance == len(records)
    assert res.records_without_provenance == 0


def test_provenance_qa_missing_fields():
    engine = ProductionQAEngine()
    r1 = _create_sample_record(0, has_provenance=True)
    r2 = _create_sample_record(1, has_provenance=False)
    r3 = _create_sample_record(2, has_provenance=True, generator="")
    res = engine.validate_provenance([r1, r2, r3])
    assert res.records_with_provenance == 1
    assert res.records_without_provenance == 2
    assert res.provenance_completeness == pytest.approx(1.0 / 3.0)


# ============================================================================
# 3. DISTRIBUTION QA (DOMAIN & DIFFICULTY)
# ============================================================================

def test_distribution_qa_all_domains_represented():
    engine = ProductionQAEngine()
    records = _create_full_domain_records(count_per_domain=5)
    dom_qa, diff_qa = engine.validate_distributions(records)

    assert dom_qa.all_represented is True
    assert len(dom_qa.missing_domains) == 0
    assert diff_qa.all_represented is True
    assert len(diff_qa.missing_difficulties) == 0
    assert len(dom_qa.breakdowns) == 13
    assert len(diff_qa.breakdowns) == 4


def test_distribution_qa_missing_domain():
    engine = ProductionQAEngine()
    records = [_create_sample_record(i, domain="programming") for i in range(10)]
    dom_qa, diff_qa = engine.validate_distributions(records)

    assert dom_qa.all_represented is False
    assert "cybersecurity" in dom_qa.missing_domains
    assert len(dom_qa.missing_domains) == 12


# ============================================================================
# 4. QUALITY QA TESTS
# ============================================================================

def test_quality_qa_statistics():
    engine = ProductionQAEngine()
    records = [
        _create_sample_record(0, quality_score=0.86),
        _create_sample_record(1, quality_score=0.90),
        _create_sample_record(2, quality_score=0.94),
        _create_sample_record(3, quality_score=0.98),
    ]
    res = engine.validate_quality(records)
    assert res.evaluated_count == 4
    assert res.mean == pytest.approx(0.92)
    assert res.minimum == 0.86
    assert res.maximum == 0.98
    assert res.count_ge_085 == 4
    assert res.count_ge_090 == 3
    assert res.pct_ge_085 == 1.0


# ============================================================================
# 5. DUPLICATE & LEAKAGE QA
# ============================================================================

def test_duplicate_qa_detection():
    engine = ProductionQAEngine()
    r1 = _create_sample_record(0, user_text="What is a binary search tree?")
    r2 = _create_sample_record(1, user_text="What is a binary search tree?")  # Exact duplicate
    r3 = _create_sample_record(2, user_text="Explain Dijkstra algorithm for shortest path.")

    res = engine.validate_duplicates([r1, r2, r3])
    assert res.exact_duplicate_count == 1
    assert res.duplicate_rate == pytest.approx(1.0 / 3.0)


def test_cross_split_leakage_clean_and_dirty():
    engine = ProductionQAEngine()
    tr = [_create_sample_record(0, user_text="Train prompt 1")]
    val = [_create_sample_record(1, user_text="Val prompt 1")]
    test = [_create_sample_record(2, user_text="Test prompt 1")]

    clean_res = engine.validate_cross_split_leakage(tr, val, test)
    assert clean_res.is_clean is True
    assert clean_res.total_exact_leaks == 0

    # Dirty split: leaking train into test
    dirty_test = [_create_sample_record(3, user_text="Train prompt 1")]
    dirty_res = engine.validate_cross_split_leakage(tr, val, dirty_test)
    assert dirty_res.is_clean is False
    assert dirty_res.train_test_exact == 1
    assert dirty_res.total_exact_leaks == 1


# ============================================================================
# 6. TOKENIZATION & TRAINING ESTIMATE TESTS
# ============================================================================

def test_token_qa_with_mock_tokenizer():
    mock_tok = MockQwenTokenizer()
    engine = ProductionQAEngine(tokenizer_override=mock_tok)

    records = [
        _create_sample_record(0, user_text="Short prompt", asst_text="Short answer"),
        _create_sample_record(1, user_text="A " * 50, asst_text="B " * 100),
    ]

    tok_res = engine.analyze_tokens(records, max_sequence_length=4096)
    assert tok_res.is_available is True
    assert tok_res.total_conversation_tokens > 0
    assert tok_res.gt_4096_count == 0
    assert tok_res.mean_tokens > 0


def test_token_qa_unavailable_fallback():
    engine = ProductionQAEngine()
    # Force tokenizer to None
    engine._tokenizer = None
    engine._tokenizer_loaded = True
    engine._tokenizer_status = "TOKEN_ANALYSIS_UNAVAILABLE (Model not found)"

    records = [_create_sample_record(0)]
    tok_res = engine.analyze_tokens(records)
    assert tok_res.is_available is False
    assert "TOKEN_ANALYSIS_UNAVAILABLE" in tok_res.tokenizer_status
    assert tok_res.total_conversation_tokens == 0


def test_training_estimate_budget():
    engine = ProductionQAEngine()
    res = engine.estimate_training_budget(
        record_count=1000,
        total_tokens=250000,
        epochs=[1, 2, 3],
        micro_batch_size=1,
        gradient_accumulation_steps=8,
    )
    assert res.is_estimate is True
    assert res.effective_batch_size == 8
    assert res.steps_per_epoch == 125
    assert res.tokens_for_epochs[1] == 250000
    assert res.tokens_for_epochs[2] == 500000
    assert res.tokens_for_epochs[3] == 750000
    assert res.total_steps_for_epochs[3] == 375


# ============================================================================
# 7. YIELD LOSS ATTRIBUTION (STAGE-B VERIFICATION)
# ============================================================================

def test_yield_loss_attribution_stage_b():
    engine = ProductionQAEngine()
    res = engine.analyze_yield(
        candidate_count=100,
        clean_count=100,
        quality_count=99,
        unique_count=94,
        balanced_selected_count=59,
    )
    assert res.candidate_count == 100
    assert res.loss_at_cleaning == 0
    assert res.loss_at_quality == 1
    assert res.loss_at_exact_duplicate == 5
    assert res.loss_at_balancing == 35
    assert res.balanced_selected_count == 59
    assert res.overall_yield_pct == 0.59
    assert len(res.loss_notes) == 4


# ============================================================================
# 8. READINESS GATES & FREEZE TESTS
# ============================================================================

def test_readiness_gates_pass(tmp_path):
    engine = ProductionQAEngine(tokenizer_override=MockQwenTokenizer())
    records = _create_full_domain_records(count_per_domain=4)

    report = engine.run_qa(records, version="test-v1.0")
    assert report.critical_gates_passed is True
    assert report.overall_readiness in [ReadinessStatus.PASS, ReadinessStatus.WARN]


def test_freeze_lifecycle_and_protection(tmp_path):
    engine = ProductionQAEngine(tokenizer_override=MockQwenTokenizer())
    records = _create_full_domain_records(count_per_domain=5)

    # Save candidates jsonl
    data_file = tmp_path / "candidate_dataset.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    # Create initial manifest
    planner = ProductionPlanner()
    plan = planner.plan(target_count=len(records), version="dataset-v1.0", output_dir=tmp_path)
    manifest = planner.create_initial_manifest(plan)
    man_path = tmp_path / "production_manifest.json"
    manifest.save(man_path)

    # Execute Freeze
    reports_dir = tmp_path / "reports"
    frozen_man, rep = engine.freeze_dataset(
        manifest_path=man_path,
        dataset_records=data_file,
        reports_dir=reports_dir,
        force=True,
    )

    assert frozen_man.status == DatasetFreezeState.FROZEN.value
    assert rep.freeze_qa.is_frozen is True
    assert rep.freeze_qa.dataset_sha256 is not None
    assert (reports_dir / "qa_report.json").is_file()
    assert (reports_dir / "qa_report.md").is_file()

    # Verify Generator Protection against frozen dataset
    gen_engine = ProductionGenerationEngine()
    with pytest.raises(RuntimeError, match="FROZEN and immutable"):
        gen_engine.generate_all(
            target_count=len(records),
            version="dataset-v1.0",
            output_dir=tmp_path,
        )


def test_qa_reproducibility():
    engine = ProductionQAEngine(tokenizer_override=MockQwenTokenizer())
    records = _create_full_domain_records(count_per_domain=4)

    rep1 = engine.run_qa(records, version="repro-v1.0")
    rep2 = engine.run_qa(records, version="repro-v1.0")

    assert rep1.record_count == rep2.record_count
    assert rep1.quality_qa.mean == rep2.quality_qa.mean
    assert rep1.provenance_qa.provenance_completeness == rep2.provenance_qa.provenance_completeness
    assert rep1.overall_readiness == rep2.overall_readiness
