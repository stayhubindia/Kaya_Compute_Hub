"""
Tests for Synthetic Dataset Generation Engine (Phase 2.3.3).
Validates GenerationRequest, GenerationResult, SampleSyntheticGenerator,
reproducibility, template integration, error handling, and pipeline preservation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.dataset.generator import (
    GenerationRequest,
    GenerationResult,
    SampleSyntheticGenerator,
)
from src.dataset.pipeline import DatasetPipeline
from src.dataset.schema import DatasetRecord, DifficultyLevel, Role, TaskType
from src.dataset.template_registry import TaskTemplate, TemplateRegistry


@pytest.fixture
def manifest_path() -> Path:
    return Path("configs/domain_templates.yaml")


@pytest.fixture
def template_registry(manifest_path: Path) -> TemplateRegistry:
    return TemplateRegistry.from_yaml(manifest_path)


@pytest.fixture
def sample_template() -> TaskTemplate:
    return TaskTemplate(
        id="programming_python_debugging_intermediate",
        domain="programming",
        topic="python",
        task_type="debugging",
        difficulty="intermediate",
        objective="diagnose_and_fix_python_runtime_defects",
        description="Identify subtle edge-case errors, mutable default arguments, or scope leaks in Python code and provide corrected implementations.",
        supported_difficulties=["intermediate", "advanced"],
        quality_requirements={"min_answer_length": 120, "require_code_blocks": True, "require_reasoning": True},
    )


# ============================================================================
# 1. GENERATION REQUEST TESTS
# ============================================================================

def test_generation_request_valid():
    req = GenerationRequest(
        template_id="programming_python_debugging_intermediate",
        number_of_examples=5,
        seed=123,
        generation_batch_id="custom-batch-001",
    )
    assert req.template_id == "programming_python_debugging_intermediate"
    assert req.number_of_examples == 5
    assert req.seed == 123
    assert req.get_effective_batch_id() == "custom-batch-001"


def test_generation_request_default_batch_id():
    req = GenerationRequest(
        template_id="linux_systems_filesystem_permissions_beginner",
        number_of_examples=2,
        seed=42,
    )
    batch_id = req.get_effective_batch_id()
    assert "linux_systems_filesystem_permissions_beginner" in batch_id
    assert "s42" in batch_id


def test_generation_request_invalid_count():
    with pytest.raises(ValidationError):
        GenerationRequest(
            template_id="programming_python_debugging_intermediate",
            number_of_examples=0,
        )

    with pytest.raises(ValidationError):
        GenerationRequest(
            template_id="programming_python_debugging_intermediate",
            number_of_examples=-10,
        )


def test_generation_request_invalid_difficulty():
    with pytest.raises(ValidationError):
        GenerationRequest(
            template_id="programming_python_debugging_intermediate",
            difficulty="super_extreme_hard",
        )


def test_generation_request_invalid_task_type():
    with pytest.raises(ValidationError):
        GenerationRequest(
            template_id="programming_python_debugging_intermediate",
            task_type="unregistered_task",
        )


def test_generation_request_validate_against_template_conflicts(sample_template: TaskTemplate):
    # Domain mismatch
    req_domain_mismatch = GenerationRequest(
        template_id=sample_template.id,
        domain="cybersecurity",
    )
    with pytest.raises(ValueError, match="conflicts with template domain"):
        req_domain_mismatch.validate_against_template(sample_template)

    # Topic mismatch
    req_topic_mismatch = GenerationRequest(
        template_id=sample_template.id,
        topic="cryptography",
    )
    with pytest.raises(ValueError, match="conflicts with template topic"):
        req_topic_mismatch.validate_against_template(sample_template)

    # Task type mismatch
    req_task_mismatch = GenerationRequest(
        template_id=sample_template.id,
        task_type="system_design",
    )
    with pytest.raises(ValueError, match="conflicts with template task_type"):
        req_task_mismatch.validate_against_template(sample_template)

    # Unsupported difficulty
    req_diff_mismatch = GenerationRequest(
        template_id=sample_template.id,
        difficulty="beginner",
    )
    with pytest.raises(ValueError, match="not supported by template"):
        req_diff_mismatch.validate_against_template(sample_template)


# ============================================================================
# 2. SAMPLE GENERATOR & REPRODUCIBILITY TESTS
# ============================================================================

def test_sample_generator_deterministic_reproducibility(template_registry: TemplateRegistry):
    generator = SampleSyntheticGenerator()
    req = GenerationRequest(
        template_id="programming_python_debugging_intermediate",
        number_of_examples=3,
        seed=42,
        generation_batch_id="repro_batch_001",
    )

    result_1 = generator.generate_batch(req, template_registry=template_registry)
    result_2 = generator.generate_batch(req, template_registry=template_registry)

    assert result_1.is_successful is True
    assert result_2.is_successful is True
    assert result_1.generated_count == 3
    assert result_2.generated_count == 3

    for rec1, rec2 in zip(result_1.records, result_2.records):
        assert rec1.canonical_content_hash() == rec2.canonical_content_hash()
        assert rec1.messages[0].content == rec2.messages[0].content
        assert rec1.messages[1].content == rec2.messages[1].content
        assert rec1.metadata.provenance.source_id == rec2.metadata.provenance.source_id


def test_sample_generator_different_seeds_vary(template_registry: TemplateRegistry):
    generator = SampleSyntheticGenerator()
    req1 = GenerationRequest(
        template_id="programming_python_debugging_intermediate",
        number_of_examples=2,
        seed=42,
    )
    req2 = GenerationRequest(
        template_id="programming_python_debugging_intermediate",
        number_of_examples=2,
        seed=9999,
    )

    result_1 = generator.generate_batch(req1, template_registry=template_registry)
    result_2 = generator.generate_batch(req2, template_registry=template_registry)

    assert result_1.is_successful is True
    assert result_2.is_successful is True

    hashes_1 = [r.canonical_content_hash() for r in result_1.records]
    hashes_2 = [r.canonical_content_hash() for r in result_2.records]
    assert hashes_1 != hashes_2


def test_sample_generator_provenance_preservation(template_registry: TemplateRegistry):
    generator = SampleSyntheticGenerator()
    req = GenerationRequest(
        template_id="linux_systems_systemd_services_troubleshooting_intermediate",
        number_of_examples=2,
        seed=77,
        generation_batch_id="systemd_test_batch",
    )

    result = generator.generate_batch(req, template_registry=template_registry)
    assert result.is_successful is True
    assert result.generated_count == 2

    for i, rec in enumerate(result.records):
        assert rec.metadata.source_type == "synthetic"
        assert rec.metadata.source == "synthetic_generator"
        assert rec.metadata.generator == "sample_test_generator"
        assert rec.metadata.generator_version == "1.0.0"
        assert rec.metadata.provenance is not None
        assert rec.metadata.provenance.source_id == f"systemd_test_batch_{i+1}"
        assert rec.metadata.domain == "linux_systems"
        assert rec.metadata.topic == "systemd_services"
        assert rec.metadata.task_type == "troubleshooting"


def test_sample_generator_explicit_failure_unknown_template(template_registry: TemplateRegistry):
    generator = SampleSyntheticGenerator()
    req = GenerationRequest(
        template_id="unknown_nonexistent_template_xyz",
        number_of_examples=3,
        seed=42,
    )

    result = generator.generate_batch(req, template_registry=template_registry)
    assert result.is_successful is False
    assert result.failed_count == 3
    assert result.generated_count == 0
    assert len(result.errors) > 0
    assert "not found in registry" in result.errors[0]


def test_generation_result_save_jsonl_and_overwrite_guard(tmp_path: Path, template_registry: TemplateRegistry):
    generator = SampleSyntheticGenerator()
    req = GenerationRequest(
        template_id="cybersecurity_cryptography_explanation_intermediate",
        number_of_examples=2,
        seed=42,
    )
    result = generator.generate_batch(req, template_registry=template_registry)
    assert result.is_successful is True

    out_file = tmp_path / "test_out.jsonl"
    count = result.save_jsonl(out_file, overwrite=False)
    assert count == 2
    assert out_file.is_file()

    # Overwrite without overwrite=True raises FileExistsError
    with pytest.raises(FileExistsError):
        result.save_jsonl(out_file, overwrite=False)

    # With overwrite=True succeeds
    count_2 = result.save_jsonl(out_file, overwrite=True)
    assert count_2 == 2


# ============================================================================
# 3. PIPELINE INTEGRATION & PROVENANCE PRESERVATION
# ============================================================================

def test_pipeline_integration_synthetic_batch_end_to_end(tmp_path: Path, template_registry: TemplateRegistry):
    """
    Generates a synthetic batch, saves it, and runs it through the full Phase 2.2 processing pipeline.
    Verifies that normalizer, cleaner, deduplicator, quality validator, splitter, and statistics
    process the synthetic records and preserve provenance to the final train/val/test splits.
    """
    generator = SampleSyntheticGenerator()
    req = GenerationRequest(
        template_id="ai_ml_qlora_peft_coding_advanced",
        number_of_examples=8,
        seed=42,
        generation_batch_id="pilot_e2e_qlora",
    )
    gen_result = generator.generate_batch(req, template_registry=template_registry)
    assert gen_result.is_successful is True

    raw_file = tmp_path / "raw_synthetic.jsonl"
    gen_result.save_jsonl(raw_file)

    processed_dir = tmp_path / "processed"
    pipeline = DatasetPipeline(config_path=Path("configs/dataset.yaml"))
    pipe_result = pipeline.run(
        input_path=raw_file,
        output_dir=processed_dir,
        save_outputs=True,
    )

    # Pipeline validations
    assert pipe_result.total_raw == 8
    assert pipe_result.accepted_count == 8
    assert pipe_result.rejected_count == 0
    assert pipe_result.split_result is not None

    # Check generated files exist
    assert (processed_dir / "train.jsonl").is_file()
    assert (processed_dir / "dataset_report.json").is_file()
    assert (processed_dir / "source_report.json").is_file()

    # Verify provenance survives in output train split
    with open(processed_dir / "train.jsonl", "r", encoding="utf-8") as f:
        train_lines = [json.loads(line) for line in f]

    assert len(train_lines) > 0
    for record_dict in train_lines:
        rec = DatasetRecord.from_dict(record_dict)
        assert rec.metadata.source_type == "synthetic"
        assert rec.metadata.provenance is not None
        assert "pilot_e2e_qlora" in rec.metadata.provenance.source_id
        assert rec.metadata.provenance.generator == "sample_test_generator"


# ============================================================================
# 4. CLI INTEGRATION TEST
# ============================================================================

def test_cli_generate_dataset_script(tmp_path: Path):
    out_file = tmp_path / "cli_generated.jsonl"
    cmd = [
        sys.executable,
        "scripts/generate_dataset.py",
        "--template", "networking_tcp_udp_troubleshooting_advanced",
        "--count", "3",
        "--seed", "100",
        "--output", str(out_file),
        "--overwrite",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed: {res.stderr}"
    assert "Status:           SUCCESS" in res.stdout
    assert "Requested:        3" in res.stdout
    assert "Generated:        3" in res.stdout
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 3
    for l in lines:
        assert l["metadata"]["domain"] == "networking"
        assert l["metadata"]["topic"] == "tcp_udp"
