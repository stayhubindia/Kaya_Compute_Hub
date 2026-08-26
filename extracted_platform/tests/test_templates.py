"""
Tests for Domain Dataset Task Templates and Template Registry (Phase 2.3.2).
Validates template schema, registry loading, filtering, distribution statistics,
and generator integration.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from pydantic import ValidationError

from src.dataset.cleaner import DatasetCleaner
from src.dataset.generator import SampleSyntheticGenerator
from src.dataset.quality import QualityValidator
from src.dataset.schema import DatasetRecord, DifficultyLevel, TaskType
from src.dataset.template_registry import TaskTemplate, TemplateRegistry


@pytest.fixture
def sample_template() -> TaskTemplate:
    return TaskTemplate(
        id="programming_python_debugging_intermediate",
        domain="programming",
        topic="python",
        task_type="debugging",
        difficulty="intermediate",
        objective="diagnose_and_fix_python_runtime_defects",
        description="Identify subtle edge-case errors in Python and provide corrected implementations.",
        supported_difficulties=["intermediate", "advanced"],
        quality_requirements={"min_answer_length": 100, "require_code_blocks": True},
    )


@pytest.fixture
def manifest_path() -> Path:
    return Path("configs/domain_templates.yaml")


@pytest.fixture
def dataset_config_path() -> Path:
    return Path("configs/dataset.yaml")


def test_template_yaml_loading(manifest_path: Path):
    """Verifies that the YAML template manifest loads successfully with all 13 domains represented."""
    registry = TemplateRegistry.from_yaml(manifest_path)
    templates = registry.list_templates()

    assert len(templates) >= 30, f"Expected at least 30 templates, got {len(templates)}"

    domains_in_manifest = {t.domain for t in templates}
    expected_domains = {
        "programming",
        "software_engineering",
        "cybersecurity",
        "linux_systems",
        "networking",
        "ai_ml",
        "mathematics",
        "science",
        "psychology",
        "human_behavior",
        "reasoning",
        "technology",
        "general_knowledge",
    }
    assert expected_domains.issubset(domains_in_manifest), f"Missing domains: {expected_domains - domains_in_manifest}"


def test_template_validation_against_dataset_config(manifest_path: Path, dataset_config_path: Path):
    """Validates that all templates in domain_templates.yaml conform to dataset.yaml taxonomies."""
    registry = TemplateRegistry.from_yaml(manifest_path)
    result = registry.validate(dataset_config_path=dataset_config_path)

    assert result["is_valid"] is True, f"Validation errors found: {result['errors']}"
    assert len(result["errors"]) == 0


def test_invalid_template_id_empty():
    """Verifies that an empty or whitespace ID raises a validation error."""
    with pytest.raises(ValidationError):
        TaskTemplate(
            id="   ",
            domain="programming",
            topic="python",
            task_type="coding",
            difficulty="beginner",
            objective="test",
            description="test description",
        )


def test_invalid_difficulty_rejected():
    """Verifies that unsupported difficulty strings are rejected."""
    with pytest.raises(ValidationError):
        TaskTemplate(
            id="test_invalid_diff",
            domain="programming",
            topic="python",
            task_type="coding",
            difficulty="super_hard",
            objective="test",
            description="test description",
        )


def test_invalid_task_type_rejected():
    """Verifies that unapproved task types are rejected."""
    with pytest.raises(ValidationError):
        TaskTemplate(
            id="test_invalid_task",
            domain="programming",
            topic="python",
            task_type="unregistered_task_mode",
            difficulty="beginner",
            objective="test",
            description="test description",
        )


def test_duplicate_template_id_rejection(sample_template: TaskTemplate):
    """Verifies that registering a duplicate template ID without overwrite raises ValueError."""
    registry = TemplateRegistry()
    registry.register_template(sample_template)

    with pytest.raises(ValueError, match="already registered"):
        registry.register_template(sample_template, overwrite=False)

    # Overwrite should succeed
    registry.register_template(sample_template, overwrite=True)
    assert len(registry) == 1


def test_template_lookup_and_filtering(manifest_path: Path):
    """Tests template lookup by ID and filtering by domain, task_type, difficulty, and topic."""
    registry = TemplateRegistry.from_yaml(manifest_path)

    # By ID
    tmpl = registry.get_template("linux_systems_filesystem_permissions_beginner")
    assert tmpl.domain == "linux_systems"
    assert tmpl.difficulty == "beginner"

    # Missing ID raises KeyError
    with pytest.raises(KeyError):
        registry.get_template("non_existent_template_id")

    # lookup_template returns None for missing
    assert registry.lookup_template("non_existent_id") is None

    # Filter by domain
    linux_tmpls = registry.list_by_domain("linux_systems")
    assert len(linux_tmpls) >= 4
    assert all(t.domain == "linux_systems" for t in linux_tmpls)

    # Filter by task type
    troubleshooting_tmpls = registry.list_by_task_type("troubleshooting")
    assert len(troubleshooting_tmpls) >= 2
    assert all(t.task_type == "troubleshooting" for t in troubleshooting_tmpls)

    # Filter by difficulty
    expert_tmpls = registry.list_by_difficulty("expert")
    assert len(expert_tmpls) >= 3
    for t in expert_tmpls:
        assert "expert" in [t.difficulty] + t.supported_difficulties

    # Filter by topic
    math_proof_tmpls = registry.list_by_topic("mathematics", "logic_proofs")
    assert len(math_proof_tmpls) >= 1
    assert math_proof_tmpls[0].task_type == "proof"


def test_template_statistics(manifest_path: Path):
    """Verifies that template distribution statistics are properly computed."""
    registry = TemplateRegistry.from_yaml(manifest_path)
    stats = registry.template_statistics()

    assert stats["total_templates"] >= 30
    assert "programming" in stats["by_domain"]
    assert "linux_systems" in stats["by_domain"]
    assert "reasoning" in stats["by_domain"]

    # Check percentages sum to approximately 100%
    pct_sum = sum(stats["domain_percentages"].values())
    assert 99.0 <= pct_sum <= 101.0


def test_generator_template_integration(sample_template: TaskTemplate):
    """
    Verifies that the synthetic generator can generate records directly from a TaskTemplate,
    and the resulting records cleanly pass through schema, cleaning, and quality validation.
    """
    generator = SampleSyntheticGenerator()
    records = generator.generate_from_template(sample_template, number_of_examples=2)

    assert len(records) == 2
    for rec in records:
        assert isinstance(rec, DatasetRecord)
        assert rec.metadata.domain == sample_template.domain
        assert rec.metadata.topic == sample_template.topic
        assert rec.metadata.task_type == sample_template.task_type
        assert rec.metadata.difficulty == sample_template.difficulty
        assert rec.metadata.source_type == "synthetic"
        assert rec.metadata.provenance is not None
        assert sample_template.id in rec.metadata.provenance.source_id

    # Pipeline cleaning pass
    cleaner = DatasetCleaner()
    clean_records, clean_report = cleaner.clean_records(records)
    assert len(clean_records) == 2
    assert clean_report.rejected_count == 0

    # Pipeline quality validation pass
    validator = QualityValidator(minimum_score=0.85)
    accepted, qual_report = validator.validate_records(clean_records)
    assert len(accepted) == 2
    assert qual_report.passed_count == 2
