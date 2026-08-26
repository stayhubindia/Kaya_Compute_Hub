"""
Unit tests for Evaluation Reports & Manifest Generation (Phase 4.4).
"""

import json
from pathlib import Path
from src.evaluation.benchmark import OverallBenchmarkReport
from src.evaluation.config import EvaluationConfig
from src.evaluation.metrics import AggregatedMetrics
from src.evaluation.reports import EvaluationManifest, EvaluationReportManager


def test_manifest_creation_and_save(tmp_path: Path):
    mgr = EvaluationReportManager(reports_dir=tmp_path)
    cfg = EvaluationConfig()

    manifest = mgr.create_manifest(
        config=cfg,
        sample_count=7,
        dataset_sha256="testsha256",
        status="PLANNED",
    )

    assert manifest.status == "PLANNED"
    assert manifest.sample_count == 7
    manifest_path = tmp_path / "evaluation_manifest.json"
    assert manifest_path.exists()

    loaded = EvaluationManifest.load(manifest_path)
    assert loaded.evaluation_id == manifest.evaluation_id
    assert loaded.status == "PLANNED"


def test_benchmark_reports_generation(tmp_path: Path):
    mgr = EvaluationReportManager(reports_dir=tmp_path)
    cfg = EvaluationConfig()

    report = OverallBenchmarkReport(
        model_name="Qwen3-4B-Base",
        model_type="base",
        dataset_version="dataset-v1.0",
        dataset_sha256="sha123",
        sample_count=5,
        overall_metrics=AggregatedMetrics(
            total_samples=5,
            valid_responses=5,
            validity_rate=1.0,
            avg_formatting_score=0.9,
        ),
        domain_metrics={
            "programming": AggregatedMetrics(total_samples=3, valid_responses=3, validity_rate=1.0),
            "ai_ml": AggregatedMetrics(total_samples=2, valid_responses=2, validity_rate=1.0),
        },
        difficulty_metrics={
            "beginner": AggregatedMetrics(total_samples=5, valid_responses=5, validity_rate=1.0),
        },
    )

    mgr.save_benchmark_reports(report)

    assert (tmp_path / "evaluation_report.json").exists()
    assert (tmp_path / "evaluation_report.md").exists()
    assert (tmp_path / "domain_report.json").exists()
    assert (tmp_path / "difficulty_report.json").exists()
    assert (tmp_path / "task_report.json").exists()

    with open(tmp_path / "domain_report.json", "r", encoding="utf-8") as f:
        dom_data = json.load(f)
        assert "programming" in dom_data
        assert dom_data["programming"]["total_samples"] == 3
