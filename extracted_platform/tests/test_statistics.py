from src.dataset.schema import DatasetRecord, Message, RecordMetadata, Role
from src.dataset.statistics import DatasetStatistics


def test_statistics_report_generation():
    records = [
        DatasetRecord(
            messages=[
                Message(role=Role.USER, content="How do I use pytest in Python?"),
                Message(role=Role.ASSISTANT, content="Run pytest command in terminal to execute your test functions."),
            ],
            metadata=RecordMetadata(
                domain="programming",
                topic="python",
                task_type="coding",
                difficulty="beginner",
                quality_score=0.92,
                source="test",
                source_type="synthetic",
                created_at="2026-08-11T16:00:00Z",
            ),
        ),
        DatasetRecord(
            messages=[
                Message(role=Role.USER, content="What is an inode in Linux?"),
                Message(role=Role.ASSISTANT, content="An inode is a data structure storing file metadata on disk."),
            ],
            metadata=RecordMetadata(
                domain="linux_systems",
                topic="storage",
                task_type="explanation",
                difficulty="intermediate",
                quality_score=0.94,
                source="test",
                source_type="synthetic",
                created_at="2026-08-11T16:00:00Z",
            ),
        ),
    ]

    stats = DatasetStatistics(
        domain_targets={"programming": 0.5, "linux_systems": 0.5},
        difficulty_targets={"beginner": 0.5, "intermediate": 0.5},
    )
    metrics = stats.compute_metrics(raw_total=2, accepted_records=records)

    assert metrics["summary"]["total_raw_inputs"] == 2
    assert metrics["summary"]["accepted_examples"] == 2
    assert "programming" in metrics["domain_distribution"]
    assert "linux_systems" in metrics["domain_distribution"]

    md_report = stats.generate_markdown_report(metrics)
    assert "# Dataset Engineering & Quality Report" in md_report
    assert "programming" in md_report
