from src.dataset.quality import QualityState, QualityValidator
from src.dataset.schema import DatasetRecord, Message, RecordMetadata, Role


def make_record_with_score(score: float | None) -> DatasetRecord:
    return DatasetRecord(
        messages=[
            Message(role=Role.USER, content="Explain symmetric encryption vs asymmetric encryption."),
            Message(role=Role.ASSISTANT, content="Symmetric uses single key; asymmetric uses key pairs."),
        ],
        metadata=RecordMetadata(
            domain="cybersecurity",
            topic="cryptography",
            task_type="explanation",
            difficulty="intermediate",
            quality_score=score,
            source="test",
            source_type="synthetic",
            created_at="2026-08-11T16:00:00Z",
        ),
    )


def test_quality_validator_thresholds():
    validator = QualityValidator(minimum_score=0.85, preferred_score=0.90, enforce_threshold=True)

    r_high = make_record_with_score(0.95)
    r_med = make_record_with_score(0.86)
    r_low = make_record_with_score(0.70)
    r_unscored = make_record_with_score(None)

    accepted, report = validator.validate_records([r_high, r_med, r_low, r_unscored])

    assert len(accepted) == 3  # high, med, unscored accepted (allow_unscored=True)
    assert report.passed_count == 2
    assert report.failed_count == 1
    assert report.unscored_count == 1
    assert report.preferred_quality_count == 1  # 0.95 >= 0.90
