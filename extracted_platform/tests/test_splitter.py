from src.dataset.schema import DatasetRecord, Message, RecordMetadata, Role
from src.dataset.splitter import DatasetSplitter


def make_dummy_dataset(count: int = 20) -> list[DatasetRecord]:
    domains = ["programming", "cybersecurity", "linux_systems", "ai_ml"]
    records = []
    for i in range(count):
        d = domains[i % len(domains)]
        records.append(
            DatasetRecord(
                messages=[
                    Message(role=Role.USER, content=f"User prompt number {i} in domain {d}"),
                    Message(role=Role.ASSISTANT, content=f"Assistant response number {i} providing technical details."),
                ],
                metadata=RecordMetadata(
                    domain=d,
                    topic="general",
                    task_type="explanation",
                    difficulty="intermediate",
                    quality_score=0.9,
                    source="test_gen",
                    source_type="synthetic",
                    created_at="2026-08-11T16:00:00Z",
                ),
            )
        )
    return records


def test_splitter_reproducibility():
    records = make_dummy_dataset(30)

    splitter_1 = DatasetSplitter(random_seed=42)
    res_1 = splitter_1.split(records)

    splitter_2 = DatasetSplitter(random_seed=42)
    res_2 = splitter_2.split(records)

    # Identical record ordering in train, val, test
    assert [r.canonical_content_hash() for r in res_1.train] == [r.canonical_content_hash() for r in res_2.train]
    assert [r.canonical_content_hash() for r in res_1.validation] == [r.canonical_content_hash() for r in res_2.validation]
    assert [r.canonical_content_hash() for r in res_1.test] == [r.canonical_content_hash() for r in res_2.test]


def test_splitter_no_leakage():
    records = make_dummy_dataset(40)
    splitter = DatasetSplitter(train_ratio=0.90, validation_ratio=0.05, test_ratio=0.05, random_seed=123)
    res = splitter.split(records)

    train_hashes = {r.canonical_content_hash() for r in res.train}
    val_hashes = {r.canonical_content_hash() for r in res.validation}
    test_hashes = {r.canonical_content_hash() for r in res.test}

    assert len(train_hashes.intersection(val_hashes)) == 0
    assert len(train_hashes.intersection(test_hashes)) == 0
    assert len(val_hashes.intersection(test_hashes)) == 0
    assert res.split_summary["leakage_detected"] is False
