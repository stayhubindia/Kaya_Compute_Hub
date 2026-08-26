from src.dataset.deduplicator import DatasetDeduplicator
from src.dataset.schema import DatasetRecord, Message, RecordMetadata, Role


def make_record(prompt: str, response: str, source: str = "src1") -> DatasetRecord:
    return DatasetRecord(
        messages=[
            Message(role=Role.USER, content=prompt),
            Message(role=Role.ASSISTANT, content=response),
        ],
        metadata=RecordMetadata(
            domain="programming",
            topic="python",
            task_type="coding",
            difficulty="intermediate",
            quality_score=0.9,
            source=source,
            source_type="synthetic",
            created_at="2026-08-11T16:00:00Z",
        ),
    )


def test_exact_deduplication():
    r1 = make_record("How do I sort a dictionary by value?", "Use sorted(d.items(), key=lambda x: x[1])", "src1")
    r2 = make_record("How do I sort a dictionary by value?", "Use sorted(d.items(), key=lambda x: x[1])", "src2")
    r3 = make_record("How to create a generator?", "Use yield keyword inside a function.", "src3")

    dedup = DatasetDeduplicator(enable_near_dedup=False)
    unique, report = dedup.deduplicate([r1, r2, r3])

    assert len(unique) == 2
    assert report.exact_duplicates == 1
    assert report.unique_records == 2


def test_near_deduplication():
    r1 = make_record(
        "Explain memory locking via mlockall in Linux systems and how it prevents memory swapping.",
        "mlockall locks all pages mapped into address space into physical RAM memory.",
    )
    r2 = make_record(
        "Explain memory locking with mlockall in Linux systems and how it prevents memory swapping.",
        "mlockall locks all pages mapped into address space into physical RAM memory.",
    )
    r3 = make_record(
        "What is quantum entanglement in physics?",
        "Quantum entanglement is a physical phenomenon where pairs of particles remain connected.",
    )

    dedup = DatasetDeduplicator(enable_near_dedup=True, near_duplicate_threshold=0.80)
    unique, report = dedup.deduplicate([r1, r2, r3])

    assert len(unique) == 2
    assert report.near_duplicates == 1
    assert report.unique_records == 2
