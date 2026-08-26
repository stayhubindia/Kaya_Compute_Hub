from src.dataset.generator import SampleSyntheticGenerator
from src.dataset.schema import Role


def test_sample_synthetic_generator():
    generator = SampleSyntheticGenerator()
    records = generator.generate(
        domain="programming",
        topic="python",
        task_type="coding",
        difficulty="intermediate",
        number_of_examples=2,
    )

    assert len(records) == 2
    for r in records:
        assert r.metadata.domain == "programming"
        assert r.metadata.topic == "python"
        assert r.metadata.source_type == "synthetic"
        assert r.metadata.generator == "sample_test_generator"
        assert r.metadata.quality_score >= 0.90
        assert r.messages[0].role == Role.USER
        assert r.messages[1].role == Role.ASSISTANT
