from src.dataset.schema import Role, SourceType
from src.dataset.source_adapters import (
    DocumentationAdapter,
    ExistingDatasetAdapter,
    HumanAuthoredAdapter,
    SyntheticAdapter,
    create_source_adapter,
)
from src.dataset.source_registry import SourceDefinition


def test_existing_dataset_adapter_sharegpt():
    defn = SourceDefinition(
        source_id="ext-sharegpt-v1",
        source_type=SourceType.EXISTING_DATASET,
        name="ShareGPT Archive",
        license="CC-BY-4.0",
    )
    adapter = ExistingDatasetAdapter(source_definition=defn)

    raw_item = {
        "conversations": [
            {"from": "human", "value": "How do I reverse a string in Python?"},
            {"from": "gpt", "value": "Use slice notation `s[::-1]`."},
        ],
        "domain": "programming",
        "topic": "python",
        "task_type": "coding",
        "difficulty": "beginner",
    }

    record = adapter.adapt(raw_item)
    assert len(record.messages) == 2
    assert record.messages[0].role == Role.USER
    assert record.messages[1].role == Role.ASSISTANT
    assert record.metadata.provenance.source == "ShareGPT Archive"
    assert record.metadata.provenance.source_type == "existing_dataset"
    assert record.metadata.provenance.license == "CC-BY-4.0"


def test_documentation_adapter():
    defn = SourceDefinition(
        source_id="doc-linux-sysctl",
        source_type=SourceType.DOCUMENTATION,
        name="Linux Sysctl Reference",
        license="GPLv2",
    )
    adapter = DocumentationAdapter(source_definition=defn)

    raw_doc = {
        "title": "vm.swappiness",
        "question": "What is the role of vm.swappiness in Linux memory management?",
        "answer": "vm.swappiness controls the aggressiveness of page cache eviction vs anonymous page swapping.",
        "domain": "linux_systems",
        "topic": "kernel_internals",
        "difficulty": "intermediate",
    }

    record = adapter.adapt(raw_doc)
    assert record.messages[0].content == "What is the role of vm.swappiness in Linux memory management?"
    assert record.metadata.provenance.source == "Linux Sysctl Reference"
    assert record.metadata.provenance.source_type == "documentation"


def test_human_authored_adapter():
    defn = SourceDefinition(
        source_id="internal-qa-001",
        source_type=SourceType.HUMAN_AUTHORED,
        name="Internal Curated Questions",
    )
    adapter = HumanAuthoredAdapter(source_definition=defn)

    raw_data = {
        "prompt": "How does Raft handle leader elections during network partitions?",
        "response": "Raft uses randomized election timers and majority quorums to elect leaders safely.",
        "domain": "software_engineering",
        "topic": "distributed_systems",
        "task_type": "explanation",
        "difficulty": "advanced",
    }

    record = adapter.adapt(raw_data)
    assert record.metadata.provenance.source_type == "human_authored"
    assert record.metadata.provenance.source == "Internal Curated Questions"


def test_synthetic_adapter():
    defn = SourceDefinition(
        source_id="synthetic-batch-v1",
        source_type=SourceType.SYNTHETIC,
        name="Synthetic Math Engine",
        generator="sample_test_generator",
        generator_version="1.0.0",
    )
    adapter = SyntheticAdapter(source_definition=defn)

    raw_data = {
        "prompt": "Calculate the eigenvalues of a 2x2 identity matrix.",
        "response": "The eigenvalues of a 2x2 identity matrix are lambda_1 = 1 and lambda_2 = 1.",
        "domain": "mathematics",
        "topic": "linear_algebra",
        "task_type": "reasoning",
        "difficulty": "beginner",
    }

    record = adapter.adapt(raw_data)
    assert record.metadata.provenance.source_type == "synthetic"
    assert record.metadata.provenance.generator == "sample_test_generator"
    assert record.metadata.provenance.generator_version == "1.0.0"


def test_create_source_adapter_factory():
    defn = SourceDefinition(
        source_id="test-factory",
        source_type=SourceType.DOCUMENTATION,
        name="Doc Factory Test",
    )
    adapter = create_source_adapter(defn)
    assert isinstance(adapter, DocumentationAdapter)
