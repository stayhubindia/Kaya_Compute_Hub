import pytest
from pathlib import Path

from src.dataset.schema import SourceType
from src.dataset.source_registry import SourceDefinition, SourceRegistry


def test_source_registry_register_and_lookup():
    registry = SourceRegistry()
    defn = SourceDefinition(
        source_id="test-source-1",
        source_type=SourceType.HUMAN_AUTHORED,
        name="Test Human Source",
        version="1.0",
        license="MIT",
    )
    registry.register_source(defn)

    found = registry.lookup_source("test-source-1")
    assert found is not None
    assert found.name == "Test Human Source"
    assert found.source_type == "human_authored"
    assert found.license == "MIT"


def test_source_registry_unknown_source():
    registry = SourceRegistry()
    assert registry.lookup_source("non-existent-source") is None
    with pytest.raises(KeyError):
        registry.get_source("non-existent-source")


def test_source_registry_duplicate_registration_rejected():
    registry = SourceRegistry()
    defn1 = SourceDefinition(
        source_id="src-dup",
        source_type=SourceType.DOCUMENTATION,
        name="Source 1",
    )
    defn2 = SourceDefinition(
        source_id="src-dup",
        source_type=SourceType.DOCUMENTATION,
        name="Source 2",
    )
    registry.register_source(defn1)
    with pytest.raises(ValueError, match="Duplicate source_id"):
        registry.register_source(defn2, overwrite=False)

    # With overwrite=True
    registry.register_source(defn2, overwrite=True)
    assert registry.lookup_source("src-dup").name == "Source 2"


def test_source_registry_invalid_source():
    with pytest.raises(ValueError):
        SourceDefinition(source_id="", name="Invalid", source_type=SourceType.INTERNAL)


def test_source_registry_load_manifest():
    manifest_path = Path("configs/sources.yaml")
    assert manifest_path.is_file(), "configs/sources.yaml must exist"

    registry = SourceRegistry.from_manifest(manifest_path)
    sources = registry.list_sources()

    assert len(sources) >= 4
    internal_py = registry.lookup_source("internal-python-v1")
    assert internal_py is not None
    assert internal_py.source_type == "human_authored"

    synth_pilot = registry.lookup_source("synthetic-pilot-v1")
    assert synth_pilot is not None
    assert synth_pilot.generator == "sample_test_generator"
