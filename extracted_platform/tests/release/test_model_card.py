"""
Unit tests for ModelCardGenerator and ReadmeGenerator (Phase 5.1).
"""

from pathlib import Path
import pytest

from src.release.manifest import ReleaseManifest, ReleaseStatus
from src.release.model_card import ModelCardGenerator, ReadmeGenerator


def test_model_card_all_16_sections():
    manifest = ReleaseManifest(
        release_id="qwen3-4b-qlora-v1.0",
        release_version="v1.0",
        status=ReleaseStatus.PLANNED,
        dataset_version="dataset-v1.0",
        training_config_hash="abc_hash",
    )
    card = ModelCardGenerator.generate_model_card(manifest)

    required_sections = [
        "1. Model Summary", "2. Base Model", "3. Adapter Type",
        "4. Intended Use", "5. Training Method", "6. Dataset",
        "7. Dataset Provenance", "8. Training Configuration",
        "9. Quantization", "10. Hardware", "11. Evaluation",
        "12. Limitations", "13. Reproducibility", "14. Version",
        "15. Integrity", "16. Change Log",
    ]
    for sec in required_sections:
        assert sec in card, f"Missing section: {sec}"


def test_readme_generator():
    manifest = ReleaseManifest(
        release_id="qwen3-4b-qlora-v1.0",
        release_version="v1.0",
        status=ReleaseStatus.PLANNED,
    )
    readme = ReadmeGenerator.generate_readme(manifest)
    assert "# qwen3-4b-qlora-v1.0" in readme
    assert "BitsAndBytesConfig" in readme
    assert "validate_release.py" in readme
