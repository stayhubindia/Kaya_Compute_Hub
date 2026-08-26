from pathlib import Path

from src.dataset.pipeline import DatasetPipeline


def test_pipeline_end_to_end(tmp_path: Path):
    fixture_path = Path("data/fixtures/raw/fixture_dataset.jsonl")
    assert fixture_path.is_file(), "Fixture dataset file must exist"

    pipeline = DatasetPipeline(config_path="configs/dataset.yaml")

    output_dir = tmp_path / "processed"
    result = pipeline.run(
        input_path=fixture_path,
        output_dir=output_dir,
        save_outputs=True,
    )

    # Ingested 11 valid json lines (12th line is malformed)
    assert result.total_raw == 11
    # Cleaning rejected short, corrupt unicode, template artifact (3 rejected)
    assert result.rejected_count >= 3
    # Exact duplicate removed (1 exact duplicate)
    assert result.exact_duplicates >= 1
    # Near duplicate removed (1 near duplicate)
    assert result.near_duplicates >= 1
    # Check that accepted records exist
    assert result.accepted_count > 0

    # Verify split file creation
    assert (output_dir / "train.jsonl").is_file()
    assert (output_dir / "validation.jsonl").is_file()
    assert (output_dir / "test.jsonl").is_file()
    assert (output_dir / "dataset_report.json").is_file()
    assert (output_dir / "dataset_report.md").is_file()
    assert (output_dir / "rejection_report.json").is_file()

    # Check metrics
    assert "summary" in result.metrics
    assert "domain_distribution" in result.metrics
    assert "length_statistics" in result.metrics
