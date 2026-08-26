import json
from pathlib import Path

from src.dataset.pipeline import DatasetPipeline


def test_pipeline_provenance_preservation(tmp_path: Path):
    fixture_path = Path("data/fixtures/raw/fixture_dataset.jsonl")
    assert fixture_path.is_file()

    output_dir = tmp_path / "provenance_test_out"
    pipeline = DatasetPipeline(
        config_path="configs/dataset.yaml",
        sources_path="configs/sources.yaml",
    )

    result = pipeline.run(
        input_path=fixture_path,
        output_dir=output_dir,
        save_outputs=True,
    )

    # 1. Verify train.jsonl records have complete provenance metadata
    train_file = output_dir / "train.jsonl"
    assert train_file.is_file()

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert "metadata" in rec
            meta = rec["metadata"]
            assert "provenance" in meta
            prov = meta["provenance"]
            assert "source_type" in prov
            assert "source" in prov
            assert "created_at" in prov

    # 2. Verify source report generation
    source_json_file = output_dir / "source_report.json"
    assert source_json_file.is_file()

    with open(source_json_file, "r", encoding="utf-8") as f:
        src_stats = json.load(f)
        assert "source_type_distribution" in src_stats
        assert "source_distribution" in src_stats
        assert "license_availability" in src_stats

    # 3. Verify metrics in dataset_report.json
    report_json_file = output_dir / "dataset_report.json"
    with open(report_json_file, "r", encoding="utf-8") as f:
        report = json.load(f)
        assert "source_statistics" in report
        assert report["source_statistics"]["source_type_distribution"]
