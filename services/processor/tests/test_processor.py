import os
import shutil
import pytest
from services.processor.registry import is_stage_supported, list_supported_stages, get_stage_handler
from services.processor.pipeline import validate_pipeline_definition, execute_pipeline, StageValidationError
from services.processor.checkpoints import CheckpointManager
from services.processor.manifests import ManifestWriter
from services.processor.containers import validate_resource_policy, ResourcePolicyError, is_approved_image

def test_stage_registry_and_list():
    assert is_stage_supported("validate_files") is True
    assert is_stage_supported("inspect_schema") is True
    assert is_stage_supported("unknown_stage") is False

    stages = list_supported_stages()
    assert len(stages) >= 7
    stage_names = [s["name"] for s in stages]
    assert "deduplicate" in stage_names
    assert "split_dataset" in stage_names

def test_pipeline_validation_rejects_unsafe_code_and_urls():
    valid_pipeline = [
        {"name": "validate_files", "params": {}},
        {"name": "inspect_schema", "params": {}},
        {"name": "normalize_text", "params": {"remove_control_chars": True}},
        {"name": "generate_statistics", "params": {}}
    ]
    validate_pipeline_definition(valid_pipeline)

    unsafe_pipeline_code = [
        {"name": "validate_files", "params": {"code": "import os; os.system('rm -rf /')"}}
    ]
    with pytest.raises(StageValidationError, match="Unsafe code"):
        validate_pipeline_definition(unsafe_pipeline_code)

    unsafe_pipeline_url = [
        {"name": "validate_files", "params": {"fetch_url": "http://evil.com/payload.py"}}
    ]
    with pytest.raises(StageValidationError, match="Unsafe code or URL"):
        validate_pipeline_definition(unsafe_pipeline_url)

def test_container_resource_policy_validation():
    valid_policy = {
        "max_cpu_cores": 2.0,
        "max_memory_mb": 4096,
        "max_disk_mb": 10240,
        "timeout_seconds": 1800,
        "network_enabled": False,
        "run_as_non_root": True
    }
    policy = validate_resource_policy(valid_policy)
    assert policy["max_cpu_cores"] == 2.0

    unsafe_root_policy = dict(valid_policy, run_as_non_root=False)
    with pytest.raises(ResourcePolicyError, match="non-root"):
        validate_resource_policy(unsafe_root_policy)

    assert is_approved_image("kaya/dataset-processor:latest") is True
    assert is_approved_image("unapproved/image:latest") is False

def test_full_processor_pipeline_execution_and_checkpoint_resume(tmp_path):
    input_file = tmp_path / "sample.csv"
    input_file.write_text("id,text,label\n1,Hello World,pos\n2,Sample Text,neg\n1,Hello World,pos\n4,Another Item,pos\n", encoding="utf-8")

    stages_config = [
        {"name": "validate_files", "params": {}},
        {"name": "inspect_schema", "params": {}},
        {"name": "normalize_text", "params": {"remove_control_chars": True}},
        {"name": "deduplicate", "params": {}},
        {"name": "split_dataset", "params": {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25, "seed": 42}},
        {"name": "generate_statistics", "params": {}}
    ]

    ckpt_dir = tmp_path / "checkpoints"
    out_dir = tmp_path / "datasets"
    ckpt_mgr = CheckpointManager(base_dir=str(ckpt_dir))

    # Initial Run
    output_path, metrics, stage_log = execute_pipeline(
        run_id="run-100",
        source_path=str(input_file),
        stages_config=stages_config,
        checkpoint_mgr=ckpt_mgr,
        output_root_dir=str(out_dir)
    )

    assert os.path.exists(output_path)
    assert len(stage_log) == 6
    assert metrics["deduplicate"]["records_retained"] == 3  # 1 duplicate removed!

    # Dataset Immutability Check: Source CSV file is untouched!
    assert input_file.read_text(encoding="utf-8").startswith("id,text,label")

    # Second Run - Resumes from Checkpoints!
    output_path_2, metrics_2, stage_log_2 = execute_pipeline(
        run_id="run-100",
        source_path=str(input_file),
        stages_config=stages_config,
        checkpoint_mgr=ckpt_mgr,
        output_root_dir=str(out_dir)
    )

    assert stage_log_2[0]["status"] == "completed_from_checkpoint"

    # Test Corrupted Checkpoint Rejection
    ckpt_file = ckpt_dir / "run-100" / "validate_files.json"
    ckpt_file.write_text('{"output_path": "/nonexistent/path.csv", "checksum": "invalid"}', encoding="utf-8")

    res = ckpt_mgr.get_checkpoint("run-100", "validate_files")
    assert res is None  # Corrupted checkpoint rejected!
