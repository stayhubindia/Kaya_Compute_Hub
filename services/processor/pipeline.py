import os
import re
from typing import Dict, Any, List, Tuple, Callable, Optional
from services.processor.registry import is_stage_supported, get_stage_handler
from services.processor.stages.base import StageValidationError
from services.processor.checkpoints.manager import CheckpointManager

UNSAFE_PATTERNS = [
    re.compile(r"import\s+os", re.IGNORECASE),
    re.compile(r"import\s+sys", re.IGNORECASE),
    re.compile(r"subprocess", re.IGNORECASE),
    re.compile(r"exec\(", re.IGNORECASE),
    re.compile(r"eval\(", re.IGNORECASE),
    re.compile(r"__import__", re.IGNORECASE),
    re.compile(r"system\(", re.IGNORECASE),
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"file://", re.IGNORECASE),
]

def _check_unsafe_values(val: Any):
    if isinstance(val, str):
        for pattern in UNSAFE_PATTERNS:
            if pattern.search(val):
                raise StageValidationError(f"Unsafe code or URL pattern detected in parameter value: '{val[:50]}'")
    elif isinstance(val, dict):
        for k, v in val.items():
            _check_unsafe_values(k)
            _check_unsafe_values(v)
    elif isinstance(val, list):
        for item in val:
            _check_unsafe_values(item)

def validate_pipeline_definition(stages_config: List[Dict[str, Any]]) -> None:
    if not isinstance(stages_config, list) or not stages_config:
        raise StageValidationError("Pipeline definition must be a non-empty list of stage configurations.")

    seen_stages = set()
    for idx, stage_item in enumerate(stages_config):
        if not isinstance(stage_item, dict):
            raise StageValidationError(f"Stage configuration at index {idx} must be a object/dictionary.")
        stage_name = stage_item.get("name")
        if not stage_name or not is_stage_supported(stage_name):
            raise StageValidationError(f"Unknown or unsupported stage name '{stage_name}' at index {idx}.")

        params = stage_item.get("params", {})
        _check_unsafe_values(params)

        handler = get_stage_handler(stage_name)
        handler.validate_params(params)
        seen_stages.add(stage_name)

def execute_pipeline(
    run_id: str,
    source_path: str,
    stages_config: List[Dict[str, Any]],
    checkpoint_mgr: CheckpointManager,
    output_root_dir: str = "storage/datasets",
    progress_callback: Optional[Callable[[int, int, str, float], None]] = None
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:

    validate_pipeline_definition(stages_config)

    current_input_path = source_path
    accumulated_metrics: Dict[str, Any] = {}
    stage_events_log: List[Dict[str, Any]] = []

    total_stages = len(stages_config)
    run_output_dir = os.path.join(output_root_dir, str(run_id))
    os.makedirs(run_output_dir, exist_ok=True)

    for idx, stage_item in enumerate(stages_config):
        stage_name = stage_item["name"]
        params = stage_item.get("params", {})
        handler = get_stage_handler(stage_name)

        pct = round(((idx) / total_stages) * 100, 2)
        if progress_callback:
            progress_callback(idx, total_stages, stage_name, pct)

        # Check for checkpoint resume
        checkpoint = checkpoint_mgr.get_checkpoint(run_id, stage_name)
        if checkpoint:
            out_path, stage_metrics = checkpoint
            current_input_path = out_path
            accumulated_metrics[stage_name] = stage_metrics
            stage_events_log.append({
                "stage_name": stage_name,
                "status": "completed_from_checkpoint",
                "output_uri": out_path,
                "metrics": stage_metrics
            })
            continue

        # Execute stage
        stage_output_dir = os.path.join(run_output_dir, f"stage_{idx}_{stage_name}")
        out_path, stage_metrics = handler.execute(current_input_path, stage_output_dir, params)

        # Save checkpoint
        checkpoint_mgr.save_checkpoint(run_id, stage_name, out_path, stage_metrics)

        current_input_path = out_path
        accumulated_metrics[stage_name] = stage_metrics
        stage_events_log.append({
            "stage_name": stage_name,
            "status": "completed",
            "output_uri": out_path,
            "metrics": stage_metrics
        })

    if progress_callback:
        progress_callback(total_stages, total_stages, "succeeded", 100.0)

    return current_input_path, accumulated_metrics, stage_events_log
