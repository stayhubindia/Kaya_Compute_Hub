from typing import Dict, List, Optional
from services.processor.stages import (
    BaseStage,
    StageValidationError,
    ValidateFilesStage,
    InspectSchemaStage,
    NormalizeTextStage,
    ConvertFormatStage,
    DeduplicateStage,
    SplitDatasetStage,
    GenerateStatisticsStage
)

_STAGE_REGISTRY: Dict[str, BaseStage] = {
    "validate_files": ValidateFilesStage(),
    "inspect_schema": InspectSchemaStage(),
    "normalize_text": NormalizeTextStage(),
    "convert_format": ConvertFormatStage(),
    "deduplicate": DeduplicateStage(),
    "split_dataset": SplitDatasetStage(),
    "generate_statistics": GenerateStatisticsStage(),
}

def is_stage_supported(stage_name: str) -> bool:
    return stage_name in _STAGE_REGISTRY

def get_stage_handler(stage_name: str) -> BaseStage:
    if not is_stage_supported(stage_name):
        raise StageValidationError(f"Stage '{stage_name}' is not in the allowlisted processor stage registry.")
    return _STAGE_REGISTRY[stage_name]

def list_supported_stages() -> List[Dict[str, str]]:
    return [
        {"name": stage.name, "description": stage.description}
        for stage in _STAGE_REGISTRY.values()
    ]
