"""
Training subsystem package (Phase 4.1).
Exposes configuration, dataset loaders, tokenizers, chat formatters, collators,
QLoRA setup, preflight validators, and dry-run execution engines.
"""

from src.training.config import (
    CheckpointConfig,
    DatasetConfig,
    EvaluationConfig,
    LoraConfig,
    ModelConfig,
    QuantizationConfig,
    TokenizerConfig,
    TrainingConfig,
    TrainingHyperparameters,
)
from src.training.utils import (
    HardwareEnvironmentInfo,
    TrainingManifest,
    compute_file_sha256,
    detect_hardware_environment,
    estimate_training_schedule,
    set_seed,
)

try:
    from src.training.dataset import (
        DatasetIntegrityError,
        QwenTrainingDataset,
        SplitIsolationError,
        TrainingDatasetLoader,
    )
except ImportError:
    pass

try:
    from src.training.tokenizer import (
        MockQwenTokenizer,
        TokenLengthReport,
        TrainingTokenizerWrapper,
    )
except ImportError:
    pass

try:
    from src.training.formatter import (
        ConversationFormatter,
        FormattedConversation,
        TurnInfo,
    )
except ImportError:
    pass

try:
    from src.training.collator import (
        DataCollatorForAssistantOnlyLoss,
        mask_labels_for_assistant_only,
    )
except ImportError:
    pass

try:
    from src.training.qlora import (
        ParameterAnalysisReport,
        QLoRAConfigurator,
    )
except ImportError:
    pass

try:
    from src.training.validation import (
        GateStatus,
        PreflightGateResult,
        PreflightReport,
        TrainingPreflightValidator,
    )
except ImportError:
    pass

try:
    from src.training.trainer import (
        DryRunExecutor,
        DryRunResult,
    )
except ImportError:
    pass

try:
    from src.training.evaluation import (
        EvaluationReport,
        StratifiedMetric,
        TrainingEvaluator,
    )
except ImportError:
    pass

try:
    from src.training.checkpoint import (
        CheckpointMetadata,
        TrainingCheckpointManager,
    )
except ImportError:
    pass

try:
    from src.training.sft_trainer import (
        ProductionSFTTrainer,
        SmokeTestResult,
        TrainingTelemetry,
    )
except ImportError:
    pass

__all__ = [
    "TrainingConfig",
    "ModelConfig",
    "DatasetConfig",
    "TokenizerConfig",
    "QuantizationConfig",
    "LoraConfig",
    "TrainingHyperparameters",
    "EvaluationConfig",
    "CheckpointConfig",
    "HardwareEnvironmentInfo",
    "TrainingManifest",
    "compute_file_sha256",
    "detect_hardware_environment",
    "estimate_training_schedule",
    "set_seed",
    "DatasetIntegrityError",
    "SplitIsolationError",
    "QwenTrainingDataset",
    "TrainingDatasetLoader",
    "MockQwenTokenizer",
    "TokenLengthReport",
    "TrainingTokenizerWrapper",
    "ConversationFormatter",
    "FormattedConversation",
    "TurnInfo",
    "DataCollatorForAssistantOnlyLoss",
    "mask_labels_for_assistant_only",
    "ParameterAnalysisReport",
    "QLoRAConfigurator",
    "GateStatus",
    "PreflightGateResult",
    "PreflightReport",
    "TrainingPreflightValidator",
    "DryRunExecutor",
    "DryRunResult",
    "EvaluationReport",
    "StratifiedMetric",
    "TrainingEvaluator",
    "CheckpointMetadata",
    "TrainingCheckpointManager",
    "ProductionSFTTrainer",
    "SmokeTestResult",
    "TrainingTelemetry",
]
