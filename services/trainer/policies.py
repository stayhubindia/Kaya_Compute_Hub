import re
from typing import Dict, Any

class TrainingPolicyError(Exception):
    pass

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
    re.compile(r"\.\./", re.IGNORECASE),
]

ALLOWED_BACKENDS = ["demo", "pytorch"]
ALLOWED_OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop"]
ALLOWED_SCHEDULERS = ["linear", "cosine", "constant", "none"]
ALLOWED_PRECISION = ["fp16", "bf16", "fp32", "none"]

def _check_unsafe_values(val: Any):
    if isinstance(val, str):
        for pattern in UNSAFE_PATTERNS:
            if pattern.search(val):
                raise TrainingPolicyError(f"Unsafe pattern detected in configuration parameter: '{val[:50]}'")
    elif isinstance(val, dict):
        for k, v in val.items():
            _check_unsafe_values(k)
            _check_unsafe_values(v)
    elif isinstance(val, list):
        for item in val:
            _check_unsafe_values(item)

def validate_training_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise TrainingPolicyError("Training configuration must be a dictionary.")

    _check_unsafe_values(config)

    backend = config.get("backend", "demo").lower()
    if backend not in ALLOWED_BACKENDS:
        raise TrainingPolicyError(f"Unsupported backend '{backend}'. Allowed backends: {ALLOWED_BACKENDS}")

    model_name = config.get("model_name", "default_model")
    if not isinstance(model_name, str) or len(model_name) > 100:
        raise TrainingPolicyError("Invalid model_name string.")

    batch_size = config.get("batch_size", 32)
    if not isinstance(batch_size, int) or not (1 <= batch_size <= 1024):
        raise TrainingPolicyError("batch_size must be an integer between 1 and 1024.")

    learning_rate = config.get("learning_rate", 0.001)
    if not isinstance(learning_rate, (int, float)) or not (1e-7 <= learning_rate <= 1.0):
        raise TrainingPolicyError("learning_rate must be a number between 1e-7 and 1.0.")

    epochs = config.get("epochs", 5)
    if not isinstance(epochs, int) or not (1 <= epochs <= 1000):
        raise TrainingPolicyError("epochs must be an integer between 1 and 1000.")

    max_steps = config.get("max_steps", 10000)
    if not isinstance(max_steps, int) or not (1 <= max_steps <= 1000000):
        raise TrainingPolicyError("max_steps must be an integer between 1 and 1,000,000.")

    optimizer = config.get("optimizer", "adamw").lower()
    if optimizer not in ALLOWED_OPTIMIZERS:
        raise TrainingPolicyError(f"Unsupported optimizer '{optimizer}'. Allowed: {ALLOWED_OPTIMIZERS}")

    scheduler = config.get("scheduler", "linear").lower()
    if scheduler not in ALLOWED_SCHEDULERS:
        raise TrainingPolicyError(f"Unsupported scheduler '{scheduler}'. Allowed: {ALLOWED_SCHEDULERS}")

    precision = config.get("mixed_precision", "fp16").lower()
    if precision not in ALLOWED_PRECISION:
        raise TrainingPolicyError(f"Unsupported mixed_precision '{precision}'. Allowed: {ALLOWED_PRECISION}")

    grad_accum = config.get("gradient_accumulation_steps", 1)
    if not isinstance(grad_accum, int) or not (1 <= grad_accum <= 64):
        raise TrainingPolicyError("gradient_accumulation_steps must be an integer between 1 and 64.")

    return {
        "backend": backend,
        "model_name": model_name,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "epochs": epochs,
        "max_steps": max_steps,
        "seed": config.get("seed", 42),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "evaluation_frequency": config.get("evaluation_frequency", 1),
        "checkpoint_frequency": config.get("checkpoint_frequency", 1),
        "metric_name": config.get("metric_name", "val_loss"),
        "metric_direction": config.get("metric_direction", "minimize"),
        "mixed_precision": precision,
        "gradient_accumulation_steps": grad_accum,
    }

def validate_training_resource_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        policy = {}

    max_cpu = policy.get("max_cpu_cores", 4.0)
    if not isinstance(max_cpu, (int, float)) or not (0.5 <= max_cpu <= 16.0):
        raise TrainingPolicyError("max_cpu_cores must be between 0.5 and 16.0.")

    max_mem = policy.get("max_memory_mb", 8192)
    if not isinstance(max_mem, int) or not (512 <= max_mem <= 65536):
        raise TrainingPolicyError("max_memory_mb must be between 512MB and 65,536MB.")

    gpu_count = policy.get("requested_gpus", 0)
    if not isinstance(gpu_count, int) or not (0 <= gpu_count <= 8):
        raise TrainingPolicyError("requested_gpus must be an integer between 0 and 8.")

    timeout = policy.get("timeout_seconds", 86400)
    if not isinstance(timeout, int) or not (60 <= timeout <= 86400):
        raise TrainingPolicyError("timeout_seconds must be between 60 and 86400 (24h).")

    run_as_non_root = policy.get("run_as_non_root", True)
    if not run_as_non_root:
        raise TrainingPolicyError("Container execution requires non-root user enforcement ('run_as_non_root': true).")

    return {
        "max_cpu_cores": float(max_cpu),
        "max_memory_mb": max_mem,
        "requested_gpus": gpu_count,
        "timeout_seconds": timeout,
        "network_enabled": policy.get("network_enabled", False),
        "run_as_non_root": True,
    }
