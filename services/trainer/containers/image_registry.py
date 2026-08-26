from typing import Dict, Any, Optional

APPROVED_TRAINING_IMAGES: Dict[str, Dict[str, Any]] = {
    "kaya/ml-trainer:pytorch-2.2": {
        "name": "kaya/ml-trainer:pytorch-2.2",
        "digest": "sha256:7f8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b",
        "backend": "pytorch",
        "supported_frameworks": ["pytorch", "transformers", "accelerate"],
        "allowed_resource_range": {
            "min_cpu": 1.0,
            "max_cpu": 16.0,
            "min_memory_mb": 2048,
            "max_memory_mb": 65536,
            "max_gpus": 8,
        },
        "gpu_compatibility": True,
        "enabled": True,
    },
    "kaya/ml-trainer:demo": {
        "name": "kaya/ml-trainer:demo",
        "digest": "sha256:1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b",
        "backend": "demo",
        "supported_frameworks": ["demo-framework"],
        "allowed_resource_range": {
            "min_cpu": 0.5,
            "max_cpu": 4.0,
            "min_memory_mb": 512,
            "max_memory_mb": 8192,
            "max_gpus": 4,
        },
        "gpu_compatibility": True,
        "enabled": True,
    }
}

def is_approved_training_image(image_name: str) -> bool:
    img_meta = APPROVED_TRAINING_IMAGES.get(image_name)
    return bool(img_meta and img_meta.get("enabled", False))

def get_image_metadata(image_name: str) -> Optional[Dict[str, Any]]:
    return APPROVED_TRAINING_IMAGES.get(image_name)
