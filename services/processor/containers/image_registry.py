from typing import Dict, Any, List

APPROVED_IMAGES = {
    "kaya/dataset-processor:latest": {
        "digest": "sha256:d8f76e1a9b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e",
        "allowed_stages": ["validate_files", "inspect_schema", "normalize_text", "convert_format", "deduplicate", "split_dataset", "generate_statistics"],
        "non_root": True
    },
    "kaya/dataset-processor:v1.0.0": {
        "digest": "sha256:a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
        "allowed_stages": ["validate_files", "inspect_schema", "normalize_text", "convert_format", "deduplicate", "split_dataset", "generate_statistics"],
        "non_root": True
    }
}

def is_approved_image(image_name: str) -> bool:
    return image_name in APPROVED_IMAGES

def get_image_metadata(image_name: str) -> Dict[str, Any]:
    if not is_approved_image(image_name):
        raise ValueError(f"Image '{image_name}' is not in the approved container image registry.")
    return APPROVED_IMAGES[image_name]
