from typing import Dict, Any

DEFAULT_RESOURCE_LIMITS = {
    "max_cpu_cores": 4.0,
    "max_memory_mb": 8192,
    "max_disk_mb": 51200,
    "timeout_seconds": 3600,
    "network_enabled": False,
    "run_as_non_root": True,
}

class ResourcePolicyError(Exception):
    pass

def validate_resource_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        raise ResourcePolicyError("Resource policy must be a dictionary.")

    merged = dict(DEFAULT_RESOURCE_LIMITS)
    merged.update(policy)

    if merged["max_cpu_cores"] <= 0 or merged["max_cpu_cores"] > 16.0:
        raise ResourcePolicyError("max_cpu_cores must be between 0.1 and 16.0.")
    if merged["max_memory_mb"] < 256 or merged["max_memory_mb"] > 65536:
        raise ResourcePolicyError("max_memory_mb must be between 256MB and 64GB.")
    if merged["max_disk_mb"] < 100 or merged["max_disk_mb"] > 524288:
        raise ResourcePolicyError("max_disk_mb must be between 100MB and 500GB.")
    if merged["timeout_seconds"] < 10 or merged["timeout_seconds"] > 86400:
        raise ResourcePolicyError("timeout_seconds must be between 10 and 86400 seconds.")

    if not merged["run_as_non_root"]:
        raise ResourcePolicyError("Container must run as non-root user.")

    return merged
