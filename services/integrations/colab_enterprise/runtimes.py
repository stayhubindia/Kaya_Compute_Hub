from typing import Dict, Any

def get_runtime_template(machine_type: str = "n1-standard-4", gpu_type: str = "NVIDIA_TESLA_T4") -> Dict[str, Any]:
    """Construct runtime hardware specification for Colab Enterprise execution environments."""
    return {
        "machineSpec": {
            "machineType": machine_type,
            "acceleratorType": gpu_type,
            "acceleratorCount": 1,
        }
    }
