from typing import Dict, Any, Optional

def create_schedule_config(cron_expression: str, display_name: str, notebook_resource_name: str) -> Dict[str, Any]:
    """Helper to construct recurring schedule payloads for Colab Enterprise jobs."""
    return {
        "displayName": display_name,
        "cron": cron_expression,
        "target": {
            "notebookExecutionJob": {
                "displayName": f"Scheduled-{display_name}",
                "dataformRepositorySource": {
                    "dataformRepository": notebook_resource_name
                }
            }
        }
    }
