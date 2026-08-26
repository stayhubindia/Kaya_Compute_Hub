import os
import requests
from typing import Dict, Any, List, Optional
from services.integrations.google.errors import GoogleIntegrationError

VERTEX_AI_BASE = "https://{region}-aiplatform.googleapis.com/v1"

def get_allowed_projects() -> List[str]:
    raw = os.environ.get("GOOGLE_ALLOWED_PROJECTS", "")
    return [p.strip() for p in raw.split(",") if p.strip()]

def get_allowed_regions() -> List[str]:
    raw = os.environ.get("GOOGLE_ALLOWED_REGIONS", "us-central1,us-east4,europe-west1")
    return [r.strip() for r in raw.split(",") if r.strip()]

def validate_project_and_region(project_id: str, region: str) -> None:
    """Ensure GCP project and region are in the permitted allowlists."""
    allowed_projects = get_allowed_projects()
    allowed_regions = get_allowed_regions()

    if allowed_projects and project_id not in allowed_projects:
        raise GoogleIntegrationError(f"GCP Project '{project_id}' is not in the allowed project allowlist.")
    
    if allowed_regions and region not in allowed_regions:
        raise GoogleIntegrationError(f"GCP Region '{region}' is not in the allowed region allowlist.")

class ColabEnterpriseClient:
    def __init__(self, access_token: str, project_id: str, region: str = "us-central1"):
        validate_project_and_region(project_id, region)
        self.access_token = access_token
        self.project_id = project_id
        self.region = region
        self.base_url = VERTEX_AI_BASE.format(region=region)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def list_notebooks() -> List[Dict[str, Any]]:
        """List configured external Colab Enterprise notebooks."""
        # Simulated or registry-backed notebook catalog
        return [
            {
                "id": "nb-ml-preprocessing",
                "display_name": "ML Data Preprocessing & Sanitization",
                "notebook_resource_name": f"projects/{self.project_id}/locations/{self.region}/notebooks/ml-preprocessing",
                "status": "ready",
            },
            {
                "id": "nb-fine-tuning-eval",
                "display_name": "Model Evaluation & Benchmark Suite",
                "notebook_resource_name": f"projects/{self.project_id}/locations/{self.region}/notebooks/model-evaluation",
                "status": "ready",
            }
        ]

    def create_execution(self, notebook_resource_name: str, output_uri: Optional[str] = None) -> Dict[str, Any]:
        """Submit notebook execution request to Colab Enterprise Vertex AI API."""
        url = f"{self.base_url}/projects/{self.project_id}/locations/{self.region}/notebookExecutions"
        
        payload = {
            "notebookExecutionJob": {
                "displayName": f"Kaya-Run-{notebook_resource_name.split('/')[-1]}",
                "gcsOutputUri": output_uri or os.environ.get("COLAB_ENTERPRISE_DEFAULT_OUTPUT_BUCKET", "gs://kaya-outputs"),
                "dataformRepositorySource": {
                    "dataformRepository": notebook_resource_name
                }
            }
        }

        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=20)
            if not resp.ok:
                raise GoogleIntegrationError(f"Colab Enterprise Execution Submission Failed ({resp.status_code}): {resp.text}")
            return resp.json()
        except requests.RequestException as e:
            raise GoogleIntegrationError(f"Network error submitting Colab Enterprise execution: {str(e)}") from e

    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Poll notebook execution status."""
        url = f"{self.base_url}/projects/{self.project_id}/locations/{self.region}/notebookExecutionJobs/{execution_id}"
        
        try:
            resp = requests.get(url, headers=self._headers(), timeout=15)
            if not resp.ok:
                raise GoogleIntegrationError(f"Failed to poll Colab Enterprise status ({resp.status_code}): {resp.text}")
            return resp.json()
        except requests.RequestException as e:
            raise GoogleIntegrationError(f"Network error polling Colab Enterprise execution: {str(e)}") from e

    def cancel_execution(self, execution_id: str) -> Dict[str, Any]:
        """Cancel running Colab Enterprise notebook execution."""
        url = f"{self.base_url}/projects/{self.project_id}/locations/{self.region}/notebookExecutionJobs/{execution_id}:cancel"
        
        try:
            resp = requests.post(url, headers=self._headers(), timeout=15)
            if not resp.ok:
                raise GoogleIntegrationError(f"Failed to cancel Colab Enterprise execution ({resp.status_code}): {resp.text}")
            return resp.json()
        except requests.RequestException as e:
            raise GoogleIntegrationError(f"Network error cancelling Colab Enterprise execution: {str(e)}") from e
