from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Callable

class BaseTrainerBackend(ABC):
    @abstractmethod
    def train(
        self,
        run_id: str,
        dataset_uri: str,
        output_dir: str,
        configuration: Dict[str, Any],
        resume_from: Optional[Tuple[int, int, Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Executes model training for a run.
        Returns (model_artifact_path, final_summary_metrics).
        """
        pass
