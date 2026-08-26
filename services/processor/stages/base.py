from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class StageValidationError(Exception):
    """Raised when stage parameters or inputs fail validation."""
    pass

class BaseStage(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> None:
        """Validate stage configuration parameters."""
        pass

    @abstractmethod
    def execute(self, input_path: str, output_dir: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Execute stage transformation on input_path and save result in output_dir.
        Returns tuple: (output_path, metrics_dict)
        """
        pass
