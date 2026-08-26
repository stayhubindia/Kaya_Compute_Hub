from dataclasses import dataclass
from typing import Dict, Any, Optional

MAX_METRIC_NAME_LENGTH = 100
MAX_METRICS_PER_STEP = 50

@dataclass
class MetricRecord:
    step: int
    epoch: int
    name: str
    value: float
    split: str = "train"

    def validate(self):
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer.")
        if not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        if not isinstance(self.name, str) or not self.name or len(self.name) > MAX_METRIC_NAME_LENGTH:
            raise ValueError(f"metric name must be a string between 1 and {MAX_METRIC_NAME_LENGTH} chars.")
        if not isinstance(self.value, (int, float)):
            raise ValueError("metric value must be a number.")
