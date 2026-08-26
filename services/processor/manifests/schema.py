from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

@dataclass
class DatasetManifestData:
    schema_version: str = "1.0"
    dataset_id: str = ""
    file_count: int = 0
    total_bytes: int = 0
    checksum: str = ""
    format: str = ""
    columns: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
