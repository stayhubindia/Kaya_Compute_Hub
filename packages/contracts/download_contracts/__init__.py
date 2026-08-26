from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

class DownloadStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RESOLVING = "resolving"
    DOWNLOADING = "downloading"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class ChecksumAlgorithm(str, Enum):
    SHA256 = "sha256"
    SHA512 = "sha512"
    MD5 = "md5"

@dataclass
class DownloadRequestContract:
    url: str
    expected_checksum: Optional[str] = None
    checksum_algorithm: ChecksumAlgorithm = ChecksumAlgorithm.SHA256
    extract: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProviderMetadataContract:
    provider_name: str
    filename: Optional[str] = None
    content_type: Optional[str] = None
    expected_size_bytes: Optional[int] = None
    supports_range: bool = False
