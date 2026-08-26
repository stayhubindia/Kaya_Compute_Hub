import hashlib
from typing import Tuple

class ChecksumMismatchError(ValueError):
    """Raised when calculated file checksum does not match expected checksum."""
    pass

SUPPORTED_ALGORITHMS = {
    'sha256': hashlib.sha256,
    'sha512': hashlib.sha512,
    'md5': hashlib.md5,
}

def calculate_file_checksum(filepath: str, algorithm: str = 'sha256') -> str:
    """
    Calculates the hex digest checksum of a file using streaming 64KB chunks.
    """
    algo_lower = algorithm.lower()
    if algo_lower not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported checksum algorithm '{algorithm}'. Supported algorithms: {list(SUPPORTED_ALGORITHMS.keys())}")

    hasher = SUPPORTED_ALGORITHMS[algo_lower]()
    with open(filepath, 'rb') as f:
        while chunk := f.read(64 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()

def verify_file_checksum(filepath: str, expected_checksum: str, algorithm: str = 'sha256') -> Tuple[bool, str]:
    """
    Calculates actual checksum and verifies against expected_checksum.
    Returns (is_valid, actual_checksum).
    """
    actual = calculate_file_checksum(filepath, algorithm=algorithm)
    if expected_checksum:
        is_valid = (actual.strip().lower() == expected_checksum.strip().lower())
        return is_valid, actual
    return True, actual
