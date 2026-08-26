"""
Ingestion Manifest Builder (Phase 3.3).
Constructs comprehensive cryptographic manifests documenting the provenance,
statistical distributions, and integrity checksums of generated knowledge corpora.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.training.utils import compute_file_sha256


class IngestionManifestBuilder:
    """Builds and serializes manifest.json for ingested knowledge corpora."""

    def __init__(
        self,
        execution_id: str,
        source: str,
        input_directory: str,
        seed: int = 42,
        version: str = "1.0.0",
    ):
        self.execution_id = execution_id
        self.source = source
        self.input_directory = input_directory
        self.seed = seed
        self.version = version

    def build_manifest(
        self,
        doc_count: int,
        successful_count: int,
        partial_count: int,
        failed_count: int,
        duplicate_count: int,
        total_pages: int,
        total_characters: int,
        total_sections: int,
        total_chunks: int,
        total_equations: int,
        total_tables: int,
        domain_dist: Dict[str, int],
        license_dist: Dict[str, int],
        quality_dist: Dict[str, int],
        source_file_hashes: Dict[str, str],
        output_file_hashes: Dict[str, str],
    ) -> Dict[str, Any]:
        """Assembles the complete canonical manifest payload."""
        return {
            "ingestion_version": self.version,
            "execution_id": self.execution_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "input_directory": self.input_directory,
            "seed": self.seed,
            "counts": {
                "documents_discovered": doc_count,
                "documents_successful": successful_count,
                "documents_partial": partial_count,
                "documents_failed": failed_count,
                "documents_duplicate": duplicate_count,
                "total_pages": total_pages,
                "total_characters": total_characters,
                "total_sections": total_sections,
                "total_chunks": total_chunks,
                "total_equations": total_equations,
                "total_tables": total_tables,
            },
            "distributions": {
                "domains": domain_dist,
                "licenses": license_dist,
                "quality": quality_dist,
            },
            "output_files": output_file_hashes,
            "source_files_count": len(source_file_hashes),
        }

    def write_manifest(self, manifest_data: Dict[str, Any], output_path: Union[str, Path]) -> Path:
        """Writes manifest.json to the target destination."""
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        return p
