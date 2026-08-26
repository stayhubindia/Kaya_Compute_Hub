import os
import json
import hashlib
from typing import Dict, Any
from services.processor.manifests.schema import DatasetManifestData

class ManifestWriter:
    @staticmethod
    def _calculate_checksum(filepath: str) -> str:
        hasher = hashlib.sha256()
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    hasher.update(chunk)
        elif os.path.isdir(filepath):
            for root, _, files in os.walk(filepath):
                for file in sorted(files):
                    fp = os.path.join(root, file)
                    with open(fp, 'rb') as f:
                        for chunk in iter(lambda: f.read(65536), b''):
                            hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def create_manifest(
        cls,
        output_dataset_path: str,
        dataset_id: str,
        format_detected: str,
        columns: list,
        statistics: Dict[str, Any],
        provenance: Dict[str, Any],
        destination_manifest_path: str
    ) -> DatasetManifestData:
        file_count = 0
        total_bytes = 0

        if os.path.isfile(output_dataset_path):
            file_count = 1
            total_bytes = os.path.getsize(output_dataset_path)
        elif os.path.isdir(output_dataset_path):
            for root, _, files in os.walk(output_dataset_path):
                for f in files:
                    file_count += 1
                    total_bytes += os.path.getsize(os.path.join(root, f))

        checksum = cls._calculate_checksum(output_dataset_path) if os.path.exists(output_dataset_path) else ""

        manifest_data = DatasetManifestData(
            schema_version="1.0",
            dataset_id=str(dataset_id),
            file_count=file_count,
            total_bytes=total_bytes,
            checksum=checksum,
            format=format_detected,
            columns=columns or [],
            statistics=statistics or {},
            provenance=provenance or {}
        )

        os.makedirs(os.path.dirname(destination_manifest_path), exist_ok=True)
        with open(destination_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data.to_dict(), f, indent=2)

        return manifest_data
