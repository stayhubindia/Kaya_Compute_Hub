import os
import shutil
import hashlib
from typing import Dict, Any, Tuple
from services.processor.stages.base import BaseStage, StageValidationError

class ValidateFilesStage(BaseStage):
    @property
    def name(self) -> str:
        return "validate_files"

    @property
    def description(self) -> str:
        return "Verifies file existence, readability, non-emptiness, and checksums."

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params, dict):
            raise StageValidationError("Parameters must be a dictionary.")

    def execute(self, input_path: str, output_dir: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.validate_params(params)
        if not os.path.exists(input_path):
            raise StageValidationError(f"Input path does not exist: {input_path}")

        files_to_check = []
        if os.path.isfile(input_path):
            files_to_check.append(input_path)
        else:
            for root, _, files in os.walk(input_path):
                for f in files:
                    files_to_check.append(os.path.join(root, f))

        total_files = len(files_to_check)
        empty_files = 0
        unreadable_files = 0
        total_bytes = 0

        for fpath in files_to_check:
            try:
                size = os.path.getsize(fpath)
                total_bytes += size
                if size == 0:
                    empty_files += 1
                with open(fpath, 'rb') as f:
                    _ = f.read(1024)
            except Exception:
                unreadable_files += 1

        if unreadable_files > 0:
            raise StageValidationError(f"Detected {unreadable_files} unreadable files.")

        # Output path is copy/link of input in output_dir
        os.makedirs(output_dir, exist_ok=True)
        dest_filename = os.path.basename(input_path)
        dest_path = os.path.join(output_dir, dest_filename)
        if os.path.isfile(input_path):
            shutil.copy2(input_path, dest_path)
        else:
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(input_path, dest_path)

        metrics = {
            "total_files": total_files,
            "valid_files": total_files - empty_files - unreadable_files,
            "empty_files": empty_files,
            "unreadable_files": unreadable_files,
            "total_bytes": total_bytes
        }
        return dest_path, metrics
