"""
Raw Dataset Ingestion Loader.
Supports JSON and JSONL formats, single files and recursive directory traversal.
Guarantees non-destructive read-only handling of raw files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class RawRecord:
    data: Any
    source_file: str
    line_number: Optional[int] = None
    raw_text: Optional[str] = None


@dataclass
class LoadingError:
    source_file: str
    error_message: str
    line_number: Optional[int] = None
    raw_line: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "line_number": self.line_number,
            "error_message": self.error_message,
            "raw_line": self.raw_line[:200] if self.raw_line else None,
        }


class DatasetLoader:
    """Ingests raw JSON and JSONL data files safely and deterministically."""

    SUPPORTED_EXTENSIONS = {".json", ".jsonl"}

    def __init__(self, continue_on_error: bool = True):
        self.continue_on_error = continue_on_error

    def load_file(self, file_path: Union[str, Path]) -> Tuple[List[RawRecord], List[LoadingError]]:
        path = Path(file_path).resolve()
        if not path.is_file():
            return [], [LoadingError(source_file=str(path), error_message=f"File not found: {path}")]

        ext = path.suffix.lower()
        if ext == ".jsonl":
            return self._load_jsonl(path)
        elif ext == ".json":
            return self._load_json(path)
        else:
            return [], [LoadingError(source_file=str(path), error_message=f"Unsupported file extension: {ext}")]

    def _load_jsonl(self, path: Path) -> Tuple[List[RawRecord], List[LoadingError]]:
        records: List[RawRecord] = []
        errors: List[LoadingError] = []

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue  # Skip blank lines cleanly

                    try:
                        parsed = json.loads(stripped)
                        records.append(
                            RawRecord(
                                data=parsed,
                                source_file=str(path),
                                line_number=line_idx,
                                raw_text=stripped,
                            )
                        )
                    except json.JSONDecodeError as jde:
                        err = LoadingError(
                            source_file=str(path),
                            line_number=line_idx,
                            error_message=f"JSONDecodeError: {str(jde)}",
                            raw_line=stripped,
                        )
                        errors.append(err)
                        if not self.continue_on_error:
                            raise
        except Exception as e:
            errors.append(LoadingError(source_file=str(path), error_message=f"File read error: {str(e)}"))

        return records, errors

    def _load_json(self, path: Path) -> Tuple[List[RawRecord], List[LoadingError]]:
        records: List[RawRecord] = []
        errors: List[LoadingError] = []

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw_content = f.read().strip()
                if not raw_content:
                    return [], [LoadingError(source_file=str(path), error_message="Empty JSON file")]

                try:
                    parsed = json.loads(raw_content)
                except json.JSONDecodeError as jde:
                    return [], [
                        LoadingError(
                            source_file=str(path),
                            error_message=f"JSONDecodeError: {str(jde)}",
                            raw_line=raw_content[:200],
                        )
                    ]

                if isinstance(parsed, list):
                    for idx, item in enumerate(parsed, start=1):
                        records.append(
                            RawRecord(
                                data=item,
                                source_file=str(path),
                                line_number=idx,
                                raw_text=json.dumps(item, ensure_ascii=False),
                            )
                        )
                elif isinstance(parsed, dict):
                    records.append(
                        RawRecord(
                            data=parsed,
                            source_file=str(path),
                            line_number=1,
                            raw_text=raw_content,
                        )
                    )
                else:
                    errors.append(
                        LoadingError(
                            source_file=str(path),
                            error_message=f"Expected JSON dict or list of dicts, got {type(parsed).__name__}",
                        )
                    )
        except Exception as e:
            errors.append(LoadingError(source_file=str(path), error_message=f"File read error: {str(e)}"))

        return records, errors

    def load_directory(
        self, dir_path: Union[str, Path], recursive: bool = True
    ) -> Tuple[List[RawRecord], List[LoadingError]]:
        path = Path(dir_path).resolve()
        if not path.is_dir():
            return [], [LoadingError(source_file=str(path), error_message=f"Directory not found: {path}")]

        pattern = "**/*" if recursive else "*"
        all_records: List[RawRecord] = []
        all_errors: List[LoadingError] = []

        # Find all JSON and JSONL files sorted deterministically
        candidate_files = sorted(
            [p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        )

        for file_path in candidate_files:
            recs, errs = self.load_file(file_path)
            all_records.extend(recs)
            all_errors.extend(errs)

        return all_records, all_errors

    def load_path(self, target_path: Union[str, Path]) -> Tuple[List[RawRecord], List[LoadingError]]:
        path = Path(target_path).resolve()
        if path.is_file():
            return self.load_file(path)
        elif path.is_dir():
            return self.load_directory(path)
        else:
            return [], [LoadingError(source_file=str(path), error_message=f"Path not found: {path}")]
