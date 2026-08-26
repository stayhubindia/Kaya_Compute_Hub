"""
Canonical Schema Definition for Qwen3-4B Conversational Dataset.
Adheres to Phase 2.1 and Phase 2.3.1 specifications (configs/dataset.yaml, configs/sources.yaml).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TaskType(str, Enum):
    EXPLANATION = "explanation"
    QUESTION_ANSWERING = "question_answering"
    CODING = "coding"
    CODE_GENERATION = "code_generation"
    CODE_COMPLETION = "code_completion"
    DEBUGGING = "debugging"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    TROUBLESHOOTING = "troubleshooting"
    SYSTEM_DESIGN = "system_design"
    REASONING = "reasoning"
    COMPARISON = "comparison"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    SCENARIO_ANALYSIS = "scenario_analysis"
    DECISION_ANALYSIS = "decision_analysis"
    MULTI_TURN = "multi_turn"
    PROBLEM_SOLVING = "problem_solving"
    PROOF = "proof"
    CALCULATION = "calculation"
    DATA_INTERPRETATION = "data_interpretation"


class SourceType(str, Enum):
    EXISTING_DATASET = "existing_dataset"
    DOCUMENTATION = "documentation"
    PUBLIC_DOMAIN = "public_domain"
    LICENSED_MATERIAL = "licensed_material"
    HUMAN_AUTHORED = "human_authored"
    SYNTHETIC = "synthetic"
    INTERNAL = "internal"
    UNKNOWN = "unknown"
    # Backwards-compatible aliases from Phase 2.1/2.2
    CURATED = "curated"
    BENCHMARK = "benchmark"
    RAW = "raw"


class RecordValidationError(ValueError):
    """Raised when a dataset record fails schema or semantic validation."""
    pass


class Message(BaseModel):
    role: Role
    content: str

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, v: Any) -> Role:
        if isinstance(v, Role):
            return v
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in [r.value for r in Role]:
                return Role(clean)
        raise ValueError(f"Invalid message role '{v}'. Allowed roles: {[r.value for r in Role]}")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Message content must be a non-empty string.")
        return v

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}


class ProvenanceInfo(BaseModel):
    """Explicit provenance tracking metadata (Phase 2.3.1)."""
    source_type: str = SourceType.UNKNOWN.value
    source: str = "unknown"
    source_id: Optional[str] = None
    license: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    generator: Optional[str] = None
    generator_version: Optional[str] = None
    source_url: Optional[str] = None

    @field_validator("source_type", mode="before")
    @classmethod
    def normalize_source_type(cls, v: Any) -> str:
        if isinstance(v, SourceType):
            return v.value
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in [s.value for s in SourceType]:
                return clean
        return str(v).strip().lower() if v else SourceType.UNKNOWN.value

    @field_validator("created_at", mode="before")
    @classmethod
    def validate_created_at(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.astimezone(timezone.utc).isoformat()
        if isinstance(v, str) and v.strip():
            return v.strip()
        return datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "source_type": self.source_type,
            "source": self.source,
            "source_id": self.source_id,
            "license": self.license,
            "created_at": self.created_at,
            "generator": self.generator,
            "generator_version": self.generator_version,
        }
        if self.source_url is not None:
            data["source_url"] = self.source_url
        return data


class RecordMetadata(BaseModel):
    domain: str
    topic: str
    task_type: str
    difficulty: str
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source: str = "unknown"
    source_type: str = SourceType.UNKNOWN.value
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_id: Optional[str] = None
    license: Optional[str] = None
    generator: Optional[str] = None
    generator_version: Optional[str] = None
    dimensions: Optional[Dict[str, float]] = None
    provenance: Optional[ProvenanceInfo] = None
    mixing: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def synchronize_provenance(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = data.copy()
        prov_dict = d.get("provenance")
        if isinstance(prov_dict, ProvenanceInfo):
            prov_dict = prov_dict.to_dict()
            d["provenance"] = prov_dict

        if isinstance(prov_dict, dict):
            # Provenance sub-dictionary provided: populate top-level fallbacks
            if "source" not in d or not d["source"] or d["source"] == "unknown":
                d["source"] = prov_dict.get("source", "unknown")
            if "source_type" not in d or not d["source_type"] or d["source_type"] == SourceType.UNKNOWN.value:
                d["source_type"] = prov_dict.get("source_type", SourceType.UNKNOWN.value)
            if "created_at" not in d or not d["created_at"]:
                d["created_at"] = prov_dict.get("created_at", datetime.now(timezone.utc).isoformat())
            if "source_id" not in d or d["source_id"] is None:
                d["source_id"] = prov_dict.get("source_id")
            if "license" not in d or d["license"] is None:
                d["license"] = prov_dict.get("license")
            if "generator" not in d or d["generator"] is None:
                d["generator"] = prov_dict.get("generator")
            if "generator_version" not in d or d["generator_version"] is None:
                d["generator_version"] = prov_dict.get("generator_version")
        else:
            # Construct provenance info from top-level fields
            d["provenance"] = ProvenanceInfo(
                source_type=d.get("source_type", SourceType.UNKNOWN.value),
                source=d.get("source", "unknown"),
                source_id=d.get("source_id"),
                license=d.get("license"),
                created_at=d.get("created_at") or datetime.now(timezone.utc).isoformat(),
                generator=d.get("generator"),
                generator_version=d.get("generator_version"),
            )

        return d

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, v: Any) -> str:
        if isinstance(v, DifficultyLevel):
            return v.value
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in [d.value for d in DifficultyLevel]:
                return clean
        raise ValueError(f"Invalid difficulty '{v}'. Allowed: {[d.value for d in DifficultyLevel]}")

    @field_validator("created_at", mode="before")
    @classmethod
    def validate_created_at(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.astimezone(timezone.utc).isoformat()
        if isinstance(v, str) and v.strip():
            return v.strip()
        return datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "domain": self.domain,
            "topic": self.topic,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "quality_score": self.quality_score,
            "source": self.source,
            "source_type": self.source_type,
            "created_at": self.created_at,
        }
        if self.source_id is not None:
            data["source_id"] = self.source_id
        if self.license is not None:
            data["license"] = self.license
        if self.generator is not None:
            data["generator"] = self.generator
        if self.generator_version is not None:
            data["generator_version"] = self.generator_version
        if self.dimensions is not None:
            data["dimensions"] = self.dimensions
        if self.mixing is not None:
            data["mixing"] = self.mixing
        if self.provenance is not None:
            data["provenance"] = self.provenance.to_dict()
        else:
            data["provenance"] = ProvenanceInfo(
                source_type=self.source_type,
                source=self.source,
                source_id=self.source_id,
                license=self.license,
                created_at=self.created_at,
                generator=self.generator,
                generator_version=self.generator_version,
            ).to_dict()
        return data


class DatasetRecord(BaseModel):
    messages: List[Message]
    metadata: RecordMetadata

    @field_validator("messages")
    @classmethod
    def validate_messages_non_empty(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("Record messages cannot be empty.")
        return v

    @model_validator(mode="after")
    def validate_conversation_flow(self) -> DatasetRecord:
        msgs = self.messages
        if not msgs:
            raise ValueError("Messages list is empty.")

        # Find first non-system message
        first_non_sys_idx = 0
        if msgs[0].role == Role.SYSTEM:
            if len(msgs) == 1:
                raise ValueError("Record contains only a system message with no user turn.")
            first_non_sys_idx = 1

        if msgs[first_non_sys_idx].role != Role.USER:
            raise ValueError(f"First non-system message must be from 'user', got '{msgs[first_non_sys_idx].role}'.")

        # Check alternating user/assistant turns
        prev_role: Optional[Role] = None
        for i, msg in enumerate(msgs):
            if i == 0 and msg.role == Role.SYSTEM:
                prev_role = Role.SYSTEM
                continue
            if prev_role is not None:
                if prev_role == msg.role:
                    raise ValueError(f"Consecutive messages from same role '{msg.role}' at index {i}.")
            prev_role = msg.role

        return self

    def canonical_content_hash(self) -> str:
        """
        Computes a stable SHA-256 content hash of the conversation messages.
        Metadata is excluded so identical conversations with different metadata
        are accurately detected as content duplicates.
        """
        canonical_repr = [
            {"role": m.role.value, "content": " ".join(m.content.strip().split())}
            for m in self.messages
        ]
        serialized = json.dumps(canonical_repr, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def is_single_turn(self) -> bool:
        """True if the conversation consists of a single user-assistant exchange."""
        non_sys = [m for m in self.messages if m.role != Role.SYSTEM]
        return len(non_sys) == 2 and non_sys[0].role == Role.USER and non_sys[1].role == Role.ASSISTANT

    def turn_count(self) -> int:
        """Returns the number of assistant turns in the conversation."""
        return sum(1 for m in self.messages if m.role == Role.ASSISTANT)

    def total_chars(self) -> int:
        """Total character length across all messages."""
        return sum(len(m.content) for m in self.messages)

    def total_words(self) -> int:
        """Total word count across all messages."""
        return sum(len(m.content.split()) for m in self.messages)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetRecord:
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> DatasetRecord:
        data = json.loads(json_str)
        return cls.from_dict(data)
