"""
Dataset Normalizer.
Normalizes raw variations into the canonical Phase 2.1 schema without altering semantic content.
Handles Unicode NFC normalization, whitespace normalization, role mapping, and structure adapters.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.dataset.loader import RawRecord


class DatasetNormalizer:
    """Normalizes arbitrary raw conversational inputs into canonical structure."""

    ROLE_MAPPINGS = {
        "user": "user",
        "human": "user",
        "client": "user",
        "customer": "user",
        "prompter": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "ai": "assistant",
        "model": "assistant",
        "response": "assistant",
        "system": "system",
    }

    def __init__(self, default_domain: str = "general_knowledge", default_source: str = "raw_ingestion"):
        self.default_domain = default_domain
        self.default_source = default_source

    def normalize_text(self, text: Any) -> str:
        """Applies Unicode NFC normalization and clean whitespace handling."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        # 1. Unicode NFC Normalization
        text = unicodedata.normalize("NFC", text)

        # 2. Standardize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Clean line trailing whitespace while preserving code indentation
        lines = [line.rstrip() for line in text.split("\n")]

        # 4. Remove leading/trailing blank lines
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        return "\n".join(lines)

    def normalize_role(self, role: Any) -> str:
        """Maps diverse role strings to canonical 'system' | 'user' | 'assistant'."""
        if not role:
            return "user"
        clean = str(role).strip().lower()
        return self.ROLE_MAPPINGS.get(clean, clean)

    def normalize_record(self, raw_record: RawRecord) -> Dict[str, Any]:
        """
        Converts a RawRecord into a canonical dictionary payload
        ready for cleaning and validation.
        """
        data = raw_record.data
        if not isinstance(data, dict):
            return {"messages": [], "metadata": {}, "_raw_record": raw_record}

        normalized_messages: List[Dict[str, str]] = []
        raw_metadata: Dict[str, Any] = {}

        # Extract messages depending on structure pattern
        if "messages" in data and isinstance(data["messages"], list):
            # Canonical / OpenAI Chat format
            for msg in data["messages"]:
                if isinstance(msg, dict):
                    role = self.normalize_role(msg.get("role", "user"))
                    content = self.normalize_text(msg.get("content", ""))
                    normalized_messages.append({"role": role, "content": content})

        elif "conversations" in data and isinstance(data["conversations"], list):
            # ShareGPT format: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
            for msg in data["conversations"]:
                if isinstance(msg, dict):
                    role = self.normalize_role(msg.get("from", msg.get("role", "user")))
                    content = self.normalize_text(msg.get("value", msg.get("content", "")))
                    normalized_messages.append({"role": role, "content": content})

        elif "prompt" in data and "response" in data:
            # Prompt-Response pair format
            prompt_content = self.normalize_text(data.get("prompt", ""))
            response_content = self.normalize_text(data.get("response", ""))
            if "system" in data and data["system"]:
                normalized_messages.append({"role": "system", "content": self.normalize_text(data["system"])})
            normalized_messages.append({"role": "user", "content": prompt_content})
            normalized_messages.append({"role": "assistant", "content": response_content})

        elif "instruction" in data and "output" in data:
            # Alpaca format: instruction + optional input -> output
            instruction = self.normalize_text(data.get("instruction", ""))
            inp = self.normalize_text(data.get("input", ""))
            output = self.normalize_text(data.get("output", ""))

            full_prompt = f"{instruction}\n\n{inp}".strip() if inp else instruction
            normalized_messages.append({"role": "user", "content": full_prompt})
            normalized_messages.append({"role": "assistant", "content": output})

        elif "question" in data and "answer" in data:
            # QA pair format
            q = self.normalize_text(data.get("question", ""))
            a = self.normalize_text(data.get("answer", ""))
            normalized_messages.append({"role": "user", "content": q})
            normalized_messages.append({"role": "assistant", "content": a})

        # Extract and normalize metadata
        if "metadata" in data and isinstance(data["metadata"], dict):
            raw_metadata = data["metadata"].copy()
        else:
            # Collect any top-level metadata fields
            for key in ["domain", "topic", "task_type", "difficulty", "quality_score", "source", "source_type", "created_at", "source_id", "generator"]:
                if key in data:
                    raw_metadata[key] = data[key]

        # Apply default metadata values if absent
        metadata: Dict[str, Any] = {
            "domain": raw_metadata.get("domain", self.default_domain),
            "topic": raw_metadata.get("topic", "general"),
            "task_type": raw_metadata.get("task_type", "question_answering"),
            "difficulty": raw_metadata.get("difficulty", "intermediate"),
            "quality_score": raw_metadata.get("quality_score"),
            "source": raw_metadata.get("source", raw_record.source_file or self.default_source),
            "source_type": raw_metadata.get("source_type", "raw"),
            "created_at": raw_metadata.get("created_at", datetime.now(timezone.utc).isoformat()),
        }

        # Preserve extra optional provenance
        for opt_key in ["source_id", "generator", "generator_version", "dimensions"]:
            if opt_key in raw_metadata:
                metadata[opt_key] = raw_metadata[opt_key]

        return {
            "messages": normalized_messages,
            "metadata": metadata,
            "_raw_record": raw_record,
        }
