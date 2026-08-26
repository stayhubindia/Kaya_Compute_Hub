"""
Conversation Formatter (Phase 4.1).
Converts canonical dataset messages into exact ChatML conversational training representations,
preserving turn boundaries for deterministic assistant loss masking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from src.dataset.schema import DatasetRecord, Message, Role
from src.training.tokenizer import MockQwenTokenizer, TrainingTokenizerWrapper


class TurnInfo(BaseModel):
    """Metadata describing a single conversational turn."""
    role: str
    content: str
    formatted_turn: str
    is_assistant: bool


class FormattedConversation(BaseModel):
    """Complete formatted conversational document with turn segmentation."""
    text: str
    turns: List[TurnInfo]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationFormatter:
    """
    Formats canonical DatasetRecord messages into ChatML text using the tokenizer's
    native chat template while preserving explicit turn boundaries.
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def format_record(self, record: DatasetRecord) -> FormattedConversation:
        """Format a canonical DatasetRecord into structured ChatML text and turn spans."""
        conv_dicts = [{"role": m.role.value, "content": m.content} for m in record.messages]

        # Use tokenizer's native chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_text = self.tokenizer.apply_chat_template(
                conv_dicts,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Fallback ChatML construction
            formatted_text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in conv_dicts)

        turns: List[TurnInfo] = []
        for msg in record.messages:
            turn_formatted = f"<|im_start|>{msg.role.value}\n{msg.content}<|im_end|>\n"
            turns.append(
                TurnInfo(
                    role=msg.role.value,
                    content=msg.content,
                    formatted_turn=turn_formatted,
                    is_assistant=(msg.role == Role.ASSISTANT),
                )
            )

        return FormattedConversation(
            text=formatted_text,
            turns=turns,
            metadata=record.metadata.model_dump() if hasattr(record.metadata, "model_dump") else {},
        )

    def format_messages(self, messages: List[Union[Message, Dict[str, str]]]) -> str:
        """Convenience method to format a list of messages or dictionaries."""
        conv_dicts = []
        for m in messages:
            if isinstance(m, Message):
                conv_dicts.append({"role": m.role.value, "content": m.content})
            elif isinstance(m, dict):
                conv_dicts.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            else:
                raise ValueError(f"Unsupported message type: {type(m)}")

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(conv_dicts, tokenize=False, add_generation_prompt=False)
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in conv_dicts)
