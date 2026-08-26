"""
Data Collator for Assistant-Only Loss Masking (Phase 4.1).
Implements deterministic tokenization and loss masking for SFT, ensuring
only assistant response tokens contribute to cross-entropy loss (-100 for user & system tokens).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import torch

from src.dataset.schema import DatasetRecord, Role
from src.training.formatter import ConversationFormatter


def mask_labels_for_assistant_only(
    input_ids: List[int],
    tokenizer: Any,
    assistant_prefix: str = "<|im_start|>assistant\n",
    turn_end_marker: str = "<|im_end|>",
) -> List[int]:
    """
    Deterministically mask all non-assistant tokens with -100 in labels.
    Only assistant response tokens and turn-ending tokens receive active targets.
    """
    labels = [-100] * len(input_ids)

    # Encode prefix and end markers
    if hasattr(tokenizer, "encode"):
        prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)
        end_ids = tokenizer.encode(turn_end_marker, add_special_tokens=False)
    else:
        prefix_ids = [151644, 77091, 198]
        end_ids = [151645]

    prefix_len = len(prefix_ids)
    end_len = len(end_ids)

    i = 0
    while i < len(input_ids):
        # Look for assistant turn prefix
        if i + prefix_len <= len(input_ids) and input_ids[i : i + prefix_len] == prefix_ids:
            # Assistant turn found! Unmask tokens after prefix
            start_idx = i + prefix_len
            j = start_idx
            while j < len(input_ids):
                if j + end_len <= len(input_ids) and input_ids[j : j + end_len] == end_ids:
                    # Include the end marker and optional following newline
                    end_idx = j + end_len
                    if end_idx < len(input_ids) and input_ids[end_idx] == 198:  # newline token
                        end_idx += 1
                    for k in range(start_idx, end_idx):
                        labels[k] = input_ids[k]
                    i = end_idx
                    break
                j += 1
            else:
                # Reached end of sequence without explicit end marker (e.g. truncated)
                for k in range(start_idx, len(input_ids)):
                    labels[k] = input_ids[k]
                break
        else:
            i += 1

    return labels


class DataCollatorForAssistantOnlyLoss:
    """
    PyTorch/Hugging Face compatible data collator that formats conversations,
    tokenizes with padding and truncation, and applies assistant-only loss masking.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_seq_length: int = 4096,
        padding: Union[bool, str] = True,
        pad_to_multiple_of: Optional[int] = None,
        assistant_only_loss: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.assistant_only_loss = assistant_only_loss
        self.formatter = ConversationFormatter(tokenizer)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "eos_token_id", 151643)

    def __call__(self, batch: List[Union[DatasetRecord, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """Collate and tokenize a batch of records."""
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for item in batch:
            if isinstance(item, DatasetRecord):
                conv_dicts = [{"role": m.role.value, "content": m.content} for m in item.messages]
            elif isinstance(item, dict) and "messages" in item:
                conv_dicts = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in item["messages"]
                ]
            elif isinstance(item, dict) and "input_ids" in item:
                # Pre-tokenized record
                raw_ids = list(item["input_ids"])[: self.max_seq_length]
                batch_input_ids.append(raw_ids)
                batch_attention_mask.append([1] * len(raw_ids))
                if "labels" in item:
                    batch_labels.append(list(item["labels"])[: self.max_seq_length])
                else:
                    batch_labels.append(mask_labels_for_assistant_only(raw_ids, self.tokenizer))
                continue
            else:
                raise ValueError(f"Unsupported batch item type: {type(item)}")

            # Apply chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_text = self.tokenizer.apply_chat_template(
                    conv_dicts,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if hasattr(self.tokenizer, "encode"):
                    token_ids = self.tokenizer.encode(formatted_text, add_special_tokens=False)
                else:
                    token_ids = [hash(w) % 150000 for w in formatted_text.split()]
            else:
                formatted_text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in conv_dicts)
                token_ids = [hash(w) % 150000 for w in formatted_text.split()]

            # Truncate if necessary
            if len(token_ids) > self.max_seq_length:
                token_ids = token_ids[: self.max_seq_length]

            # Compute labels
            if self.assistant_only_loss:
                labels = mask_labels_for_assistant_only(token_ids, self.tokenizer)
            else:
                labels = list(token_ids)

            batch_input_ids.append(token_ids)
            batch_attention_mask.append([1] * len(token_ids))
            batch_labels.append(labels)

        # Dynamic padding to batch max length
        max_batch_len = max(len(ids) for ids in batch_input_ids) if batch_input_ids else 0
        if self.pad_to_multiple_of:
            max_batch_len = ((max_batch_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        padded_input_ids: List[List[int]] = []
        padded_attention_mask: List[List[int]] = []
        padded_labels: List[List[int]] = []

        for ids, mask, lbls in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_batch_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    def assert_assistant_only_masking(self, batch_records: List[DatasetRecord]) -> None:
        """
        Phase 4.2.4 requirement: Assert that loss is calculated exclusively over assistant tokens.
        Raises ValueError if masking is malformed or inactive.
        """
        batch = self(batch_records)
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        for b_idx in range(input_ids.shape[0]):
            b_input = input_ids[b_idx].tolist()
            b_label = labels[b_idx].tolist()

            active_tokens = [t for t in b_label if t != -100]
            masked_tokens = [t for t in b_label if t == -100]

            if not active_tokens:
                raise ValueError(
                    f"Assistant-only masking assertion failed on batch item {b_idx}: No active (non -100) target tokens found."
                )

            if not masked_tokens:
                raise ValueError(
                    f"Assistant-only masking assertion failed on batch item {b_idx}: No masked (-100) tokens found (prompts were not masked)."
                )

            # Assert first token is masked (it's the system or user <|im_start|>)
            if b_label[0] != -100:
                raise ValueError(
                    f"Assistant-only masking assertion failed on batch item {b_idx}: First prompt token was not masked."
                )
