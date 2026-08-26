from src.dataset.loader import RawRecord
from src.dataset.normalizer import DatasetNormalizer


def test_normalizer_nfc_and_whitespace():
    norm = DatasetNormalizer()
    raw_text = "  Line 1   \r\n\r\n  Line 2 with trailing spaces    \r\n\r\n"
    cleaned = norm.normalize_text(raw_text)
    assert "\r" not in cleaned
    assert cleaned == "  Line 1\n\n  Line 2 with trailing spaces"


def test_normalizer_role_mapping():
    norm = DatasetNormalizer()
    assert norm.normalize_role("HUMAN") == "user"
    assert norm.normalize_role("gpt") == "assistant"
    assert norm.normalize_role("bot") == "assistant"
    assert norm.normalize_role("system") == "system"


def test_normalizer_sharegpt_format():
    norm = DatasetNormalizer()
    raw = RawRecord(
        data={
            "conversations": [
                {"from": "human", "value": "Write a python script"},
                {"from": "gpt", "value": "print('hello world')"},
            ]
        },
        source_file="test.json",
    )
    result = norm.normalize_record(raw)
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"


def test_normalizer_alpaca_format():
    norm = DatasetNormalizer()
    raw = RawRecord(
        data={
            "instruction": "Summarize the text",
            "input": "Long paragraph...",
            "output": "Short summary.",
        },
        source_file="test.json",
    )
    result = norm.normalize_record(raw)
    assert len(result["messages"]) == 2
    assert "Long paragraph" in result["messages"][0]["content"]
    assert result["messages"][1]["content"] == "Short summary."
