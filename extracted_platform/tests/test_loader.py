import json
from pathlib import Path

from src.dataset.loader import DatasetLoader


def test_loader_jsonl(tmp_path: Path):
    jsonl_file = tmp_path / "test.jsonl"
    lines = [
        json.dumps({"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}),
        json.dumps({"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "Good"}]}),
        "INVALID JSON LINE",
    ]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    loader = DatasetLoader(continue_on_error=True)
    records, errors = loader.load_file(jsonl_file)

    assert len(records) == 2
    assert len(errors) == 1
    assert errors[0].line_number == 3


def test_loader_json_array(tmp_path: Path):
    json_file = tmp_path / "test.json"
    data = [
        {"messages": [{"role": "user", "content": "Question 1"}, {"role": "assistant", "content": "Answer 1"}]},
        {"messages": [{"role": "user", "content": "Question 2"}, {"role": "assistant", "content": "Answer 2"}]},
    ]
    json_file.write_text(json.dumps(data), encoding="utf-8")

    loader = DatasetLoader()
    records, errors = loader.load_file(json_file)

    assert len(records) == 2
    assert len(errors) == 0


def test_loader_directory_recursive(tmp_path: Path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    f1 = tmp_path / "f1.jsonl"
    f2 = sub / "f2.jsonl"

    f1.write_text(json.dumps({"prompt": "A", "response": "B"}) + "\n")
    f2.write_text(json.dumps({"prompt": "C", "response": "D"}) + "\n")

    loader = DatasetLoader()
    records, errors = loader.load_directory(tmp_path)

    assert len(records) == 2
    assert len(errors) == 0
