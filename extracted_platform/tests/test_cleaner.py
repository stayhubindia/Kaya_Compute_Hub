from src.dataset.cleaner import DatasetCleaner, RejectionReason


def test_cleaner_accepts_valid():
    cleaner = DatasetCleaner(min_message_chars=5)
    normalized = [
        {
            "messages": [
                {"role": "user", "content": "How do Linux signals work?"},
                {"role": "assistant", "content": "Signals are asynchronous notifications sent by the kernel to a process."},
            ],
            "metadata": {
                "domain": "linux_systems",
                "topic": "kernel_internals",
                "task_type": "explanation",
                "difficulty": "intermediate",
                "quality_score": 0.9,
                "source": "unit_test",
                "source_type": "synthetic",
                "created_at": "2026-08-11T16:00:00Z",
            },
        }
    ]
    accepted, report = cleaner.clean_records(normalized)
    assert len(accepted) == 1
    assert report.rejected_count == 0


def test_cleaner_rejects_excessively_short():
    cleaner = DatasetCleaner(min_message_chars=10)
    normalized = [
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there! How can I help you today?"},
            ],
            "metadata": {
                "domain": "programming",
                "topic": "python",
                "task_type": "explanation",
                "difficulty": "beginner",
                "quality_score": 0.9,
                "source": "unit_test",
                "source_type": "synthetic",
                "created_at": "2026-08-11T16:00:00Z",
            },
        }
    ]
    accepted, report = cleaner.clean_records(normalized)
    assert len(accepted) == 0
    assert report.rejected_count == 1
    assert report.rejected_records[0].reason == RejectionReason.EXCESSIVELY_SHORT


def test_cleaner_rejects_corrupted_unicode():
    cleaner = DatasetCleaner()
    normalized = [
        {
            "messages": [
                {"role": "user", "content": "What is this character \ufffd in my string?"},
                {"role": "assistant", "content": "It is a replacement character indicating decoding failure."},
            ],
            "metadata": {
                "domain": "programming",
                "topic": "python",
                "task_type": "explanation",
                "difficulty": "beginner",
                "quality_score": 0.9,
                "source": "unit_test",
                "source_type": "synthetic",
                "created_at": "2026-08-11T16:00:00Z",
            },
        }
    ]
    accepted, report = cleaner.clean_records(normalized)
    assert len(accepted) == 0
    assert report.rejected_records[0].reason == RejectionReason.CORRUPTED_UNICODE


def test_cleaner_rejects_template_artifacts():
    cleaner = DatasetCleaner()
    normalized = [
        {
            "messages": [
                {"role": "user", "content": "Here is a prompt: <|im_start|>user explain async"},
                {"role": "assistant", "content": "Async programming allows non-blocking execution."},
            ],
            "metadata": {
                "domain": "programming",
                "topic": "python",
                "task_type": "explanation",
                "difficulty": "beginner",
                "quality_score": 0.9,
                "source": "unit_test",
                "source_type": "synthetic",
                "created_at": "2026-08-11T16:00:00Z",
            },
        }
    ]
    accepted, report = cleaner.clean_records(normalized)
    assert len(accepted) == 0
    assert report.rejected_records[0].reason == RejectionReason.FORMATTING_ARTIFACTS
