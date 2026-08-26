"""
Unit tests for TaskSelector (src/generation/task_selector.py).
"""

import pytest
from src.dataset.schema import TaskType
from src.generation.models import ContentType, KnowledgeUnit
from src.generation.task_selector import TaskSelector


def test_task_selector_derivation():
    ts = TaskSelector(allow_multi_turn=False)
    unit = KnowledgeUnit(
        unit_id="ku_1",
        document_id="doc_1",
        section_id="sec_1",
        text="Derivation text.",
        content_types=[ContentType.DERIVATION, ContentType.EQUATION],
    )

    tasks = ts.select_tasks_for_unit(unit, max_tasks=3)
    assert TaskType.PROOF.value in tasks
    assert TaskType.EXPLANATION.value in tasks


def test_task_selector_calculation():
    ts = TaskSelector(allow_multi_turn=False)
    unit = KnowledgeUnit(
        unit_id="ku_2",
        document_id="doc_1",
        section_id="sec_1",
        text="Calculate energy.",
        content_types=[ContentType.CALCULATION],
    )

    tasks = ts.select_tasks_for_unit(unit, max_tasks=2)
    assert TaskType.CALCULATION.value in tasks
    assert TaskType.PROBLEM_SOLVING.value in tasks


def test_task_selector_table():
    ts = TaskSelector(allow_multi_turn=False)
    unit = KnowledgeUnit(
        unit_id="ku_3",
        document_id="doc_1",
        section_id="sec_1",
        text="Table of data.",
        content_types=[ContentType.TABLE_DATA],
    )

    tasks = ts.select_tasks_for_unit(unit, max_tasks=2)
    assert TaskType.DATA_INTERPRETATION.value in tasks


def test_task_selector_multi_turn():
    ts = TaskSelector(allow_multi_turn=True)
    long_text = " ".join(["scientific"] * 100)
    unit = KnowledgeUnit(
        unit_id="ku_4",
        document_id="doc_1",
        section_id="sec_1",
        text=long_text,
        content_types=[ContentType.CONCEPT],
    )

    tasks = ts.select_tasks_for_unit(unit, max_tasks=3)
    assert TaskType.MULTI_TURN.value in tasks
