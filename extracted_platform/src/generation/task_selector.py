"""
Task Selector for Scientific Instruction Synthesis (Phase 3.4).
Maps knowledge unit characteristics and content types to appropriate task taxonomy types,
preventing forced or irrelevant task assignments.
"""

from __future__ import annotations

from typing import List, Optional

from src.dataset.schema import TaskType
from src.generation.models import ContentType, KnowledgeUnit


class TaskSelector:
    """Selects valid scientific task types based strictly on knowledge unit capabilities."""

    def __init__(self, allow_multi_turn: bool = True, deterministic_seed: int = 42):
        self.allow_multi_turn = allow_multi_turn
        self.deterministic_seed = deterministic_seed

    def select_tasks_for_unit(self, unit: KnowledgeUnit, max_tasks: int = 3, count: Optional[int] = None) -> List[str]:
        """Returns ordered list of compatible TaskType string values for a given KnowledgeUnit."""
        limit = count if count is not None else max_tasks
        candidate_tasks: List[str] = []

        types = set(unit.content_types)

        # 1. Derivation and proof tasks
        if ContentType.DERIVATION in types:
            candidate_tasks.append(TaskType.PROOF.value)
            candidate_tasks.append(TaskType.EXPLANATION.value)
            candidate_tasks.append(TaskType.REASONING.value)

        # 2. Calculation and numerical tasks
        if ContentType.CALCULATION in types:
            candidate_tasks.append(TaskType.CALCULATION.value)
            candidate_tasks.append(TaskType.PROBLEM_SOLVING.value)

        # 3. Table and experimental data tasks
        if ContentType.TABLE_DATA in types:
            candidate_tasks.append(TaskType.DATA_INTERPRETATION.value)
            candidate_tasks.append(TaskType.ANALYSIS.value)

        # 4. Comparison tasks
        if ContentType.COMPARISON in types:
            candidate_tasks.append(TaskType.COMPARISON.value)
            candidate_tasks.append(TaskType.ANALYSIS.value)

        # 5. Experimental and methodology tasks
        if ContentType.EXPERIMENT in types or ContentType.METHODOLOGY in types:
            candidate_tasks.append(TaskType.ANALYSIS.value)
            candidate_tasks.append(TaskType.SCENARIO_ANALYSIS.value)

        # 6. Conclusion and summary tasks
        if ContentType.CONCLUSION in types:
            candidate_tasks.append(TaskType.SUMMARIZATION.value)

        # 7. Definition and conceptual tasks
        if ContentType.DEFINITION in types:
            candidate_tasks.append(TaskType.QUESTION_ANSWERING.value)
            candidate_tasks.append(TaskType.EXPLANATION.value)

        if ContentType.CONCEPT in types:
            candidate_tasks.append(TaskType.EXPLANATION.value)
            candidate_tasks.append(TaskType.QUESTION_ANSWERING.value)

        # 8. Multi-turn dialogue task
        words = len((unit.text or "").split())
        if self.allow_multi_turn and words >= 80:
            candidate_tasks.append(TaskType.MULTI_TURN.value)

        # Fallback to explanation if no task matched
        if not candidate_tasks:
            candidate_tasks.append(TaskType.EXPLANATION.value)

        # Preserve ordering, remove duplicates
        unique_tasks: List[str] = []
        for t in candidate_tasks:
            if t not in unique_tasks:
                unique_tasks.append(t)

        return unique_tasks[:limit]
