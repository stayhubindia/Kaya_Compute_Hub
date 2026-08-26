"""
Scientific Prompt Builder (Phase 3.4).
Constructs rich, natural, source-grounded technical questions and prompts for various task types.
"""

from __future__ import annotations

import re
from typing import Optional

from src.dataset.schema import TaskType
from src.generation.models import KnowledgeUnit


class ScientificPromptBuilder:
    """Builds user prompts grounded in KnowledgeUnit content without hallucinating external context."""

    def __init__(self, deterministic_seed: int = 42):
        self.deterministic_seed = deterministic_seed

    def build_prompt(self, unit: KnowledgeUnit, task_type: str) -> str:
        """Constructs a targeted prompt for the given task type and knowledge unit."""
        topic_clean = unit.subtopic or unit.topic or "the topic"
        topic_clean = topic_clean.replace("_", " ").title()

        title_clean = unit.title or topic_clean
        title_clean = re.sub(r"^(?:##?\s*|\d+\.\s*)", "", title_clean).strip()

        if task_type in [TaskType.QUESTION_ANSWERING.value]:
            if "definition" in [ct.value for ct in unit.content_types]:
                return f"What is the formal scientific definition and physical significance of {title_clean} in {topic_clean}?"
            return f"What are the foundational principles of {title_clean} within {topic_clean}?"

        elif task_type in [TaskType.EXPLANATION.value]:
            return (
                f"Explain the physical mechanisms and theoretical foundation of {title_clean} in {topic_clean}. "
                f"Provide a clear, structured explanation outlining key concepts and implications."
            )

        elif task_type in [TaskType.PROOF.value, "derivation"]:
            return (
                f"Derive the mathematical formulation for {title_clean} in {topic_clean}. "
                f"Show the step-by-step algebraic progression, state the starting equations, and define all variables."
            )

        elif task_type in [TaskType.CALCULATION.value]:
            return (
                f"Calculate the quantitative results and demonstrate the step-by-step numerical evaluation for {title_clean} "
                f"based on the relevant equations in {topic_clean}. Include all physical units."
            )

        elif task_type in [TaskType.PROBLEM_SOLVING.value]:
            return (
                f"Solve the scientific problem concerning {title_clean} in {topic_clean}. "
                f"Identify the governing equations, apply boundary conditions, and determine the exact solution."
            )

        elif task_type in [TaskType.COMPARISON.value]:
            return (
                f"Compare and contrast the distinct physical regimes or theoretical formulations discussed regarding {title_clean} in {topic_clean}. "
                f"Highlight their key differences, domain of validity, and practical implications."
            )

        elif task_type in [TaskType.DATA_INTERPRETATION.value]:
            return (
                f"Interpret the data and tabular results presented for {title_clean} in {topic_clean}. "
                f"Explain the physical significance of the observed measurements and numerical trends."
            )

        elif task_type in [TaskType.ANALYSIS.value, TaskType.SCENARIO_ANALYSIS.value]:
            return (
                f"Analyze the methodology, experimental setup, and theoretical assumptions presented for {title_clean} in {topic_clean}. "
                f"Evaluate how these factors support the reported scientific conclusions."
            )

        elif task_type in [TaskType.SUMMARIZATION.value]:
            return (
                f"Provide a concise, technically rigorous summary of the key findings and conclusions concerning {title_clean} in {topic_clean}."
            )

        elif task_type in [TaskType.REASONING.value]:
            return (
                f"Explain the causal scientific reasoning and theoretical arguments underlying {title_clean} in {topic_clean}."
            )

        # Fallback generic prompt
        return f"Discuss the core scientific principles of {title_clean} in {topic_clean}."

    def build_user_prompt(self, unit: KnowledgeUnit, task_type: str, seed: Optional[int] = None) -> str:
        return self.build_prompt(unit, task_type)


InstructionPromptBuilder = ScientificPromptBuilder
