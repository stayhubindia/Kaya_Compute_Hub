"""
Instruction Generator Interface and Central Dispatcher (Phase 3.4).
Defines the model-agnostic abstract base interface for instruction generation
and dispatches requests to specialized scientific generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.dataset.schema import DatasetRecord
from src.generation.models import ExtendedProvenance, KnowledgeUnit


class InstructionGeneratorInterface(ABC):
    """Model-agnostic abstract base interface for scientific instruction generation."""

    @abstractmethod
    def generate_candidate(
        self,
        unit: KnowledgeUnit,
        task_type: str,
        prompt: str,
        seed: int = 42,
        **kwargs: Any,
    ) -> DatasetRecord:
        """Generates a canonical DatasetRecord strictly grounded in the provided KnowledgeUnit."""
        pass


class ScientificInstructionDispatcher:
    """Central dispatcher selecting and invoking the optimal scientific generator for a given task."""

    def __init__(self, version: str = "1.0.0"):
        from src.generation.equation_generator import EquationGenerator
        from src.generation.multi_turn_generator import MultiTurnGenerator
        from src.generation.problem_generator import ProblemGenerator
        from src.generation.reasoning_generator import ReasoningGenerator
        from src.generation.scientific_generator import ScientificGenerator

        self.version = version
        self.scientific_gen = ScientificGenerator(version=version)
        self.equation_gen = EquationGenerator(version=version)
        self.problem_gen = ProblemGenerator(version=version)
        self.reasoning_gen = ReasoningGenerator(version=version)
        self.multi_turn_gen = MultiTurnGenerator(version=version)

    def dispatch_and_generate(
        self,
        unit: KnowledgeUnit,
        task_type: str,
        prompt: str,
        seed: int = 42,
        **kwargs: Any,
    ) -> DatasetRecord:
        """Routes generation to the specialized generator based on task type."""
        clean_task = task_type.lower().strip()

        if clean_task in ["proof", "derivation"]:
            return self.equation_gen.generate_candidate(unit, clean_task, prompt, seed=seed, **kwargs)
        elif clean_task in ["calculation", "problem_solving"]:
            return self.problem_gen.generate_candidate(unit, clean_task, prompt, seed=seed, **kwargs)
        elif clean_task in ["reasoning"]:
            return self.reasoning_gen.generate_candidate(unit, clean_task, prompt, seed=seed, **kwargs)
        elif clean_task in ["multi_turn"]:
            return self.multi_turn_gen.generate_candidate(unit, clean_task, prompt, seed=seed, **kwargs)
        else:
            return self.scientific_gen.generate_candidate(unit, clean_task, prompt, seed=seed, **kwargs)
