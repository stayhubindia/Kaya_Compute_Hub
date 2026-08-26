"""
Knowledge Unit Selector and Feature Extractor (Phase 3.4).
Analyzes knowledge units from chunks/sections/documents, computes scientific densities,
classifies content types, and evaluates instruction synthesis suitability.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.dataset.schema import DifficultyLevel
from src.generation.models import ContentType, KnowledgeUnit
from src.ingestion.models import Equation, KnowledgeChunk, Section, Table


class KnowledgeSelector:
    """Analyzes scientific text, assigns content types, estimates difficulty, and selects knowledge units."""

    # Content type regex patterns
    DEFINITION_PATTERNS = [
        r"(?i)\b(?:is|are)\s+defined\s+as\b",
        r"(?i)\brefers\s+to\b",
        r"(?i)\bdenotes\b",
        r"(?i)\b(?:we\s+define|definition\s*[:\-])\b",
        r"(?i)\bis\s+called\s+the\b",
        r"(?i)\bknown\s+as\s+the\b",
    ]

    DERIVATION_PATTERNS = [
        r"(?i)\b(?:we\s+derive|derivation|deriving)\b",
        r"(?i)\bsubstituting\s+(?:eq|equation|\(.*?\))\b",
        r"(?i)\bdifferentiating\s+with\s+respect\s+to\b",
        r"(?i)\bintegrating\s+both\s+sides\b",
        r"(?i)\btaking\s+the\s+limit\b",
        r"(?i)\bfrom\s+(?:eq|equation)\s+\(?\d+\)?\s*,\s*we\s+obtain\b",
        r"(?i)\byields\s+the\s+(?:following|relation)\b",
    ]

    CALCULATION_PATTERNS = [
        r"\b\d+(?:\.\d+)?\s*(?:m/s|km/s|m/s\^2|eV|keV|MeV|GeV|J|kJ|N|Pa|kPa|MPa|Hz|kHz|MHz|GHz|K|°C|mol|kg|g|mg|C|V|A|mA|Ω|T|W|kW|MW)\b",
        r"(?i)\b(?:calculate|determine|compute|find\s+the\s+value|numerical\s+value)\b",
    ]

    EXPERIMENT_PATTERNS = [
        r"(?i)\b(?:experiment|experimental\s+setup|apparatus|measurement|measured|sample|detector|spectroscopy)\b",
        r"(?i)\b(?:laboratory|observed|trial|observation|calibrated)\b",
    ]

    COMPARISON_PATTERNS = [
        r"(?i)\b(?:whereas|in\s+contrast\s+to|compared\s+(?:with|to)|differs\s+from|on\s+the\s+other\s+hand)\b",
        r"(?i)\b(?:distinction\s+between|versus|vs\.?)\b",
    ]

    METHODOLOGY_PATTERNS = [
        r"(?i)\b(?:methodology|formalism|formulation|theoretical\s+framework|algorithm|protocol)\b",
    ]

    CONCLUSION_PATTERNS = [
        r"(?i)\b(?:in\s+conclusion|to\s+summarize|summary|concluding\s+remarks|we\s+have\s+demonstrated)\b",
    ]

    def __init__(
        self,
        min_token_estimate: int = 25,
        max_token_estimate: int = 3000,
    ):
        self.min_token_estimate = min_token_estimate
        self.max_token_estimate = max_token_estimate

    def analyze_and_enrich_unit(self, unit: KnowledgeUnit) -> KnowledgeUnit:
        """Computes mathematical density, content types, and difficulty estimate for a knowledge unit."""
        text = unit.text or ""
        char_count = len(text)
        if char_count == 0:
            unit.difficulty_estimate = DifficultyLevel.BEGINNER.value
            unit.selection_rationale = "Empty text unit"
            return unit

        # 1. Mathematical density computation
        math_symbols = len(re.findall(r"[$\=+\-*/^_(){}\[\]\d]|\\(?:alpha|beta|gamma|delta|omega|nabla|partial|int|sum|frac|sqrt)", text))
        latex_blocks = len(re.findall(r"\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)", text, re.DOTALL))
        math_density = min(1.0, (math_symbols + (latex_blocks * 20)) / max(1, char_count))
        unit.mathematical_density = math_density

        # 2. Extract content types
        content_types: List[ContentType] = []

        # Definition check
        def_matches = sum(len(re.findall(p, text)) for p in self.DEFINITION_PATTERNS)
        if def_matches > 0:
            content_types.append(ContentType.DEFINITION)
            unit.definition_density = min(1.0, def_matches / (len(text.split()) / 50 + 1))

        # Derivation check
        if any(re.search(p, text) for p in self.DERIVATION_PATTERNS) or (unit.equations and math_density > 0.15):
            content_types.append(ContentType.DERIVATION)

        # Equations check
        if unit.equations or latex_blocks > 0 or math_density > 0.08:
            content_types.append(ContentType.EQUATION)

        # Calculation check
        if any(re.search(p, text) for p in self.CALCULATION_PATTERNS):
            content_types.append(ContentType.CALCULATION)

        # Experiment check
        if any(re.search(p, text) for p in self.EXPERIMENT_PATTERNS):
            content_types.append(ContentType.EXPERIMENT)

        # Comparison check
        if any(re.search(p, text) for p in self.COMPARISON_PATTERNS):
            content_types.append(ContentType.COMPARISON)

        # Methodology check
        if any(re.search(p, text) for p in self.METHODOLOGY_PATTERNS):
            content_types.append(ContentType.METHODOLOGY)

        # Conclusion check
        if any(re.search(p, text) for p in self.CONCLUSION_PATTERNS):
            content_types.append(ContentType.CONCLUSION)

        # Table data check
        if unit.tables or "|---|" in text:
            content_types.append(ContentType.TABLE_DATA)

        # Default concept type if nothing specific
        if not content_types or ContentType.DEFINITION in content_types:
            content_types.append(ContentType.CONCEPT)

        # De-duplicate content types while preserving order
        unique_types = []
        for ct in content_types:
            if ct not in unique_types:
                unique_types.append(ct)
        unit.content_types = unique_types

        # 3. Difficulty estimation
        difficulty, diff_reason = self._estimate_difficulty(unit, math_density, len(unit.equations), len(text.split()))
        unit.difficulty_estimate = difficulty
        unit.difficulty_rationale = diff_reason

        # 4. Selection rationale
        unit.selection_rationale = (
            f"Content types: {[ct.value for ct in unit.content_types]}, "
            f"Math density: {unit.mathematical_density:.2f}, "
            f"Equations: {len(unit.equations)}, Tables: {len(unit.tables)}, "
            f"Difficulty: {unit.difficulty_estimate} ({diff_reason})"
        )

        return unit

    def _estimate_difficulty(
        self, unit: KnowledgeUnit, math_density: float, num_equations: int, word_count: int
    ) -> Tuple[str, str]:
        """Estimates scientific difficulty tier based on quantitative and structural indicators."""
        if ContentType.DERIVATION in unit.content_types and (num_equations >= 2 or math_density >= 0.20):
            if word_count > 300 or num_equations >= 4:
                return DifficultyLevel.EXPERT.value, "Multi-step complex mathematical derivation with rigorous formalisms"
            return DifficultyLevel.ADVANCED.value, "Mathematical derivation with algebraic steps and equations"

        if ContentType.CALCULATION in unit.content_types or (ContentType.EQUATION in unit.content_types and math_density >= 0.10):
            return DifficultyLevel.INTERMEDIATE.value, "Quantitative formulation requiring equation application or numerical substitution"

        if ContentType.METHODOLOGY in unit.content_types or ContentType.EXPERIMENT in unit.content_types:
            return DifficultyLevel.ADVANCED.value, "Scientific methodology and experimental analysis"

        if ContentType.DEFINITION in unit.content_types or ContentType.CONCEPT in unit.content_types:
            if word_count > 250:
                return DifficultyLevel.INTERMEDIATE.value, "In-depth conceptual exposition with detailed explanatory context"
            return DifficultyLevel.BEGINNER.value, "Fundamental definition or foundational conceptual summary"

        return DifficultyLevel.INTERMEDIATE.value, "Standard scientific domain topic"

    def select_units(self, units: List[KnowledgeUnit]) -> List[KnowledgeUnit]:
        """Filters and enriches knowledge units suitable for instruction generation."""
        selected: List[KnowledgeUnit] = []
        for u in units:
            word_count = len((u.text or "").split())
            if word_count < self.min_token_estimate:
                continue
            enriched = self.analyze_and_enrich_unit(u)
            selected.append(enriched)
        return selected
