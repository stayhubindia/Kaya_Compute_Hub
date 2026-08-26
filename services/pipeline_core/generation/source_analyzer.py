"""
Source Chunk Analyzer (Phase 3.4).
Deeply analyzes KnowledgeChunk text to extract mathematical formulas, Markdown tables,
scientific definitions, physical laws, and suitable task types for instruction generation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.generation.models import ChunkAnalysis, ScientificTaskType


class SourceChunkAnalyzer:
    """Analyzes ingested scientific chunks to determine features, complexity, and task eligibility."""

    # Math regex patterns
    DISPLAY_MATH_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
    INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)")
    NUMBERED_EQ_RE = re.compile(r"(?:Eq(?:uation)?\.?\s*\((\d+(?:\.\d+)*)\)|\((\d+(?:\.\d+)*)\)\s*$)")
    MATH_OPERATORS_RE = re.compile(r"[∂∇∫∑∏√±×÷≡≈≠≤≥∝αβγδεθλμπρστφψω\\]")

    # Table regex pattern (Markdown table syntax)
    TABLE_RE = re.compile(r"(?:^|\n)(\|.+?\|\n\|(?:\s*[-:]+[-| :]*)\|\n(?:\|.+?\|\n*)+)", re.MULTILINE)

    # Definition & Law detection patterns
    DEF_PATTERNS = [
        re.compile(r"([A-Z][a-zA-Z\s]{2,30})\s+is\s+(?:defined|known|referred to)\s+as\s+([^.\n]+)", re.IGNORECASE),
        re.compile(r"([A-Z][a-zA-Z\s]{2,30})\s+(?:represents|denotes|signifies)\s+([^.\n]+)", re.IGNORECASE),
        re.compile(r"([A-Z][a-zA-Z\s]{2,30})\s+states\s+that\s+([^.\n]+)", re.IGNORECASE),
        re.compile(r"called\s+(?:the\s+)?([a-zA-Z\s]{3,30})[.\n]", re.IGNORECASE),
    ]

    DERIVATION_PATTERNS = [
        re.compile(r"\b(?:substituting|introducing|differentiating|integrating|simplifying|expanding|yielding)\b", re.IGNORECASE),
        re.compile(r"\b(?:we\s+obtain|we\s+get|which\s+gives|leads\s+to|can\s+be\s+written\s+as)\b", re.IGNORECASE),
        re.compile(r"\b(?:from\s+equation|using\s+Eq|by\s+Gauss|by\s+Newton)\b", re.IGNORECASE),
    ]

    NUMERICAL_PATTERNS = [
        re.compile(r"\b\d+(?:\.\d+)?\s*(?:[eE][+-]?\d+)?\s*(?:m/s|km/s|m/s\^2|eV|keV|MeV|GeV|TeV|J|kJ|N|kN|Pa|kPa|MPa|Hz|kHz|MHz|GHz|K|°C|mol|kg|g|mg|V|mV|kV|A|mA|Ω|kΩ|W|kW|MW|nm|μm|mm|cm|km|rad)\b"),
        re.compile(r"\b\d+\s*[%°]\b"),
    ]

    def __init__(self):
        pass

    def analyze_chunk(
        self,
        chunk_dict: Dict[str, Any],
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> ChunkAnalysis:
        """Performs full structural, mathematical, and semantic analysis of a chunk."""
        chunk_id = chunk_dict.get("chunk_id", "unknown_chunk")
        doc_id = chunk_dict.get("document_id", "unknown_doc")
        section_id = chunk_dict.get("section_id", "unknown_section")
        text = chunk_dict.get("text", "").strip()
        token_est = chunk_dict.get("token_estimate", len(text.split()))
        domain = chunk_dict.get("domain", "science")
        topic = chunk_dict.get("topic", "physics")
        subdomain = chunk_dict.get("subdomain") or topic
        license_str = chunk_dict.get("license") or "CC-BY-NC-SA-4.0"

        # Document-level context if provided
        title = ""
        source_file = None
        source_url = None
        if doc_metadata:
            meta = doc_metadata.get("metadata", {})
            title = meta.get("title", "")
            source_file = doc_metadata.get("source_path") or meta.get("extra", {}).get("source_file")
            source_url = meta.get("extra", {}).get("source_url")

        # 1. Extract Equations
        equations = self._extract_equations(text)

        # 2. Extract Tables
        tables = self._extract_tables(text)

        # 3. Extract Definitions
        definitions = self._extract_definitions(text)

        # 4. Extract Laws / Theorems
        theorems_or_laws = self._extract_laws(text)

        # 5. Check Derivation / Step cues
        has_derivation = any(bool(p.search(text)) for p in self.DERIVATION_PATTERNS)
        has_numerical = any(bool(p.search(text)) for p in self.NUMERICAL_PATTERNS)

        # 6. Evaluate Natural Difficulty
        difficulty = self._evaluate_difficulty(
            text=text,
            equations=equations,
            tables=tables,
            has_derivation=has_derivation,
        )

        # 7. Select Suitable Tasks
        suitable_tasks = self._determine_suitable_tasks(
            text=text,
            equations=equations,
            tables=tables,
            definitions=definitions,
            has_derivation=has_derivation,
            has_numerical=has_numerical,
            difficulty=difficulty,
        )

        return ChunkAnalysis(
            chunk_id=chunk_id,
            document_id=doc_id,
            section_id=section_id,
            domain=domain,
            subdomain=subdomain,
            topic=topic,
            title=title,
            text=text,
            token_estimate=token_est,
            equations=equations,
            tables=tables,
            definitions=definitions,
            theorems_or_laws=theorems_or_laws,
            has_derivation_steps=has_derivation,
            has_numerical_values=has_numerical,
            suitable_tasks=suitable_tasks,
            natural_difficulty=difficulty,
            license=license_str,
            source_file=source_file,
            source_url=source_url,
        )

    def _extract_equations(self, text: str) -> List[str]:
        """Finds explicit display equations and standalone mathematical expressions."""
        equations: List[str] = []

        # Find $$...$$ display math
        for m in self.DISPLAY_MATH_RE.finditer(text):
            eq = m.group(1).strip()
            if eq and len(eq) > 2:
                equations.append(eq)

        # Find inline $...$ if substantial
        for m in self.INLINE_MATH_RE.finditer(text):
            eq = m.group(1).strip()
            if eq and len(eq) > 3 and ("=" in eq or self.MATH_OPERATORS_RE.search(eq)):
                equations.append(eq)

        # Standalone lines with equation numbers or calculus operators
        for line in text.split("\n"):
            line_s = line.strip()
            if line_s.startswith("$$") or line_s.endswith("$$"):
                continue
            if self.NUMBERED_EQ_RE.search(line_s) and self.MATH_OPERATORS_RE.search(line_s):
                equations.append(line_s)

        # Deduplicate while preserving order
        unique_eqs = []
        seen = set()
        for eq in equations:
            if eq not in seen:
                seen.add(eq)
                unique_eqs.append(eq)

        return unique_eqs

    def _extract_tables(self, text: str) -> List[str]:
        """Finds formatted Markdown tables within chunk text."""
        tables = []
        for m in self.TABLE_RE.finditer(text):
            tbl = m.group(1).strip()
            if tbl.count("|") >= 4:
                tables.append(tbl)
        return tables

    def _extract_definitions(self, text: str) -> List[Dict[str, str]]:
        """Extracts concept definitions from the text."""
        definitions = []
        for pat in self.DEF_PATTERNS:
            for m in pat.finditer(text):
                groups = m.groups()
                if len(groups) == 2:
                    term, meaning = groups[0].strip(), groups[1].strip()
                    if 2 < len(term) < 50 and 5 < len(meaning) < 250:
                        definitions.append({"term": term, "definition": meaning})
                elif len(groups) == 1:
                    term = groups[0].strip()
                    if 2 < len(term) < 50:
                        definitions.append({"term": term, "definition": ""})
        return definitions

    def _extract_laws(self, text: str) -> List[str]:
        """Extracts physical laws and theorems referenced in the chunk."""
        law_re = re.compile(
            r"\b(?:Newton's\s+(?:first|second|third)?\s*law|conservation\s+of\s+(?:mass|momentum|energy|charge)|Gauss(?:'s)?\s+theorem|Schrodinger\s+equation|Navier[- ]Stokes\s+equation|Euler\s+equation|Bernoulli(?:'s)?\s+equation|First\s+law\s+of\s+thermodynamics|Second\s+law\s+of\s+thermodynamics|Heisenberg\s+uncertainty\s+principle)\b",
            re.IGNORECASE,
        )
        return list(set(law_re.findall(text)))

    def _evaluate_difficulty(
        self,
        text: str,
        equations: List[str],
        tables: List[str],
        has_derivation: bool,
    ) -> str:
        """Assigns reasoning complexity based on chunk features."""
        eq_count = len(equations)
        words = len(text.split())

        if eq_count >= 3 or (eq_count >= 1 and has_derivation and words > 250):
            return "advanced"
        elif eq_count >= 1 or len(tables) >= 1 or has_derivation or words > 180:
            return "intermediate"
        elif words > 80:
            return "beginner"
        else:
            return "beginner"

    def _determine_suitable_tasks(
        self,
        text: str,
        equations: List[str],
        tables: List[str],
        definitions: List[Dict[str, str]],
        has_derivation: bool,
        has_numerical: bool,
        difficulty: str,
    ) -> List[ScientificTaskType]:
        """Identifies task types that can be strictly grounded in the source chunk."""
        tasks: List[ScientificTaskType] = []

        # Standard explanation / QA are universally grounded
        tasks.append(ScientificTaskType.EXPLANATION)
        tasks.append(ScientificTaskType.QUESTION_ANSWERING)
        tasks.append(ScientificTaskType.SUMMARIZATION)

        # Equations present
        if equations:
            tasks.append(ScientificTaskType.EQUATION_INTERPRETATION)
            if has_derivation:
                tasks.append(ScientificTaskType.DERIVATION)
                tasks.append(ScientificTaskType.PROBLEM_SOLVING)
            if has_numerical or any("=" in eq for eq in equations):
                tasks.append(ScientificTaskType.CALCULATION)
                tasks.append(ScientificTaskType.NUMERICAL_REASONING)
            if "theorem" in text.lower() or "proof" in text.lower() or "valid" in text.lower():
                tasks.append(ScientificTaskType.PROOF)

        # Tables present
        if tables:
            tasks.append(ScientificTaskType.TABLE_INTERPRETATION)
            tasks.append(ScientificTaskType.DATA_INTERPRETATION)

        # Conceptual comparisons
        if "whereas" in text.lower() or "differ" in text.lower() or "comparison" in text.lower() or "contrast" in text.lower() or "in comparison" in text.lower():
            tasks.append(ScientificTaskType.CONCEPT_COMPARISON)

        # Scientific reasoning and scenarios
        if len(text.split()) > 100:
            tasks.append(ScientificTaskType.SCIENTIFIC_REASONING)
            tasks.append(ScientificTaskType.APPLICATION)

        if "if " in text.lower() or "assume" in text.lower() or "consider" in text.lower() or "suppose" in text.lower():
            tasks.append(ScientificTaskType.SCENARIO_ANALYSIS)

        if "common misconception" in text.lower() or "contrary to" in text.lower() or "note that" in text.lower() or "must not" in text.lower():
            tasks.append(ScientificTaskType.MISCONCEPTION_CORRECTION)

        # Multi-turn conversation capability
        if len(text.split()) > 150 and (equations or definitions or len(text.split("\n\n")) >= 2):
            tasks.append(ScientificTaskType.MULTI_TURN)

        return tasks
