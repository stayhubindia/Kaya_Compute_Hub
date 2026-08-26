"""
Equation and Mathematical Expression Handler (Phase 3.3).
Extracts, validates, and normalizes LaTeX, MathML, and plain-text mathematical expressions.
Preserves scientific fidelity without reducing formulas to generic placeholders.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from src.ingestion.models import Equation


class EquationHandler:
    """Detects, extracts, and normalizes mathematical equations."""

    DISPLAY_PATTERNS = [
        (r"\$\$(.*?)\$\$", "display"),
        (r"\\\[(.*?)\\\]", "display"),
        (r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", "display"),
        (r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", "display"),
        (r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}", "display"),
    ]

    INLINE_PATTERN = r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)"
    PAREN_INLINE_PATTERN = r"\\\((.*?)\\\)"

    # Numbered equation pattern: e.g. "∂ρ/∂t + ∇·(ρV) = 0   (2.2)" or "E = mc^2  (1)"
    NUMBERED_EQ_PATTERN = re.compile(
        r"^(.*?)(?:\s{2,}|\t+)\((\d+(?:\.\d+)*)\)\s*$",
        re.MULTILINE,
    )

    # Standalone math line pattern
    MATH_OPERATORS = re.compile(
        r"(?:\\(?:frac|partial|nabla|int|sum|prod|sqrt|vec|hat|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|rho|sigma|tau|phi|psi|omega)|[∂∇∫∑∏√αβγδεθλμπρστφψω]|\b(?:dm/dt|d[A-Za-z]/dt|dx/dt|dy/dt|dz/dt)\b|[=≈≡≤≥∝])"
    )

    def __init__(self, doc_id: str = "doc"):
        self.doc_id = doc_id
        self.total_equations = 0
        self.failed_equations = 0

    def normalize_latex(self, latex_str: str) -> str:
        """Cleans and standardizes raw LaTeX equation strings."""
        if not latex_str:
            return ""
        # Strip outer delimiters
        clean = latex_str.strip()
        # Remove extra whitespace inside math expression
        clean = re.sub(r"[ \t]+", " ", clean)
        # Normalize double backslashes
        clean = clean.replace("\\\\ ", "\\\\\n")
        return clean.strip()

    def validate_latex_brackets(self, latex_str: str) -> bool:
        """Basic balanced bracket validation for LaTeX expressions."""
        stack = []
        brackets = {"{": "}", "[": "]", "(": ")"}
        for ch in latex_str:
            if ch in brackets:
                stack.append(brackets[ch])
            elif ch in brackets.values():
                if not stack or stack.pop() != ch:
                    return False
        return len(stack) == 0

    def is_math_expression(self, line: str) -> bool:
        """Determines if a single line represents a standalone mathematical formula."""
        clean = line.strip()
        if not clean or len(clean) > 300 or len(clean) < 3:
            return False

        # Exclude plain text sentences ending with periods
        if clean.endswith(".") and not re.search(r"[=\d\)]\.$", clean):
            return False

        # Must have math indicators
        matches = len(self.MATH_OPERATORS.findall(clean))
        if matches >= 2 and ("=" in clean or "≈" in clean or "≡" in clean or "→" in clean or "∝" in clean):
            return True
        if matches >= 3:
            return True

        return False

    def extract_equations(self, text: str, page_number: Optional[int] = None) -> Tuple[List[Equation], str]:
        """
        Extracts mathematical equations from text and returns a list of Equation objects
        along with the text with normalized equation markers ($$ ... $$).
        """
        equations: List[Equation] = []
        seen_eq_texts: Set[str] = set()
        modified_text = text

        # 1. Extract Display LaTeX Equations ($$...$$, \begin{equation}...)
        for pat, eq_type in self.DISPLAY_PATTERNS:
            for match in re.finditer(pat, modified_text, re.DOTALL):
                raw_eq = match.group(1).strip()
                if not raw_eq or raw_eq in seen_eq_texts:
                    continue

                seen_eq_texts.add(raw_eq)
                norm_eq = self.normalize_latex(raw_eq)
                is_valid = self.validate_latex_brackets(norm_eq)
                if not is_valid:
                    self.failed_equations += 1

                self.total_equations += 1
                eq = Equation(
                    equation_id=f"{self.doc_id[:8]}:eq_{self.total_equations}",
                    latex_content=norm_eq,
                    raw_text=match.group(0),
                    equation_type=eq_type,
                    page_number=page_number,
                )
                equations.append(eq)

        # 2. Extract Numbered Equation Lines e.g. "∂ρ/∂t + ∇·(ρV) = 0   (2.2)"
        for match in self.NUMBERED_EQ_PATTERN.finditer(modified_text):
            formula_part = match.group(1).strip()
            eq_num = match.group(2).strip()
            if formula_part and self.is_math_expression(formula_part):
                if formula_part in seen_eq_texts:
                    continue
                seen_eq_texts.add(formula_part)
                self.total_equations += 1
                norm_eq = self.normalize_latex(f"{formula_part} \\tag{{{eq_num}}}")
                eq = Equation(
                    equation_id=f"{self.doc_id[:8]}:eq_{self.total_equations}",
                    latex_content=norm_eq,
                    raw_text=match.group(0),
                    equation_type="display",
                    page_number=page_number,
                )
                equations.append(eq)

        # 3. Extract Standalone Math Expression Lines
        lines = modified_text.split("\n")
        reconstructed_lines = []
        for line in lines:
            stripped = line.strip()
            # If not already wrapped in $ and matches math expression line
            if stripped and not stripped.startswith("$") and self.is_math_expression(stripped):
                if stripped not in seen_eq_texts:
                    seen_eq_texts.add(stripped)
                    self.total_equations += 1
                    norm_eq = self.normalize_latex(stripped)
                    eq = Equation(
                        equation_id=f"{self.doc_id[:8]}:eq_{self.total_equations}",
                        latex_content=norm_eq,
                        raw_text=stripped,
                        equation_type="display",
                        page_number=page_number,
                    )
                    equations.append(eq)
                    # Wrap in standard markdown math display block
                    reconstructed_lines.append(f"\n$${norm_eq}$$\n")
                    continue
            reconstructed_lines.append(line)

        modified_text = "\n".join(reconstructed_lines)

        # 4. Extract Inline Equations ($...$)
        for match in re.finditer(self.INLINE_PATTERN, modified_text):
            raw_eq = match.group(1).strip()
            if not raw_eq or len(raw_eq) < 1 or raw_eq in seen_eq_texts:
                continue

            seen_eq_texts.add(raw_eq)
            self.total_equations += 1
            norm_eq = self.normalize_latex(raw_eq)
            eq = Equation(
                equation_id=f"{self.doc_id[:8]}:eq_{self.total_equations}",
                latex_content=norm_eq,
                raw_text=match.group(0),
                equation_type="inline",
                page_number=page_number,
            )
            equations.append(eq)

        return equations, modified_text
