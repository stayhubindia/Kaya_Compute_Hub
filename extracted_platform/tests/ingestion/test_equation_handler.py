"""
Unit tests for Equation and LaTeX Expression Handler (Phase 3.3).
"""

import pytest
from src.ingestion.equation_handler import EquationHandler


def test_equation_display_and_inline_extraction():
    text = r"""
    The Schrodinger equation in one dimension is:
    $$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \frac{\partial^2 \psi}{\partial x^2} + V(x)\psi$$
    where $\psi(x,t)$ is the wave function and $V(x)$ is the potential energy.
    Also, energy is given by:
    \[ E = \hbar \omega \]
    """

    handler = EquationHandler(doc_id="doc_eq_test")
    equations, mod_text = handler.extract_equations(text)

    assert len(equations) >= 3
    # Check display equations
    display_eqs = [eq for eq in equations if eq.equation_type == "display"]
    assert len(display_eqs) >= 2
    assert any("i\\hbar \\frac{\\partial \\psi}" in eq.latex_content for eq in display_eqs)
    assert any("E = \\hbar \\omega" in eq.latex_content for eq in display_eqs)

    # Check inline equations
    inline_eqs = [eq for eq in equations if eq.equation_type == "inline"]
    assert any("\\psi(x,t)" in eq.latex_content for eq in inline_eqs)


def test_latex_bracket_validation():
    handler = EquationHandler()
    assert handler.validate_latex_brackets(r"\frac{a}{b} + [c \cdot (d + e)]") is True
    assert handler.validate_latex_brackets(r"\frac{a}{b + [c \cdot d}") is False
    assert handler.validate_latex_brackets(r"\int_{0}^{\infty} e^{-x^2} dx") is True


def test_latex_normalization():
    handler = EquationHandler()
    raw = r"  \nabla \times \mathbf{E}   =   -\frac{\partial \mathbf{B}}{\partial t}  "
    normalized = handler.normalize_latex(raw)
    assert normalized == r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}"
