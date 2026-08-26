"""
Unit tests for Section Parser (Phase 3.3).
"""

import pytest
from src.ingestion.section_parser import SectionParser


def test_academic_section_classification():
    parser = SectionParser(doc_id="test_doc")
    assert parser.classify_heading_type("Abstract") == "abstract"
    assert parser.classify_heading_type("1. Introduction") == "introduction"
    assert parser.classify_heading_type("2. Methodology and Framework") == "methodology"
    assert parser.classify_heading_type("3. Experimental Results") == "results"
    assert parser.classify_heading_type("4. Discussion") == "discussion"
    assert parser.classify_heading_type("5. Conclusion") == "conclusion"
    assert parser.classify_heading_type("References") == "references"


def test_nptel_section_classification():
    parser = SectionParser(doc_id="test_nptel")
    assert parser.classify_heading_type("Module 1: Basic Principles") == "module"
    assert parser.classify_heading_type("Lecture 4 - Blackbody Radiation") == "lecture"
    assert parser.classify_heading_type("Week 3 Overview") == "week"
    assert parser.classify_heading_type("Summary and Key Points") == "summary"
    assert parser.classify_heading_type("Assignment 1") == "assignment"


def test_text_to_sections_parsing():
    text = """
    1. Introduction
    This chapter introduces the fundamental postulates of quantum mechanics.
    Physical states are represented by rays in a complex Hilbert space.

    2. Theoretical Formulation
    Observables correspond to self-adjoint operators.
    The eigenvalues of an operator correspond to possible measurement outcomes.

    3. Summary
    In conclusion, quantum theory provides probabilistic predictions.
    """
    parser = SectionParser(doc_id="doc_qm")
    sections = parser.parse_text_into_sections(text)

    assert len(sections) == 3
    assert sections[0].title == "1. Introduction"
    assert sections[0].section_type == "introduction"
    assert "fundamental postulates" in sections[0].paragraphs[0]

    assert sections[1].title == "2. Theoretical Formulation"
    assert sections[1].section_type == "methodology"

    assert sections[2].title == "3. Summary"
    assert sections[2].section_type == "summary"
