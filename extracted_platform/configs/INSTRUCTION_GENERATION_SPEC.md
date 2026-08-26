# Specification: Scientific Knowledge to Instruction Dataset Generation Engine (Phase 3.4)

## 1. Overview & Objective
Phase 3.4 converts the structured knowledge corpus produced by Phase 3.3 (NPTEL & arXiv academic documents) into high-quality instruction-tuning candidate datasets (`dataset-v2` candidates).

The engine operates on extracted chunks, sections, and documents (`chunks.jsonl`, `sections.jsonl`, `documents.jsonl`), extracting verifiable scientific concepts, equations, derivations, problem scenarios, and paper analyses without fabricating information.

---

## 2. Ingestion to Instruction Pipeline

```
[Phase 3.3 Corpus]
 (chunks.jsonl, sections.jsonl, documents.jsonl)
       │
       ▼
[KnowledgeSelector] ──> Extracts Content Types, Math/Table Density, Selection Rationale
       │
       ▼
[TaskSelector] ──────> Maps Knowledge Unit to Valid Task Types (no irrelevant task forcing)
       │
       ▼
[PromptBuilder] ─────> Constructs Source-Grounded Technical User Prompts
       │
       ▼
[GenerationEngine] ──> Specialized Generators:
                         ├── ScientificGenerator (Conceptual QA, Deep Explanations, Paper Analysis)
                         ├── EquationGenerator (Derivations, Proofs, LaTeX preservation)
                         ├── ProblemGenerator (Numerical Calculations, Multi-step Problems)
                         ├── ReasoningGenerator (Step-by-step Reasoning Chains)
                         └── MultiTurnGenerator (Grounded Educational Dialogues)
       │
       ▼
[Validation Layer] ──> Source Grounding, Term Overlap, Equation Consistency, Calculation Verification
       │
       ▼
[Quality Auditor] ───> Multi-dimensional Quality Scoring (>=0.85 threshold, preferred >=0.90)
       │
       ▼
[Deduplicator] ──────> Exact SHA-256 and Near-Duplicate MinHash/Jaccard Filtering
       │
       ▼
[Provenance & Rights]> Attaches Extended Lineage, Enforces License Safety Gating
       │
       ▼
[Dataset Output] ────> Canonical DatasetRecord JSONL (`datasets/instruction_candidates/v2/`)
```

---

## 3. Knowledge Unit Analysis & Content Types

For each knowledge chunk/section, the selector determines:
- `content_types`: `concept`, `definition`, `derivation`, `equation`, `calculation`, `algorithm`, `procedure`, `experiment`, `result`, `comparison`, `historical`, `table_data`, `methodology`, `conclusion`
- `mathematical_density`: Ratio of mathematical symbols and LaTeX equations to total characters
- `equation_count`: Number of inline and display equations available
- `table_count`: Number of structured tables
- `difficulty_estimate`: `beginner`, `intermediate`, `advanced`, `expert` based on lexical complexity, formula density, and conceptual depth

---

## 4. Scientific Task Taxonomy & Example Types

1. **Conceptual QA (`question_answering`)**: Focused, grounded questions and answers on definitions and core scientific principles.
2. **Deep Explanation (`explanation`)**: Step-by-step explanation of physical, chemical, or algorithmic mechanisms.
3. **Mathematical Derivation (`derivation`, `proof`)**: Rigorous derivation of formulas preserving intermediate steps and LaTeX markup.
4. **Calculations & Problem Solving (`calculation`, `problem_solving`)**: Numerical or symbolic problems with verifiable intermediate steps, correct units, and independent check.
5. **Comparison (`comparison`)**: Structured comparison of two or more scientific theories, models, or implementations.
6. **Data & Table Interpretation (`data_interpretation`)**: Direct interpretation of experimental data tables and measured values.
7. **Research Paper Analysis (`analysis`, `summarization`)**: Examination of methodology, assumptions, experimental setup, limitations, and conclusions.
8. **Multi-Turn Dialogues (`multi_turn`)**: Socratic educational dialogues where all turns remain strictly source-grounded.

---

## 5. Strict Zero-Fabrication & Validation Invariants

1. **Source Grounding**: Terminology in assistant responses must have high lexical and semantic overlap with the source knowledge unit.
2. **Equation Integrity**: Equations must match or be rigorously derivable from source text; brackets and LaTeX markup must be syntactically valid.
3. **Calculation Integrity**: Numerical solutions must accurately compute formulas using source constants and inputs. If validation fails, the example is **rejected**.
4. **License Gating**: If the originating knowledge unit has `license == "UNKNOWN"` or restricted rights, `rights_verification_required = True` and `internal_only: True`.
5. **No Synthetic Ground Truth Assumption**: Every candidate record undergoes independent validation and quality auditing.

---

## 6. Output Artifacts & Reports

Output location: `datasets/instruction_candidates/v2/`
- `nptel_candidates.jsonl`
- `arxiv_candidates.jsonl`
- `combined_candidates.jsonl`
- `manifests/generation_manifest.json`
- `reports/generation_report.json`, `reports/generation_report.md`
- `reports/quality_report.json`, `reports/quality_report.md`
- `reports/provenance_report.json`
- `reports/rejection_report.json`
- `reports/statistics.json`
