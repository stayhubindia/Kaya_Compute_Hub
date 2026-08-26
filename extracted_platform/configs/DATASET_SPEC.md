# Technical Dataset Specification — Qwen3-4B QLoRA Fine-Tuning

**Phase**: 2.1 & 2.3.1 — Dataset Specification & Source Provenance Architecture  
**Model**: `Qwen/Qwen3-4B-Base`  
**Training Method**: QLoRA (4-bit quantization, LoRA adapter target)  
**Configuration Reference**: [`configs/dataset.yaml`](./dataset.yaml), [`configs/sources.yaml`](./sources.yaml)

---

## 1. Dataset Purpose

The primary objective of this dataset specification is to build a robust, high-signal, multi-domain instruction tuning dataset for training an advanced technical and general-purpose reasoning assistant (`Qwen3-4B-Base`).

The model is designed to excel across a wide spectrum of software engineering, systems programming, cybersecurity, networking, mathematics, scientific computing, cognitive reasoning, and human communication without artificial domain restrictions. Data inclusion is governed strictly by engineering standards: factual correctness, high signal-to-noise ratio, clarity, and depth.

---

## 2. Supported Domains & Taxonomy

The taxonomy is modular and extensible. The initial domain taxonomy covers 13 core areas with baseline target distributions:

| Domain | Target % | Description & Scope | Key Subtopics |
| :--- | :---: | :--- | :--- |
| **`programming`** | 18.2% | Idiomatic syntax, language features, algorithms, data structures | Python, JavaScript, C/C++, Dart/Flutter, Go, Rust, Data Structures, Algorithms |
| **`software_engineering`** | 9.1% | Software architecture, maintainability, design patterns, testing | System Design, API Design, Design Patterns, Refactoring, Testing, CI/CD |
| **`cybersecurity`** | 13.6% | Security fundamentals, binary analysis, cryptography, secure code | Cryptography, Reverse Engineering, Vulnerability Analysis, Secure Coding, Forensics |
| **`linux_systems`** | 9.1% | OS internals, shell automation, kernel subsystems, administration | Bash Scripting, SysAdmin, Kernel Internals, Process/Memory Management, systemd |
| **`networking`** | 7.3% | Network stack protocols, socket programming, packet analysis | Protocols, TCP/IP, Routing, Sockets, Packet Analysis, HTTP/DNS |
| **`ai_ml`** | 7.3% | Machine learning theory, deep learning models, training pipelines | ML, Deep Learning, NLP, Computer Vision, Model Evaluation, Fine-Tuning |
| **`mathematics`** | 4.5% | Computationally relevant mathematical foundations | Linear Algebra, Calculus, Probability & Statistics, Discrete Math, Optimization |
| **`science`** | 4.5% | Physical and computational sciences | Physics, Chemistry, Biology, Scientific Computing |
| **`psychology`** | 4.5% | Cognitive science, mental models, biases, behavioral mechanisms | Cognitive Psychology, Behavioral Psychology, Cognitive Biases, Perception |
| **`human_behavior`** | 4.5% | Interpersonal dynamics, communication frameworks, negotiation | Social Dynamics, Communication, Negotiation, Group Behavior |
| **`reasoning`** | 6.4% | First-principles deduction, root-cause analysis, decision theory | Logical Deduction, Problem Solving, Root-Cause Analysis, Decision Theory |
| **`technology`** | 4.5% | Modern computing systems, cloud infrastructure, hardware | Hardware Architecture, Distributed Systems, Cloud Infrastructure |
| **`general_knowledge`** | 6.5% | Broad technical history, interdisciplinary synthesis | History of Tech, Interdisciplinary Overviews |

---

## 3. Task Types

The dataset incorporates diverse task types to build versatile conversational and analytical capabilities:

- **`explanation`**: In-depth theoretical or practical explanations of concepts.
- **`question_answering`**: Direct, accurate responses to technical queries.
- **`coding`**: General code authoring from specifications.
- **`code_generation`**: Generating standalone scripts, modules, or templates.
- **`code_completion`**: Completing partial snippets or filling missing functionality.
- **`debugging`**: Identifying bugs, race conditions, edge cases, and prescribing fixes.
- **`code_review`**: Structural, performance, and security auditing of code.
- **`refactoring`**: Improving code clarity, modularity, and runtime complexity without changing external behavior.
- **`troubleshooting`**: System/network diagnostics, error log interpretation, and resolution steps.
- **`system_design`**: Designing high-level distributed architectures, schemas, and API contracts.
- **`reasoning`**: Step-by-step logical deduction and analytical problem solving.
- **`comparison`**: Evaluating trade-offs across frameworks, algorithms, or architectural patterns.
- **`classification`**: Categorizing code snippets, log events, or conceptual inputs.
- **`summarization`**: Distilling complex technical documentation or incident reports.
- **`analysis`**: Deep architectural or algorithmic breakdown.
- **`scenario_analysis`**: Evaluating complex hypothetical technical scenarios and failure modes.
- **`decision_analysis`**: Structured evaluation matrices for engineering decision-making.
- **`multi_turn`**: Multi-step iterative refinement, conversational context tracking, and follow-ups.

---

## 4. Difficulty Levels

To ensure the model develops foundational grounding while mastering advanced concepts, dataset difficulty is stratified into 4 levels:

- **`beginner` (25% target)**: Syntax basics, core definitions, standard library usage, straightforward queries.
- **`intermediate` (40% target)**: Multi-component scripts, system configuration, idiomatic patterns, API integration.
- **`advanced` (25% target)**: Concurrency, kernel tuning, security auditing, distributed algorithms, complex optimizations.
- **`expert` (10% target)**: Deep reverse engineering, protocol internals, custom kernel modules, advanced cryptographic implementations.

---

## 5. Message Format

Conversational sequences follow the standard chat template compatible with Qwen tokenizers:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "User prompt or technical question"
    },
    {
      "role": "assistant",
      "content": "Comprehensive, accurate response"
    }
  ]
}
```

Multi-turn interactions extend the `messages` array sequentially (`user` -> `assistant` -> `user` -> `assistant` ...). Optional system prompts use `"role": "system"`.

---

## 6. Metadata & Provenance Schema

To preserve clean training data and guarantee auditability, record metadata and explicit provenance are stored alongside the conversational payload in JSONL format:

```json
{
  "messages": [...],
  "metadata": {
    "domain": "cybersecurity",
    "topic": "cryptography",
    "task_type": "explanation",
    "difficulty": "advanced",
    "quality_score": 0.94,
    "source": "Internal Python Examples",
    "source_type": "human_authored",
    "created_at": "2026-08-11T16:40:00Z",
    "source_id": "internal-python-v1",
    "license": null,
    "generator": null,
    "generator_version": null,
    "provenance": {
      "source_type": "human_authored",
      "source": "Internal Python Examples",
      "source_id": "example-000123",
      "license": null,
      "created_at": "2026-08-11T16:40:00Z",
      "generator": null,
      "generator_version": null
    }
  }
}
```

For synthetic generation batches:

```json
{
  "provenance": {
    "source_type": "synthetic",
    "source": "synthetic_generator",
    "source_id": "generation-batch-001",
    "license": null,
    "created_at": "2026-08-11T16:40:00Z",
    "generator": "sample_test_generator",
    "generator_version": "1.0.0"
  }
}
```

---

## 7. Quality Criteria & Engineering Standards

All dataset samples must meet strict quality standards before entering training sets:

1. **Quality Score**:
   - Minimum threshold: `0.85`
   - Preferred threshold: `≥ 0.90`
2. **Quality Evaluation Dimensions**:
   - `correctness`: Factual accuracy and syntactic validity of all code and statements.
   - `relevance`: Direct adherence to user instruction without evasive or redundant commentary.
   - `clarity`: Structured, clean formatting with clear markdown headings and code blocks.
   - `completeness`: Thorough resolution of all facets of the prompt.
   - `technical_accuracy`: Idiomatic conventions, correct signatures, and sound algorithmic complexity.
   - `reasoning_quality`: Coherent, logically sound deduction and rationale.
3. **Engineering Guards**:
   - **PII Protection**: Regex and pattern-based filtering for sensitive credentials and personal data.
   - **Duplicate Elimination**: Deterministic SHA-256 canonical conversation hashing and n-gram Jaccard similarity.
   - **Provenance Tracking**: Strict preservation of origin source, generator, and timestamps.

---

## 8. Dataset Split

Dataset distribution maintains complete test isolation:

- **`train`**: `90%` — Used exclusively for parameter updates in QLoRA fine-tuning.
- **`validation`**: `5%` — Used for evaluation loss monitoring and early stopping during training.
- **`test`**: `5%` — Strict held-out evaluation set for benchmark scoring and generalization testing.

---

## 9. Storage Architecture

Paths are managed via configuration and map directly to Google Drive mounted in Colab:

- **Raw Ingestion**: `/content/drive/MyDrive/GoogleColab/AI/Qwen3/datasets/raw`
- **Processed Batches**: `/content/drive/MyDrive/GoogleColab/AI/Qwen3/datasets/processed`
- **Training Set**: `/content/drive/MyDrive/GoogleColab/AI/Qwen3/datasets/train.jsonl`
- **Validation Set**: `/content/drive/MyDrive/GoogleColab/AI/Qwen3/datasets/validation.jsonl`
- **Test Set**: `/content/drive/MyDrive/GoogleColab/AI/Qwen3/datasets/test.jsonl`

---

## 10. Dataset Sources & Provenance Architecture (Phase 2.3.1)

### 10.1 Source Types Taxonomy
The architecture establishes a strongly typed taxonomy of data sources:
- `existing_dataset`: Pre-existing open or domain datasets (e.g. ShareGPT, Alpaca).
- `documentation`: Official technical manuals, API references, kernel specifications.
- `public_domain`: Works formally released to the public domain without restriction.
- `licensed_material`: Formally licensed or permissively attributed technical datasets.
- `human_authored`: Internally curated, verified human engineering problems & solutions.
- `synthetic`: Algorithmic or model-assisted synthetic generation.
- `internal`: Project-specific notes, architecture records, and private technical docs.
- `unknown`: Explicit fallback for unverified or legacy data.

### 10.2 Source Registry & Manifest
All data sources are declaratively registered in `configs/sources.yaml` and loaded dynamically via `SourceRegistry` (`src/dataset/source_registry.py`).
- **Registration**: Ensures every source has a unique `source_id`, typed `source_type`, human-readable name, version, and optional license/generator info.
- **Validation**: Enforces non-empty identifiers and prevents silent duplicates.

### 10.3 Source Adapters
Modular adapters transform diverse ingestion streams into canonical `DatasetRecord` objects:
- `ExistingDatasetAdapter`: Normalizes diverse formats (ShareGPT, Alpaca, Prompt-Response).
- `DocumentationAdapter`: Transforms doc sections and guides into technical QA pairs.
- `HumanAuthoredAdapter`: Bridges human curated examples.
- `SyntheticAdapter`: Attaches synthetic generator name, model version, and batch ID.

### 10.4 Provenance Preservation Across Pipeline
Provenance is strictly immutable across all pipeline stages:
`Source → Adapter → RawRecord → Canonical Record → Cleaner → Deduplicator → Splitter → Train/Val/Test Output`
Even when exact duplicates are removed, the retained example retains its authoritative provenance and duplicate links are logged in telemetry.

### 10.5 Source Statistics & Reporting
The statistics engine computes and exports:
- `source_type_distribution`: Proportion of synthetic, human-authored, documentation, etc.
- `source_distribution`: Counts per registered source entity.
- `generator_distribution`: Granular breakdown of generator models/versions for synthetic data.
- `license_availability`: Metrics on verified licenses vs unspecified records.
- Reports exported to `dataset_report.json`, `dataset_report.md`, `source_report.json`.

### 10.6 Phase 2.3.x Extension Blueprint
Future collectors (Phase 2.3.2+) plug directly into `SourceAdapter` implementations, referencing declarative source entries in `configs/sources.yaml` without modifying core pipeline cleaning, deduplication, or splitting logic.

---

## 11. Domain Dataset Template Architecture (Phase 2.3.2)

### 11.1 Purpose & Role in Synthetic Generation
The Template Architecture establishes declarative, reusable blueprints (`TaskTemplate`) for generating domain-accurate, task-specific, and difficulty-calibrated conversational datasets. This ensures synthetic generation maintains high technical depth, follows strict quality criteria, and aligns with the target domain distributions.

### 11.2 Declarative Task Template Model
Each template is declared in `configs/domain_templates.yaml` and loaded into the `TemplateRegistry` (`src/dataset/template_registry.py`).

A `TaskTemplate` defines:
- **`id`**: Unique deterministic identifier following `<domain>_<topic>_<task_type>_<difficulty>` (e.g. `linux_systems_systemd_services_troubleshooting_intermediate`).
- **`domain`**: Target technical domain from `configs/dataset.yaml`.
- **`topic`**: Granular subtopic within the domain taxonomy.
- **`task_type`**: Interaction category (e.g. `coding`, `debugging`, `troubleshooting`, `system_design`, `problem_solving`, `proof`, `calculation`, `data_interpretation`).
- **`difficulty`**: Baseline difficulty level (`beginner`, `intermediate`, `advanced`, `expert`).
- **`supported_difficulties`**: Allowed difficulty variations for the template.
- **`objective`**: High-level capability being developed or evaluated.
- **`description`**: Concrete technical context, prompt instructions, and solution criteria.
- **`quality_requirements`**: Enforced quality constraints (e.g. `min_answer_length`, `require_code_blocks`, `require_reasoning`, `require_step_by_step`).
- **`prompt_guidelines`**: Specific guidance for crafting user turns and expected assistant outputs.

### 11.3 Extended Taxonomy
Phase 2.3.2 incorporates comprehensive subtopic coverage across all 13 domains and expands task types:
- **New Task Types**: `problem_solving`, `proof`, `calculation`, `data_interpretation`.
- **Domain Subtopics**: Over 80 granular subtopics across Programming, Software Engineering, Cybersecurity, Linux Systems, Networking, AI/ML, Mathematics, Science, Psychology, Human Behavior, Reasoning, Technology, and General Knowledge.

### 11.4 Template Registry API
```python
from src.dataset.template_registry import TemplateRegistry

# Initialize from YAML manifest
registry = TemplateRegistry.from_yaml("configs/domain_templates.yaml")

# Validation against dataset taxonomy
validation_report = registry.validate(dataset_config_path="configs/dataset.yaml")
assert validation_report["is_valid"] is True

# Filtering & Lookup
linux_templates = registry.list_by_domain("linux_systems")
troubleshooting = registry.list_by_task_type("troubleshooting")
expert_templates = registry.list_by_difficulty("expert")
specific = registry.get_template("ai_ml_qlora_peft_coding_advanced")

# Statistics
stats = registry.template_statistics()
print(stats["total_templates"], stats["by_domain"])
```

### 11.5 Generator Integration
The `SampleSyntheticGenerator` (and future LLM-based generators) implements `generate_from_template`:
```python
from src.dataset.generator import SampleSyntheticGenerator

generator = SampleSyntheticGenerator()
records = generator.generate_from_template(template, number_of_examples=5)

# Emitted records contain canonical schema, template-driven prompts, and complete provenance
for rec in records:
    print(rec.metadata.domain, rec.metadata.provenance.source_id)
```

---

## 12. Synthetic Dataset Generation Architecture (Phase 2.3.3)

### 12.1 Generation Architecture Overview
The Synthetic Generation Engine provides a model-agnostic, template-driven architecture for generating conversational training examples that seamlessly feed into the Phase 2.2 processing pipeline:

```text
TaskTemplate (configs/domain_templates.yaml)
   ↓
GenerationRequest (Typed parameters, seed, counts, batch ID)
   ↓
SyntheticGeneratorInterface (Model-agnostic backend)
   ├── SampleSyntheticGenerator (Deterministic verification backend)
   ├── LocalLLMGenerator (Future offline inference backend)
   └── APIGenerator (Future external API backend)
   ↓
GenerationResult (Records, requested/generated/failed metrics, errors)
   ↓
DatasetRecord (Canonical schema + immutable ProvenanceInfo)
   ↓
Phase 2.2 Processing Pipeline (Clean, deduplicate, validate, score, split)
   ↓
Final Dataset Splits (train.jsonl, validation.jsonl, test.jsonl + telemetry reports)
```

### 12.2 Strongly Typed Generation Models

#### `GenerationRequest`
Configures batch synthesis parameters and validates consistency against template definitions:
- `template_id`: Unique identifier referencing a declared `TaskTemplate`.
- `domain`, `topic`, `task_type`, `difficulty`: Optional explicit metadata validated for consistency.
- `number_of_examples`: Batch count (bounded between 1 and 100).
- `seed`: Random seed for deterministic reproducibility.
- `generation_batch_id`: Unique batch identifier (defaults to `batch_<template>_s<seed>_<uuid>`).
- `custom_parameters`: Flexible dictionary for backend-specific prompt flags.

#### `GenerationResult`
Captures generation outputs, failure telemetry, and JSONL persistence:
- `records`: List of validated `DatasetRecord` objects.
- `requested_count`, `generated_count`, `failed_count`: Explicit telemetry accounting.
- `batch_id`, `generator_name`, `generator_version`: Lineage identifiers.
- `errors`: Explicit error tracebacks (never silent failures).
- `save_jsonl(path, overwrite=False)`: Disk persistence with overwrite safeguards.

### 12.3 Model-Agnostic Interface & Sample Backend
`SyntheticGeneratorInterface` defines the contract for all generation backends:
- `generate(domain, topic, task_type, difficulty, number_of_examples=1, **kwargs)`: Legacy direct generation.
- `generate_from_template(template, number_of_examples=1, difficulty=None, **kwargs)`: Direct template synthesis.
- `generate_batch(request, template_registry=None, **kwargs)`: Structured batch generation returning `GenerationResult`.

`SampleSyntheticGenerator`:
- Deterministic testing backend operating with zero external API dependencies.
- Synthesizes diverse scenario variations per example index to prevent n-gram clustering.
- Attaches complete `ProvenanceInfo` (`source_type="synthetic"`, `source="synthetic_generator"`, `generator="sample_test_generator"`, `generator_version="1.0.0"`).

### 12.4 Provenance Lineage Guarantee
Synthetic records preserve complete provenance across all pipeline transformations:
`Generator → Raw JSONL → Loader → Normalizer → Cleaner → Deduplicator → Splitter → Train/Val/Test Splits`
The provenance attributes survive intact and are aggregated in `source_report.json`.

### 12.5 CLI Utility (`scripts/generate_dataset.py`)
Generates synthetic batches and optionally triggers pipeline processing:
```bash
# Basic template-driven generation
python scripts/generate_dataset.py \
  --template programming_python_debugging_intermediate \
  --count 10 \
  --seed 42 \
  --batch-id pilot_001 \
  --overwrite

# Full generation and automated pipeline verification
python scripts/generate_dataset.py \
  --template linux_systems_systemd_services_troubleshooting_intermediate \
  --count 10 \
  --seed 42 \
  --run-pipeline \
  --pipeline-output-dir datasets/processed/pilot
```

---

## 13. Dataset Mixing & Balancing Architecture (Phase 2.3.4)

### 13.1 Mixing Architecture Overview
The Mixing & Balancing Engine (`src/dataset/mixer.py`) unifies heterogeneous candidate pools (synthetic, human-authored, documentation, and existing datasets) into a coherent, balanced dataset meeting domain, difficulty, task, and source distribution goals:

```text
Dataset Sources (Synthetic, Human, Technical Docs, Existing Datasets)
      ↓
Source Pools / Ingestion Handlers
      ↓
DatasetMixer (src/dataset/mixer.py)
      ├── Domain Balancer (Authoritative 13 domains from configs/dataset.yaml)
      ├── Difficulty Balancer (25% beginner, 40% intermediate, 25% advanced, 10% expert)
      ├── Task Balancer (Optional task weights or natural distribution preservation)
      └── Source Balancer (Optional source weights or natural distribution preservation)
      ↓
Deterministic Stratified Selection (Proportional / Balanced Strategies)
      ↓
MixingResult + Mixing Metadata Attachment (preserves ProvenanceInfo)
      ↓
dataset_mix_report.json & dataset_mix_report.md
      ↓
Phase 2.2 Processing Pipeline (Clean → Deduplicate → Quality → Splits)
```

### 13.2 Mixing Strategies

#### `proportional`
- Allocates exact integer quotas matching authoritative distribution weights using the Hare-Niemeyer (Largest Remainder) method.
- Performs two-level joint stratification across domain $\times$ difficulty strata.
- Deterministically samples candidates within each stratum using seeded pseudo-random permutations.

#### `balanced`
- Aims to equalize representation across available domains and classes to prevent domination by high-volume sources when candidate availability is uneven.

### 13.3 Shortage Handling, Oversampling, and Undersampling

- **No Silent Fabrication or Duplication (`allow_oversampling: false`)**:
  When available candidates in a stratum are fewer than the allocated quota, the mixer selects all available records and records explicit `ShortageDetail` entries (requested, available, selected, shortage deficit).
- **Controlled Oversampling (`allow_oversampling: true`)**:
  When explicitly enabled, repeats candidate records to satisfy target quotas. Original conversational content and provenance remain untouched, while mixing metadata is appended (`oversampled: true`, `copy_index: k`).
- **Deterministic Undersampling (`allow_undersampling: true`)**:
  When available candidates exceed quota, the mixer selects a deterministic subset using the seeded PRNG and tracks discarded records.

### 13.4 Strongly Typed Mixing Models

- **`MixingRequest`**: Encapsulates input sources, target count, strategy, seed, oversampling/undersampling flags, and optional target overrides.
- **`MixingResult`**: Contains selected `DatasetRecord` objects, requested/selected/total candidate counts, discarded counts, multi-dimensional distribution reports, shortage details, and oversampling metrics.
- **`DistributionReport`**: Tracks counts, actual percentages, target percentages, and deviations (`actual_percentage - target_percentage`).

### 13.5 CLI Utility (`scripts/mix_dataset.py`)
Combines multiple source pools with rich terminal telemetry and optional immediate Phase 2.2 processing:
```bash
python scripts/mix_dataset.py \
  --input datasets/fixtures/synthetic.jsonl \
  --input datasets/fixtures/human.jsonl \
  --input datasets/fixtures/documentation.jsonl \
  --input datasets/fixtures/existing_dataset.jsonl \
  --count 100 \
  --strategy proportional \
  --seed 42 \
  --output datasets/raw/mixed/pilot_mixed.jsonl \
  --report-dir datasets/raw/mixed/reports \
  --run-pipeline \
  --pipeline-output-dir datasets/processed/mixed_pilot
---

## 14. Pilot Dataset Assembly & Validation Specification (Phase 2.3.5)

### 14.1 Overview & Architecture
The Pilot Dataset Assembly & Validation subsystem (`src/dataset/pilot.py`) serves as the definitive engineering gate prior to large-scale multi-thousand-example dataset production. It coordinates all previously built dataset subsystems into a deterministic, reproducible, auditable assembly pipeline:

```text
Domain Templates (83 templates across 13 domains) + Source Fixtures (Multi-source)
      ↓
Synthetic Candidate Generation (SampleSyntheticGenerator / Template Registry)
      ↓
Candidate Pool (Target * Multiplier)
      ↓
Dataset Pipeline Processing:
   ├── Cleaning & Normalization (RFC NFC, role sequencing, length/artifact filters)
   ├── Deduplication (Exact SHA-256 hash + Near-duplicate MinHash / Jaccard 3-grams)
   ├── Quality Evaluation (QualityValidator scoring >= 0.85 threshold)
   └── Stratified Mixing (DatasetMixer proportional 13-domain x 4-difficulty quota selection)
      ↓
Train / Validation / Test Splitting (90% / 5% / 5% stratified)
      ↓
Cross-Split Data Leakage Detection (Zero exact or near overlap across splits)
      ↓
Pilot Artifacts & Reports:
   ├── Processed Data: datasets/pilot/v1/processed/{train,validation,test}.jsonl
   ├── Manifest: datasets/pilot/v1/manifests/pilot_manifest.json
   ├── Readiness Audit: datasets/pilot/v1/reports/pilot_readiness_report.{json,md}
   └── Mixing & Pipeline Reports: datasets/pilot/v1/reports/{dataset_report,dataset_mix_report,rejection_report}.{json,md}
```

### 14.2 Pilot Readiness Evaluation Dimensions
The pilot readiness evaluator verifies the dataset across nine critical quality and distribution dimensions, computing a structured status (`PASS`, `WARN`, `FAIL`):

| Dimension | Threshold Criteria | Target Status |
| :--- | :--- | :--- |
| **Schema Validity** | 0 formatting or validation rejections in processed records | `PASS` |
| **Domain Coverage** | All 13 authoritative domains represented | `PASS` |
| **Difficulty Coverage** | All 4 difficulty tiers represented (`beginner`, `intermediate`, `advanced`, `expert`) | `PASS` |
| **Quality Score** | Dataset mean quality score $\ge 0.85$, with $\ge 90\%$ of records scoring $\ge 0.90$ | `PASS` |
| **Deduplication Rate** | Post-generation duplicate rate $\le 5.0\%$ | `PASS` |
| **Provenance Completeness** | 100% of records retain complete, immutable `ProvenanceInfo` | `PASS` |
| **Split Integrity** | Train (~90%), Validation (~5%), Test (~5%) split ratios within $\pm 5\%$ tolerance | `PASS` |
| **Cross-Split Leakage** | 0 overlapping canonical content hashes across any pair of splits | `PASS` (Strict `FAIL` if $>0$) |
| **Source Diversity** | $\ge 2$ distinct source types active in ingested candidate pool | `PASS` |

### 14.3 Output Manifest & Verification Artifacts
Every pilot execution produces a complete cryptographic record:
- **`manifests/pilot_manifest.json`**: Records pilot version, timestamp, configuration hash, random seed, candidate count, final counts, split breakdowns, file paths, and individual SHA-256 file checksums.
- **`reports/pilot_readiness_report.md`**: Markdown audit report summarizing dimension statuses, domain/difficulty balances, quality percentiles, and shortage details.
- **`reports/pilot_readiness_report.json`**: Machine-readable readiness telemetry for CI/CD gates.

### 14.4 CLI Execution & Automation
The pilot assembly can be invoked directly from the CLI via `scripts/run_pilot.py`:
```bash
python scripts/run_pilot.py \
  --count 1000 \
  --candidate-multiplier 1.2 \
  --seed 42 \
  --version pilot-v1 \
  --output-dir datasets/pilot/v1
```

---

## 15. Phase 3.1 — Production Dataset Specification & Scaling Architecture

### 15.1 Production Overview & Target Scale
Phase 3.1 transitions the validated pilot pipeline into a production-grade scaling architecture capable of planning, generating, checkpointing, and freezing datasets ranging from 10,000 to 100,000+ conversational examples for Qwen3-4B-Base.

Key Production Parameters:
- **Default Production Target**: 10,000 high-quality accepted conversational examples.
- **Candidate Multiplier**: 1.20x (12,000 raw candidate examples to absorb deduplication, quality filtering, and domain shortages).
- **Default Batch Size**: 500 candidate examples per batch (24 total batches for a 10K dataset).
- **Deterministic Seed**: Global random seed with cryptographically derived batch seeds.
- **Dry-Run Planning**: Pure mathematical quota allocation without generating synthetic records.

### 15.2 Quota Apportionment & Exact Sum Preservation
Quota allocation across domains and difficulty tiers is strictly governed by the **Hare-Niemeyer (Largest Remainder)** algorithm:
1. Exact quota $E_i = \text{Target} \times \frac{w_i}{\sum_k w_k}$
2. Integer base quota $B_i = \lfloor E_i \rfloor$
3. Remainder $R_i = E_i - B_i$
4. Deficit $\Delta = \text{Target} - \sum_i B_i$ is distributed by incrementing the categories with the largest remainders $R_i$, breaking ties deterministically by category name.

This guarantees:
$$\sum_{i=1}^{13} \text{Domain\_Quota}_i = \text{Target}$$
$$\sum_{j=1}^{4} \text{Difficulty\_Quota}_j = \text{Target}$$

### 15.3 2D Domain × Difficulty Quota Matrix (10,000 Examples)
Using hierarchical Largest Remainder allocation across domain rows, each domain row sum equals its exact Hare-Niemeyer quota, and the grand total equals 10,000:

| Domain | Weight % | Beginner (25%) | Intermediate (40%) | Advanced (25%) | Expert (10%) | Row Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `programming` | 18.2% | 455 | 728 | 455 | 182 | **1,820** |
| `cybersecurity` | 13.6% | 340 | 544 | 340 | 136 | **1,360** |
| `software_engineering` | 9.1% | 227 | 364 | 228 | 91 | **910** |
| `linux_systems` | 9.1% | 227 | 364 | 228 | 91 | **910** |
| `networking` | 7.3% | 182 | 292 | 183 | 73 | **730** |
| `ai_ml` | 7.3% | 182 | 292 | 183 | 73 | **730** |
| `general_knowledge` | 6.5% | 162 | 260 | 163 | 65 | **650** |
| `reasoning` | 6.4% | 160 | 256 | 160 | 64 | **640** |
| `mathematics` | 4.5% | 112 | 180 | 113 | 45 | **450** |
| `science` | 4.5% | 112 | 180 | 113 | 45 | **450** |
| `psychology` | 4.5% | 112 | 180 | 113 | 45 | **450** |
| `human_behavior` | 4.5% | 112 | 180 | 113 | 45 | **450** |
| `technology` | 4.5% | 112 | 180 | 113 | 45 | **450** |
| **TOTAL** | **100.0%** | **2,495** | **4,000** | **2,505** | **1,000** | **10,000** |

### 15.4 Batch Architecture & Checkpoint Management
To support resumable and distributed generation:
- **Deterministic Batch IDs**: `dataset-v1.0-batch-0001`, `dataset-v1.0-batch-0002`, ..., `dataset-v1.0-batch-0024`.
- **Deterministic Batch Seed**: `derive_batch_seed(global_seed, batch_index) = int(SHA-256("qwen3:production:seed:{global_seed}:batch:{batch_index}")[:4])`.
- **Checkpoint Files**: `checkpoints/dataset-v1.0-batch-0001.json` tracking status (`pending`, `generating`, `completed`, `failed`), candidate counts, accepted/rejected counts, quality telemetry, output files, and checksums.
- **Resume Protocol**: `ProductionCheckpointManager` scans existing checkpoints and automatically skips completed batches during execution runs.
- **Fail-Fast Support**: Configurable `fail_fast: true` (halts on first error) vs `fail_fast: false` (records batch failure and continues with remaining batches).

### 15.5 Production Manifest & Dataset Freeze States
Every production release is tracked via `manifests/production_manifest.json` across five lifecycle freeze states:
```
[PLANNED] ──> [GENERATING] ──> [VALIDATING] ──> [READY] ──> [FROZEN]
```
- **`PLANNED`**: Quotas and batch specifications computed; no records generated.
- **`GENERATING`**: Batch generation and checkpointing in progress.
- **`VALIDATING`**: Raw batches passed through cleaner, deduplicator, quality validator, and splitter.
- **`READY`**: All quality gates and cross-split leakage checks passed.
- **`FROZEN`**: SHA-256 checksums finalized; dataset locked for downstream QLoRA training.

### 15.6 Dry-Run Production Planning CLI
The dry-run planner can be executed anytime to inspect quotas and generate plan reports without writing synthetic records:
```bash
python scripts/plan_production.py \
  --target 10000 \
  --seed 42 \
  --candidate-multiplier 1.20 \
  --batch-size 500 \
  --version dataset-v1.0 \
  --output-dir datasets/production
```

---

## 16. Phase 3.2 — Production Dataset Generation Engine

### 16.1 Architecture & Pipeline Flow
Phase 3.2 implements the execution engine (`src/dataset/production_generator.py`) that materializes the planned production batches through synthetic synthesis, inline validation, atomic persistence, fault-tolerant checkpointing, cross-batch deduplication, stratified global balancing, and dataset freeze manifest updates.

```text
Production Plan (Matrix Quotas & Batch Seeds)
                     │
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Per-Batch Synthesis Loop (ProductionGenerationEngine)   │
  │  1. Deterministic template selection & synthesis        │
  │  2. Inline cleaning (CleanedRecordReport)               │
  │  3. Batch-local deduplication (exact + near)            │
  │  4. Quality gate evaluation (>= 0.85 threshold)         │
  │  5. Atomic file persistence (raw & processed .jsonl)    │
  │  6. BatchCheckpoint update (atomic os.replace)          │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Global Aggregation & Mixing Layer                      │
  │  1. Cross-batch SHA-256 deduplication                  │
  │  2. DatasetMixer global stratified balancing            │
  │  3. Quota deficit evaluation & replenishment tracking   │
  │  4. Atomic candidate_dataset.jsonl write                │
  │  5. ProductionManifest state transition (-> VALIDATING) │
  │  6. Comprehensive reporting (MD + JSON telemetry)       │
  └─────────────────────────────────────────────────────────┘
```

### 16.2 Atomic Persistence Protocol
All candidate files, batch checkpoints, dataset outputs, and manifests are written atomically using a temporary file pattern:
1. Write JSON/JSONL stream to `.tmp_<filename>_<uuid>`
2. Flush and synchronize buffer to physical storage (`os.fsync`)
3. Atomic filesystem rename (`os.replace`) to target destination

This eliminates partial-file corruption and guarantees consistency even under sudden process interruption or system crash.

### 16.3 Checkpoint & Resumability Protocol
- Batches are processed sequentially or in staged slices (`--max-batches N`).
- Before starting a batch, the engine checks `ProductionCheckpointManager.load_checkpoint(batch_id)`.
- If status is `COMPLETED` and `force=False`, batch execution is safely skipped.
- If `--retry-failed` is passed, failed batch checkpoints are reset to `PENDING` and re-attempted.

### 16.4 Production Generation CLI
Batch synthesis is controlled via `scripts/generate_production.py`:
```bash
# 1. Full Production Generation Run (10,000 target examples)
python scripts/generate_production.py \
  --target 10000 \
  --batch-size 500 \
  --candidate-multiplier 1.20 \
  --seed 42 \
  --version dataset-v1.0 \
  --output-dir datasets/production

# 2. Staged Batch Execution (e.g., initial 4 batches of 25 examples)
python scripts/generate_production.py \
  --target 100 \
  --batch-size 25 \
  --max-batches 4 \
  --version dataset-v1.0

# 3. Resuming Interrupted Run
python scripts/generate_production.py --resume --retry-failed
```

---

## 17. Phase 3.3 — Production Data Quality Assurance, Token Budget Analysis & Final Freeze

### 17.1 Production QA Engine Architecture & Multi-Dimensional Readiness Gates
Phase 3.3 provides the final validation and freeze layer (`src/dataset/production_qa.py`) to certify dataset readiness prior to downstream model fine-tuning. The QA engine is strictly non-destructive and evaluates candidate datasets across multi-dimensional quality gates:

```text
Candidate Records / Splits
           │
           ▼
┌────────────────────────────────────────────────────────┐
│  ProductionQAEngine Validation Pipeline                │
│  1. Schema Validation (100% Pydantic conformance)      │
│  2. Provenance Audit (100% complete generator/source)   │
│  3. Domain & Difficulty Distribution Coverage          │
│  4. Task & Source Diversity Analysis                   │
│  5. Statistical Quality Scoring (Mean >= 0.85/0.90)    │
│  6. Exact & Near Duplicate Analysis (Jaccard MinHash)  │
│  7. Cross-Split Leakage Check (Zero Train/Val/Test)    │
│  8. Token Accounting & Truncation Risk (Max 4096)      │
│  9. Empirical Yield Loss Attribution Tracking          │
│  10. Sizing & Multi-Scale Storage Projections (10k-100k)│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Multi-Dimensional Readiness Gate Evaluation           │
│  • PASS: Meets all critical quality thresholds         │
│  • WARN: Non-critical deviations (e.g. mock tokenizer) │
│  • FAIL: Critical blockers (schema, leakage, quality)   │
└──────────────────────────┬─────────────────────────────┘
                           │ (Readiness != FAIL)
                           ▼
┌────────────────────────────────────────────────────────┐
│  Cryptographic Freeze & Manifest Locking               │
│  • SHA-256 Checksum generation across dataset files    │
│  • ProductionManifest transition to FROZEN state       │
│  • Immutability guards lock generation modifications   │
└────────────────────────────────────────────────────────┘
```

### 17.2 Readiness Gates & Quality Thresholds
The QA engine evaluates typed gate results categorized into critical and non-critical thresholds:
- **`schema_validity`** (Critical): `invalid_records == 0`.
- **`provenance_completeness`** (Critical): `provenance_completeness == 1.0` (100%).
- **`domain_coverage`** (Critical): All 13 technical domains must be represented.
- **`difficulty_coverage`** (Critical): All 4 difficulty tiers (`beginner`, `intermediate`, `advanced`, `expert`) must be represented.
- **`quality_score_mean`** (Critical): Dataset mean quality score $\ge 0.85$ (minimum) / $\ge 0.90$ (preferred).
- **`duplicate_rate`** (Critical): Total duplicate rate $\le 5.0\%$.
- **`cross_split_leakage`** (Critical): 0 exact hash overlaps across train, validation, and test splits.
- **`context_length_risk`** (Warning): Records exceeding 4,096 tokens $\le 1.0\%$.
- **`token_analysis_availability`** (Warning/Critical): Real tokenizer loaded and verified.
- **`minimum_records`** (Warning): Final dataset meets minimum scale requirements ($\ge 50$ records).

### 17.3 Empirical Yield Loss Attribution
Yield loss across pipeline stages is tracked and attributed deterministically:
1. **Raw Synthesis**: 100% initial candidate intake.
2. **Inline Cleaning**: Strips whitespace, normalizes Unicode, removes control characters.
3. **Quality Filtering**: Removes records below minimum threshold ($< 0.85$).
4. **Deduplication**: Eliminates exact text matches and near-duplicate character n-gram similarities.
5. **Stratified Balancing**: Filters surplus records to enforce strict proportional domain $\times$ difficulty mixing without oversampling.

### 17.4 Token Accounting & Training Budget Estimation
- **Tokenizer Support**: AutoTokenizer loader with fallback detection and mock tokenizer support for offline testing.
- **Sequence Safety Margins**: Identifies records exceeding 90% context safety window (3,686 tokens) and hard cutoff (4,096 tokens).
- **Analytical Training Budget**: Computes effective batch size, steps per epoch, and cumulative token exposure across 1, 2, and 3 epochs for downstream QLoRA configuration.

### 17.5 Cryptographic Dataset Freezing & Immutability Protections
Once certified, datasets are sealed via the freeze protocol:
- Calculates SHA-256 hashes of `candidate_dataset.jsonl`, `train.jsonl`, `validation.jsonl`, and `test.jsonl`.
- Transitions `ProductionManifest` status to `FROZEN` with timestamped audit metadata.
- Updates `ProductionGenerationEngine` with runtime guards: any generation attempt targeting a `FROZEN` dataset version raises a `RuntimeError`.

### 17.6 Production QA CLI Tooling
Production QA and dataset freeze operations are executed via `scripts/qa_production.py`:
```bash
# 1. Run QA Evaluation & Auto-Split on Candidate Dataset
python scripts/qa_production.py \
  --input datasets/production/processed/candidate_dataset.jsonl \
  --manifest datasets/production/manifests/production_manifest.json \
  --output-dir datasets/production/reports \
  --auto-split

# 2. Run QA on Pre-split Dataset with Leakage Analysis
python scripts/qa_production.py \
  --train datasets/production/processed/train.jsonl \
  --validation datasets/production/processed/validation.jsonl \
  --test datasets/production/processed/test.jsonl \
  --output-dir datasets/production/reports

# 3. Seal and Cryptographically Freeze Production Dataset
python scripts/qa_production.py \
  --input datasets/production/processed/candidate_dataset.jsonl \
  --manifest datasets/production/manifests/production_manifest.json \
  --output-dir datasets/production/reports \
  --freeze --force
```






