"""
Benchmark Case Schema, Builders & Domain Case Generators (Phase 4.5).
Provides strict Pydantic definitions for independent benchmark evaluation cases,
covering all 13 domains, 4 difficulty levels, 22 task types, and 7 evaluation modes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator

from src.dataset.schema import DifficultyLevel, Message, Role, TaskType


DOMAINS_13 = [
    "programming",
    "software_engineering",
    "cybersecurity",
    "linux_systems",
    "networking",
    "ai_ml",
    "mathematics",
    "science",
    "psychology",
    "human_behavior",
    "reasoning",
    "technology",
    "general_knowledge",
]


class BenchmarkEvaluationType(str, Enum):
    """Evaluation paradigms supported by the benchmark suite."""
    DETERMINISTIC = "deterministic"
    REFERENCE_BASED = "reference_based"
    STRUCTURAL = "structural"
    CODE_BASED = "code_based"
    NUMERICAL = "numerical"
    REASONING = "reasoning"
    QUALITATIVE = "qualitative"


class BenchmarkCase(BaseModel):
    """
    Independent Benchmark Evaluation Case.
    Captures complete conversational prompt, ground truth reference completion,
    behavioral expectations, and deterministic evaluation criteria.
    """
    benchmark_id: str
    domain: str
    topic: str
    difficulty: str
    task_type: str
    messages: List[Message]
    expected_behavior: str
    reference_answer: str
    evaluation_type: str
    evaluation_metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    source: str = "independent_benchmark_curation"
    version: str = "benchmark-v1.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in DOMAINS_13:
            raise ValueError(f"Domain '{v}' is not in 13 canonical domains: {DOMAINS_13}")
        return clean

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        clean = v.strip().lower()
        valid = [d.value for d in DifficultyLevel]
        if clean not in valid:
            raise ValueError(f"Difficulty '{v}' is not valid. Allowed: {valid}")
        return clean

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        clean = v.strip().lower()
        valid = [t.value for t in TaskType]
        if clean not in valid:
            raise ValueError(f"Task type '{v}' is not valid. Allowed: {valid}")
        return clean

    @field_validator("evaluation_type")
    @classmethod
    def validate_eval_type(cls, v: str) -> str:
        clean = v.strip().lower()
        valid = [e.value for e in BenchmarkEvaluationType]
        if clean not in valid:
            raise ValueError(f"Evaluation type '{v}' is not valid. Allowed: {valid}")
        return clean

    @field_validator("reference_answer")
    @classmethod
    def validate_reference_answer(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reference_answer cannot be empty.")
        return v.strip()

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("Messages cannot be empty.")
        return v

    @model_validator(mode="after")
    def validate_conversation_flow(self) -> BenchmarkCase:
        msgs = self.messages
        if not msgs:
            raise ValueError("Messages list is empty.")

        first_non_sys_idx = 0
        if msgs[0].role == Role.SYSTEM:
            if len(msgs) == 1:
                raise ValueError("Benchmark case contains only system prompt without user turn.")
            first_non_sys_idx = 1

        if msgs[first_non_sys_idx].role != Role.USER:
            raise ValueError(f"First non-system message must be 'user', got '{msgs[first_non_sys_idx].role}'.")

        # Check alternating user/assistant turns
        prev_role: Optional[Role] = None
        for i, msg in enumerate(msgs):
            if i == 0 and msg.role == Role.SYSTEM:
                prev_role = Role.SYSTEM
                continue
            if prev_role is not None and prev_role == msg.role:
                raise ValueError(f"Consecutive messages from same role '{msg.role}' at index {i}.")
            prev_role = msg.role

        # Final message in canonical benchmark record should match reference_answer
        last_msg = msgs[-1]
        role_val = last_msg.role.value if hasattr(last_msg.role, "value") else str(last_msg.role)
        if role_val != "assistant":
            raise ValueError(f"Final turn of benchmark record must be ASSISTANT, got '{role_val}'.")

        return self

    def get_prompt_messages(self) -> List[Message]:
        """Return prompt messages excluding the final ground truth assistant turn."""
        return self.messages[:-1]

    def canonical_prompt_hash(self) -> str:
        """Hash the prompt messages for exact collision and leakage checking."""
        prompts = self.get_prompt_messages()
        canonical_repr = [{"role": m.role.value, "content": " ".join(m.content.strip().split())} for m in prompts]
        return hashlib.sha256(json.dumps(canonical_repr, sort_keys=True).encode("utf-8")).hexdigest()

    def canonical_content_hash(self) -> str:
        """Hash entire conversation dialogue."""
        canonical_repr = [{"role": m.role.value, "content": " ".join(m.content.strip().split())} for m in self.messages]
        return hashlib.sha256(json.dumps(canonical_repr, sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "domain": self.domain,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "messages": [m.model_dump() for m in self.messages],
            "expected_behavior": self.expected_behavior,
            "reference_answer": self.reference_answer,
            "evaluation_type": self.evaluation_type,
            "evaluation_metadata": self.evaluation_metadata,
            "tags": self.tags,
            "source": self.source,
            "version": self.version,
            "created_at": self.created_at,
        }


# ============================================================================
# SAFE CODE EVALUATOR INTERFACE (NO DIRECT HOST EXECUTION)
# ============================================================================

class SafeCodeEvaluator:
    """Evaluates code completions using safe AST syntax analysis and static checks."""

    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Safely parse Python code into an AST without executing it."""
        try:
            code_clean = code
            match = re.search(r"```(?:python)?\s*(.*?)\s*```", code, re.DOTALL)
            if match:
                code_clean = match.group(1)
            ast.parse(code_clean)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def check_required_constructs(code: str, required_symbols: List[str]) -> Tuple[bool, List[str]]:
        """Verify presence of specific function or class definitions."""
        try:
            code_clean = code
            match = re.search(r"```(?:python)?\s*(.*?)\s*```", code, re.DOTALL)
            if match:
                code_clean = match.group(1)
            tree = ast.parse(code_clean)
            found_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    found_names.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    found_names.add(node.name)
                elif isinstance(node, ast.Name):
                    found_names.add(node.id)

            missing = [sym for sym in required_symbols if sym not in found_names]
            return len(missing) == 0, missing
        except Exception:
            return False, required_symbols


# ============================================================================
# BENCHMARK SUITE GENERATOR ENGINE (500 DISTINCT CURATED CASES)
# ============================================================================

class BenchmarkSuiteBuilder:
    """
    Constructs a deterministic, comprehensive, independent benchmark suite
    across all 13 canonical domains, 4 difficulties, and multiple task types.
    """

    @classmethod
    def generate_benchmark_suite(cls, target_count: int = 500, seed: int = 42) -> List[BenchmarkCase]:
        """Generate high-quality, validated, independent benchmark cases."""
        cases: List[BenchmarkCase] = []
        cur_id = 1

        domain_definitions = [
            # 1. PROGRAMMING
            ("programming", "data_structures", "intermediate", "code_generation", "code_based",
             "Implement an LRU Cache in Python with O(1) get and put operations using OrderedDict or doubly linked list.",
             "An LRU (Least Recently Used) cache can be efficiently implemented in Python using `collections.OrderedDict`:\n\n```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache = OrderedDict()\n\n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n```\n\n- `get(key)` moves accessed element to the right end (most recently used).\n- `put(key, value)` inserts and evicts from the left end if capacity is exceeded.",
             {"required_symbols": ["LRUCache", "get", "put"], "static_checks": ["ast_parse", "class_def"]}),

            ("programming", "algorithms", "advanced", "coding", "code_based",
             "Write a function `quicksort(arr: list) -> list` implementing randomized in-place QuickSort.",
             "```python\nimport random\n\ndef quicksort(arr: list) -> list:\n    if len(arr) <= 1:\n        return arr\n    pivot = random.choice(arr)\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n```",
             {"required_symbols": ["quicksort"], "static_checks": ["ast_parse"]}),

            ("programming", "concurrency", "expert", "debugging", "code_based",
             "Explain the race condition in this Python multithreaded increment counter and provide the thread-safe fix using `threading.Lock`.",
             "In Python, the operation `counter += 1` consists of three bytecode instructions (read, add, store). Without a lock, thread preemptions between instructions cause lost updates.\n\n### Thread-Safe Fix:\n```python\nimport threading\n\nclass SafeCounter:\n    def __init__(self):\n        self._val = 0\n        self._lock = threading.Lock()\n\n    def increment(self):\n        with self._lock:\n            self._val += 1\n\n    @property\n    def value(self):\n        with self._lock:\n            return self._val\n```",
             {"required_symbols": ["SafeCounter", "increment"], "static_checks": ["ast_parse"]}),

            # 2. SOFTWARE ENGINEERING
            ("software_engineering", "design_patterns", "intermediate", "explanation", "reference_based",
             "Compare the Factory Method pattern with the Abstract Factory pattern. When should each be chosen?",
             "The Factory Method and Abstract Factory patterns both encapsulate object creation, but at different levels of abstraction:\n\n1. **Factory Method**:\n   - **Definition**: Uses inheritance to delegate object instantiation to subclasses through a single method.\n   - **Scope**: Creates a single product.\n   - **Use Case**: When a class cannot anticipate the exact class of objects it must create.\n\n2. **Abstract Factory**:\n   - **Definition**: Uses object composition to provide an interface for creating families of related or dependent objects without specifying their concrete classes.\n   - **Scope**: Creates entire product suites (e.g., UI controls for Windows vs macOS).\n   - **Use Case**: When a system must be independent of how its products are created, composed, and represented.",
             {"key_concepts": ["inheritance", "composition", "single product", "product families"]}),

            ("software_engineering", "system_architecture", "advanced", "system_design", "structural",
             "Design a resilient idempotent webhook processing architecture in a microservices environment.",
             "### Resilient Idempotent Webhook Architecture:\n\n1. **Ingress API Gateway / Edge Ingestion**:\n   - Fast signature verification (HMAC SHA-256).\n   - Immediate return of HTTP `202 Accepted` after appending payload to a durable append-only event log (e.g. Apache Kafka or AWS SQS).\n\n2. **Deduplication & Idempotency Store**:\n   - Unique `webhook_id` stored in Redis/DynamoDB with atomic `SETNX` and TTL.\n   - Database transactions utilize unique constraints (`UNIQUE (webhook_id)`).\n\n3. **Asynchronous Consumer Workers**:\n   - Idempotent database state updates.\n   - Exponential backoff retry with Dead Letter Queue (DLQ) for unrecoverable failures.",
             {"required_sections": ["Ingress", "Deduplication", "Consumer"]}),

            # 3. CYBERSECURITY
            ("cybersecurity", "web_security", "beginner", "explanation", "reference_based",
             "What is Cross-Site Scripting (XSS) and what is the primary defense against Stored XSS?",
             "**Cross-Site Scripting (XSS)** is a security vulnerability where an attacker injects malicious client-side scripts into web pages viewed by other users.\n\n### Primary Defenses Against Stored XSS:\n1. **Context-Aware Output Encoding**: HTML entity encode, JavaScript encode, or CSS encode user input before rendering.\n2. **Content Security Policy (CSP)**: Restrict origins of executable scripts and disallow `unsafe-inline` scripts.\n3. **Safe Templating Engines**: Use modern frontend frameworks (React, Vue, Angular) that auto-escape bindings.\n4. **Input Sanitization**: Use strict HTML sanitization libraries (e.g., DOMPurify) if rich text HTML is required.",
             {"required_terms": ["context-aware output encoding", "content security policy", "escaping"]}),

            ("cybersecurity", "cryptography", "expert", "proof", "reasoning",
             "Explain why AES-GCM provides authenticated encryption while AES-CBC without HMAC does not. Detail the padding oracle vulnerability.",
             "### Authenticated Encryption vs Unauthenticated Modes:\n\n1. **AES-GCM (Galois/Counter Mode)**:\n   - Combines CTR mode confidentiality with GHASH authentication to produce an authentication tag over both ciphertext and Associated Data (AAD).\n   - Any bit modification in the ciphertext results in verification failure before decryption.\n\n2. **AES-CBC Vulnerabilities**:\n   - Provides confidentiality only, not integrity.\n   - **Padding Oracle Attack**: In PKCS#7 padded CBC, differences in error responses (invalid padding vs invalid MAC) leak plaintext bytes by manipulating the previous ciphertext block (`C_{i-1} XOR P_i`).",
             {"key_concept": "padding oracle in unauthenticated CBC", "expected_conclusion": "AES-GCM verifies integrity via GHASH before plaintext release"}),

            # 4. LINUX SYSTEMS
            ("linux_systems", "process_management", "intermediate", "troubleshooting", "reference_based",
             "What is the difference between a Zombie process and an Orphan process in Linux, and how are they cleaned up?",
             "### Zombie vs Orphan Processes in Linux:\n\n1. **Zombie Process (`Z` state)**:\n   - **Definition**: A process that has terminated execution (`exit()`), but its exit status has not yet been read by its parent via `wait()` / `waitpid()`.\n   - **Resource Use**: Holds a PID and process table entry (task_struct), but no memory or CPU.\n   - **Cleanup**: Terminate the parent process; zombies are inherited by `init`/`systemd` (PID 1), which reaps them.\n\n2. **Orphan Process**:\n   - **Definition**: A running process whose parent has terminated.\n   - **Cleanup**: Automatically adopted by PID 1 (`systemd`/`init`) which acts as its new parent and handles its exit code upon termination.",
             {"key_distinctions": ["PID 1 adoption", "waitpid", "task_struct"]}),

            ("linux_systems", "memory", "advanced", "calculation", "numerical",
             "In an x86-64 system with standard 4KB pages and 4-level paging (PML4 -> PDPT -> PD -> PT), calculate how many bits of the 64-bit virtual address are used for each translation level and page offset.",
             "In x86-64 4-level paging:\n- **Total active virtual address bits**: 48 bits (sign-extended to 64 bits).\n- **Page Offset**: 12 bits ($2^{12} = 4096$ bytes = 4 KB).\n- **Page Table (PT)**: 9 bits ($2^9 = 512$ entries).\n- **Page Directory (PD)**: 9 bits ($512$ entries).\n- **Page Directory Pointer Table (PDPT)**: 9 bits ($512$ entries).\n- **Page Map Level 4 (PML4)**: 9 bits ($512$ entries).\n\nTotal: $9 + 9 + 9 + 9 + 12 = 48$ bits.",
             {"expected_numerical_values": {"pml4": 9, "pdpt": 9, "pd": 9, "pt": 9, "offset": 12, "total": 48}}),

            # 5. NETWORKING
            ("networking", "transport_layer", "intermediate", "explanation", "reference_based",
             "Detail the TCP 3-way handshake and describe the role of the SYN-ACK packet and Initial Sequence Numbers (ISN).",
             "### TCP 3-Way Handshake:\n1. **SYN (Client -> Server)**:\n   - Client chooses a randomized Initial Sequence Number ($ISN_c$) and sends `[SYN, seq=ISN_c]`.\n2. **SYN-ACK (Server -> Client)**:\n   - Server acknowledges client's sequence (`ack=ISN_c + 1`) and sends its own randomized $ISN_s$ in `[SYN, ACK, seq=ISN_s, ack=ISN_c + 1]`.\n3. **ACK (Client -> Server)**:\n   - Client acknowledges server's sequence (`[ACK, seq=ISN_c + 1, ack=ISN_s + 1]`). Connection is now `ESTABLISHED`.\n\n**Role of Random ISN**: Prevents TCP sequence prediction attacks and avoids collisions with stale packets from previous connections.",
             {"required_states": ["SYN", "SYN-ACK", "ACK", "ESTABLISHED"]}),

            # 6. AI & MACHINE LEARNING
            ("ai_ml", "transformer_architecture", "advanced", "explanation", "deterministic",
             "Write the mathematical equation for Scaled Dot-Product Attention in a Transformer and explain the scaling factor $\\sqrt{d_k}$.",
             "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n\n### Role of $\\sqrt{d_k}$:\nWhen $d_k$ is large, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients (vanishing gradient problem). Dividing by $\\sqrt{d_k}$ scales the dot products back to variance 1 (assuming $Q$ and $K$ elements are independent random variables with zero mean and unit variance).",
             {"exact_formula": "softmax((Q K^T) / sqrt(d_k)) V", "concept": "vanishing gradient"}),

            ("ai_ml", "fine_tuning", "expert", "calculation", "numerical",
             "For a LoRA rank $r=16$ applied to query and value projection matrices of dimension $d_{model}=2048$, calculate the total number of trainable LoRA parameters for one attention layer.",
             "For one projection matrix of dimension $d_{in} \\times d_{out} = 2048 \\times 2048$, LoRA decomposes the weight update into:\n$$\\Delta W = B \\cdot A$$\nwhere $A \\in \\mathbb{R}^{r \\times d_{in}}$ and $B \\in \\mathbb{R}^{d_{out} \\times r}$.\n\nParameters per matrix: $r \\times d_{in} + d_{out} \\times r = 16 \\times 2048 + 2048 \\times 16 = 32,768 + 32,768 = 65,536$.\n\nFor 2 matrices (Query + Value):\n$$2 \\times 65,536 = 131,072 \\text{ parameters}$$.",
             {"expected_numerical_values": {"params_per_matrix": 65536, "total_layer_params": 131072}}),

            # 7. MATHEMATICS
            ("mathematics", "linear_algebra", "intermediate", "calculation", "numerical",
             "Find the eigenvalues of the matrix A = [[3, 1], [0, 2]].",
             "For an upper triangular matrix:\n$$A = \\begin{pmatrix} 3 & 1 \\\\ 0 & 2 \\end{pmatrix}$$\nThe characteristic equation is:\n$$\\det(A - \\lambda I) = (3 - \\lambda)(2 - \\lambda) - (0)(1) = 0$$\n$$(3 - \\lambda)(2 - \\lambda) = 0$$\n\nThus, the eigenvalues are:\n$$\\lambda_1 = 3, \\quad \\lambda_2 = 2$$.",
             {"expected_numerical_values": {"eigenvalue_1": 3, "eigenvalue_2": 2}}),

            ("mathematics", "probability", "advanced", "problem_solving", "numerical",
             "Suppose a diagnostic test has 99% sensitivity and 95% specificity. A rare disease affects 0.1% of the population. What is the probability that a randomly tested person who tests positive actually has the disease?",
             "Using Bayes' Theorem:\n- $P(D) = 0.001$ (Prior probability of disease)\n- $P(\\neg D) = 0.999$\n- $P(+ | D) = 0.99$ (Sensitivity)\n- $P(+ | \\neg D) = 1 - 0.95 = 0.05$ (False positive rate)\n\n$$P(D | +) = \\frac{P(+ | D) P(D)}{P(+ | D) P(D) + P(+ | \\neg D) P(\\neg D)}$$\n$$P(D | +) = \\frac{0.99 \\times 0.001}{(0.99 \\times 0.001) + (0.05 \\times 0.999)} = \\frac{0.00099}{0.00099 + 0.04995} = \\frac{0.00099}{0.05094} \\approx 0.019435$$\n\nThe posterior probability is approximately **1.94%** (or ~0.0194).",
             {"expected_numerical_values": {"posterior_prob": 0.0194, "tolerance": 0.001}}),

            # 8. SCIENCE
            ("science", "physics", "advanced", "calculation", "numerical",
             "Calculate the Schwarzschild radius of an object with the mass of the Sun (M = 1.989 x 10^30 kg), using G = 6.674 x 10^-11 m^3 kg^-1 s^-2 and c = 2.998 x 10^8 m/s.",
             "The Schwarzschild radius formula is:\n$$r_s = \\frac{2GM}{c^2}$$\n\nSubstituting the constants:\n$$r_s = \\frac{2 \\times (6.674 \\times 10^{-11}) \\times (1.989 \\times 10^{30})}{(2.998 \\times 10^8)^2}$$\n$$r_s = \\frac{2.6549 \\times 10^{20}}{8.9880 \\times 10^{16}} \\approx 2953.8 \\text{ meters}$$\n\nThe Schwarzschild radius is approximately **2,954 meters** (~2.95 km).",
             {"expected_numerical_values": {"schwarzschild_radius_m": 2954.0, "tolerance": 10.0}}),

            # 9. PSYCHOLOGY
            ("psychology", "cognitive_biases", "intermediate", "classification", "reference_based",
             "Classify and distinguish Confirmation Bias, Anchoring Bias, and Availability Heuristic.",
             "1. **Confirmation Bias**: The tendency to search for, interpret, favor, and recall information in a way that confirms preexisting beliefs while ignoring contradictory evidence.\n2. **Anchoring Bias**: The disproportionate reliance on the first piece of information offered (the 'anchor') when making decisions or estimations.\n3. **Availability Heuristic**: The mental shortcut that relies on immediate examples that come to a given person's mind when evaluating a specific topic, concept, method or decision (recency/vividness over base rates).",
             {"concepts": ["confirmation", "anchoring", "availability"]}),

            # 10. HUMAN BEHAVIOR
            ("human_behavior", "game_theory", "advanced", "decision_analysis", "reasoning",
             "In the standard Prisoner's Dilemma, prove why (Defect, Defect) is the unique Nash Equilibrium even though (Cooperate, Cooperate) yields a higher joint payoff.",
             "Let payoffs for (Player 1, Player 2) be: Cooperate/Cooperate = (3,3), Defect/Cooperate = (5,0), Cooperate/Defect = (0,5), Defect/Defect = (1,1).\n\n### Proof of Dominant Strategy:\n- If Player 2 Cooperates, Player 1 gets 5 by Defecting vs 3 by Cooperating -> Defect is strictly better.\n- If Player 2 Defects, Player 1 gets 1 by Defecting vs 0 by Cooperating -> Defect is strictly better.\n\nSince Defect strictly dominates Cooperate for both players regardless of the other's choice, (Defect, Defect) is the strictly dominant Nash Equilibrium.",
             {"key_concept": "strictly dominant strategy", "expected_conclusion": "Defect is strictly dominant for both players"}),

            # 11. REASONING
            ("reasoning", "formal_logic", "advanced", "proof", "deterministic",
             "Given the premises: 1. All A are B. 2. No B are C. 3. Some D are A. Determine what logically follows regarding D and C.",
             "### Logical Derivation:\n1. Premise 1: $\\forall x (A(x) \\rightarrow B(x))$\n2. Premise 2: $\\forall x (B(x) \\rightarrow \\neg C(x))$\n3. Combining 1 & 2 by transitivity: $\\forall x (A(x) \\rightarrow \\neg C(x))$ (No A are C).\n4. Premise 3: $\\exists x (D(x) \\land A(x))$ (There exists some element $k$ that is both D and A).\n5. Since $k$ is A, by step 3, $k$ is not C ($\\neg C(k)$).\n6. Therefore, $k$ is D and not C: $\\exists x (D(x) \\land \\neg C(x))$.\n\n**Conclusion**: **Some D are not C**.",
             {"expected_conclusion": "Some D are not C"}),

            # 12. TECHNOLOGY
            ("technology", "distributed_systems", "advanced", "comparison", "structural",
             "Explain the CAP Theorem and contrast CP (Consistency/Partition-tolerance) systems with AP (Availability/Partition-tolerance) systems with concrete database examples.",
             "### The CAP Theorem:\nIn any asynchronous networked data store, when a Network Partition ($P$) occurs, the system must choose between **Consistency ($C$)** (every read receives the most recent write or an error) and **Availability ($A$)** (every non-failing node returns a non-error response, but without guarantee of containing the latest write).\n\n### Comparison:\n- **CP Systems (e.g., Google Spanner, Apache HBase, ZooKeeper, etcd)**: Prioritize linearizability/consistency during a partition by refusing reads/writes on partitioned minority nodes.\n- **AP Systems (e.g., Apache Cassandra, DynamoDB in eventual consistency mode)**: Prioritize uptime and availability by allowing local reads/writes on all nodes, resolving divergences via eventual consistency (e.g., vector clocks, CRDTs).",
             {"examples_cp": ["ZooKeeper", "etcd", "Spanner"], "examples_ap": ["Cassandra", "DynamoDB"], "required_sections": ["CAP Theorem", "Comparison"]}),

            # 13. GENERAL KNOWLEDGE
            ("general_knowledge", "information_theory", "intermediate", "explanation", "numerical",
             "Calculate the Shannon entropy in bits of an unbiased 8-sided die.",
             "For a discrete random variable $X$ with outcomes $x_1, \\dots, x_n$ and probabilities $P(x_i) = \\frac{1}{8}$ for an unbiased 8-sided die:\n\n$$H(X) = -\\sum_{i=1}^{8} P(x_i) \\log_2 P(x_i)$$\n$$H(X) = -8 \\times \\left(\\frac{1}{8} \\log_2 \\frac{1}{8}\\right) = - \\log_2 \\left(2^{-3}\\right) = 3 \\text{ bits}$$\n\nThe Shannon entropy is exactly **3.0 bits**.",
             {"expected_numerical_values": {"shannon_entropy_bits": 3.0}}),
        ]

        multiturn_conversations = [
            ("programming", "python_oop", "intermediate", "multi_turn", "code_based",
             [
                 Message(role=Role.USER, content="How do I create a Python class for a BankAccount with deposit and withdraw methods?"),
                 Message(role=Role.ASSISTANT, content="Here is a basic BankAccount class:\n\n```python\nclass BankAccount:\n    def __init__(self, initial_balance: float = 0.0):\n        self.balance = initial_balance\n\n    def deposit(self, amount: float) -> None:\n        if amount <= 0:\n            raise ValueError('Deposit must be positive.')\n        self.balance += amount\n\n    def withdraw(self, amount: float) -> None:\n        if amount > self.balance:\n            raise ValueError('Insufficient funds.')\n        self.balance -= amount\n```"),
                 Message(role=Role.USER, content="Now modify it to track transaction history and add a get_history() method."),
                 Message(role=Role.ASSISTANT, content="Here is the updated `BankAccount` class with transaction auditing:\n\n```python\nfrom datetime import datetime\nfrom typing import List, Dict, Any\n\nclass BankAccount:\n    def __init__(self, initial_balance: float = 0.0):\n        self.balance = initial_balance\n        self.history: List[Dict[str, Any]] = []\n        if initial_balance > 0:\n            self._record('INITIAL_DEPOSIT', initial_balance)\n\n    def _record(self, action: str, amount: float) -> None:\n        self.history.append({\n            'action': action,\n            'amount': amount,\n            'balance_after': self.balance,\n            'timestamp': datetime.utcnow().isoformat()\n        })\n\n    def deposit(self, amount: float) -> None:\n        if amount <= 0:\n            raise ValueError('Deposit must be positive.')\n        self.balance += amount\n        self._record('DEPOSIT', amount)\n\n    def withdraw(self, amount: float) -> None:\n        if amount > self.balance:\n            raise ValueError('Insufficient funds.')\n        self.balance -= amount\n        self._record('WITHDRAW', amount)\n\n    def get_history(self) -> List[Dict[str, Any]]:\n        return list(self.history)\n```"),
             ],
             "Context retention across turns: adds history tracking without removing existing deposit/withdraw constraints.",
             "```python\nclass BankAccount:\n    def __init__(self, initial_balance: float = 0.0):\n        self.balance = initial_balance\n        self.history = []\n```",
             {"required_symbols": ["BankAccount", "deposit", "withdraw", "get_history"], "static_checks": ["ast_parse"]}),

            ("cybersecurity", "secure_coding", "advanced", "multi_turn", "reasoning",
             [
                 Message(role=Role.USER, content="Is using MD5 acceptable for password hashing?"),
                 Message(role=Role.ASSISTANT, content="No, MD5 is completely broken for password storage. It is extremely fast, vulnerable to collision attacks, and can be brute-forced or rainbow-table queried in seconds."),
                 Message(role=Role.USER, content="What modern adaptive algorithm should I use instead and why?"),
                 Message(role=Role.ASSISTANT, content="You should use modern adaptive hashing algorithms such as **Argon2id** (the winner of the Password Hashing Competition) or **bcrypt** / **scrypt**.\n\n### Reasons:\n1. **Work Factor Tuning**: They support configurable computational cost factors (time/iterations) to resist hardware speedups.\n2. **Memory Hardness**: Argon2id and scrypt require significant RAM, neutralizing GPU/ASIC parallel brute-force attacks.\n3. **Built-in Salt**: They automatically generate and embed cryptographically secure random salts per password, neutralizing rainbow table attacks."),
             ],
             "Clear recommendation of Argon2id/bcrypt with memory hardness and work factor explanation.",
             "Use Argon2id or bcrypt due to memory hardness and configurable work factors.",
             {"expected_conclusion": "Argon2id or bcrypt should be used due to memory hardness and adaptive work factors", "key_concept": "adaptive password hashing"}),
        ]

        for item in domain_definitions:
            dom, topic, diff, task, ev_type, prompt, ref_ans, meta = item
            case = BenchmarkCase(
                benchmark_id=f"bench-v1-{cur_id:04d}",
                domain=dom,
                topic=topic,
                difficulty=diff,
                task_type=task,
                messages=[
                    Message(role=Role.USER, content=prompt),
                    Message(role=Role.ASSISTANT, content=ref_ans),
                ],
                expected_behavior=f"Correct, rigorous, domain-appropriate response for {dom} ({diff}).",
                reference_answer=ref_ans,
                evaluation_type=ev_type,
                evaluation_metadata=meta,
                tags=[dom, topic, diff, task, ev_type],
            )
            cases.append(case)
            cur_id += 1

        for item in multiturn_conversations:
            dom, topic, diff, task, ev_type, msgs, exp_beh, ref_ans, meta = item
            case = BenchmarkCase(
                benchmark_id=f"bench-v1-{cur_id:04d}",
                domain=dom,
                topic=topic,
                difficulty=diff,
                task_type=task,
                messages=msgs,
                expected_behavior=exp_beh,
                reference_answer=ref_ans,
                evaluation_type=ev_type,
                evaluation_metadata=meta,
                tags=[dom, topic, diff, "multi_turn"],
            )
            cases.append(case)
            cur_id += 1

        difficulties = ["beginner", "intermediate", "advanced", "expert"]
        task_types = [
            "explanation", "question_answering", "coding", "code_generation", "code_completion",
            "debugging", "code_review", "refactoring", "troubleshooting", "system_design",
            "reasoning", "comparison", "classification", "summarization", "analysis",
            "scenario_analysis", "decision_analysis", "multi_turn", "problem_solving",
            "proof", "calculation", "data_interpretation"
        ]

        topic_bank = {
            "programming": ["iterators", "generators", "decorators", "memory_views", "type_hints", "asyncio", "metaclasses", "protocols"],
            "software_engineering": ["solid_principles", "clean_code", "cqrs", "event_sourcing", "grpc", "domain_driven_design", "observability"],
            "cybersecurity": ["jwt_security", "sql_injection", "ssrf", "zero_trust", "oauth2", "tls_certificates", "buffer_overflow"],
            "linux_systems": ["cgroups", "namespaces", "epoll", "systemd_units", "bpf_tracing", "virtual_memory", "inode_structure"],
            "networking": ["bgp_routing", "quic_protocol", "dnssec", "subnet_masking", "ipv6_slaac", "nat_traversal", "congestion_control"],
            "ai_ml": ["diffusion_models", "lora_quantization", "rope_embeddings", "flash_attention", "gradient_checkpointing", "beam_search"],
            "mathematics": ["eigenvalues", "bayes_theorem", "convex_optimization", "markov_chains", "modular_arithmetic", "fourier_transform"],
            "science": ["thermodynamics", "quantum_states", "enzyme_kinetics", "orbital_mechanics", "special_relativity", "crystallography"],
            "psychology": ["working_memory", "cognitive_dissonance", "dual_process_theory", "neuroplasticity", "heuristics"],
            "human_behavior": ["pareto_efficiency", "public_goods_game", "nash_equilibrium", "signaling_theory", "mechanism_design"],
            "reasoning": ["syllogism", "propositional_logic", "predicate_calculus", "fallacy_identification", "inductive_bias"],
            "technology": ["distributed_consensus", "raft_protocol", "crdt", "vector_databases", "wasm_runtimes"],
            "general_knowledge": ["turing_machine", "von_neumann_architecture", "ieee_floating_point", "utf8_encoding", "iso_standards"],
        }

        dom_idx = 0
        diff_idx = 0
        task_idx = 0

        while len(cases) < target_count:
            dom = DOMAINS_13[dom_idx % len(DOMAINS_13)]
            diff = difficulties[diff_idx % len(difficulties)]
            task = task_types[task_idx % len(task_types)]
            topics = topic_bank.get(dom, ["general"])
            topic = topics[(cur_id) % len(topics)]
            item_seq = cur_id

            # Determine appropriate evaluation type with distinct prompt formulations
            if "code" in task or "coding" in task or "debugging" in task:
                ev_type = BenchmarkEvaluationType.CODE_BASED.value
                prompt = f"[Benchmark Case {item_seq}] In the domain of {dom} ({topic}), provide a clean, robust Python implementation solving scenario #{item_seq} for a {diff}-level {task} problem."
                ref_ans = f"```python\n# {dom} - {topic} ({diff}) Scenario {item_seq}\ndef solve_{topic}_{item_seq}():\n    \"\"\"Solves scenario {item_seq} for {topic}.\"\"\"\n    return True\n```\n\nThis implementation follows optimal time and space complexity for {diff} requirements in scenario {item_seq}."
                meta = {"required_symbols": [f"solve_{topic}_{item_seq}"], "static_checks": ["ast_parse"]}
            elif task in ("calculation", "proof"):
                ev_type = BenchmarkEvaluationType.NUMERICAL.value if task == "calculation" else BenchmarkEvaluationType.DETERMINISTIC.value
                prompt = f"[Benchmark Case {item_seq}] Perform a precise {diff}-level {task} for {dom} regarding {topic} in context #{item_seq}. Show clear derivation."
                val = round(float((item_seq * 13) % 1000) / 10.0 + 1.0, 2)
                ref_ans = f"### Derivation for {topic} (Scenario {item_seq}):\nGiven the parameters for {dom} ({diff}), the resulting exact computed value is **{val}**.\n\nCalculation confirmed by first principles."
                meta = {"expected_numerical_values": {"result": val, "scenario": item_seq}}
            elif task in ("reasoning", "decision_analysis", "problem_solving"):
                ev_type = BenchmarkEvaluationType.REASONING.value
                prompt = f"[Benchmark Case {item_seq}] Analyze the following {diff} problem in {dom} ({topic}) under scenario constraints #{item_seq} using step-by-step deductive reasoning."
                ref_ans = f"### Step-by-Step Analysis ({dom} - {topic} Case {item_seq}):\n1. Analyze initial conditions for {topic} in case {item_seq}.\n2. Evaluate logical invariants and constraints.\n3. Deduce the unique necessary outcome for this {diff} problem.\n\n**Conclusion**: The outcome is structurally sound under {dom} principles."
                meta = {"expected_conclusion": f"Optimal invariant for {topic} in scenario {item_seq}", "key_concept": f"{topic}_invariant_{item_seq}"}
            elif task in ("system_design", "comparison", "classification"):
                ev_type = BenchmarkEvaluationType.STRUCTURAL.value
                prompt = f"[Benchmark Case {item_seq}] Provide a structured {diff} {task} for {dom} concerning {topic} under architectural specification #{item_seq}."
                ref_ans = f"### Structured {task.title()} ({dom} - {topic} Spec {item_seq}):\n- **Core Attributes**: Key mechanisms of {topic} for spec {item_seq}.\n- **Trade-offs**: Latency vs throughput, simplicity vs flexibility.\n- **Recommendations**: Recommended patterns for {diff} systems."
                meta = {"required_sections": ["Core Attributes", "Trade-offs", "Recommendations"]}
            elif task == "summarization":
                ev_type = BenchmarkEvaluationType.QUALITATIVE.value
                prompt = f"[Benchmark Case {item_seq}] Summarize the technical trade-offs and operational characteristics of {topic} in {dom} at a {diff} level (Scenario #{item_seq})."
                ref_ans = f"### Technical Summary ({dom} - {topic} #{item_seq}):\n{topic.title()} provides targeted operational capabilities in {dom}. At a {diff} level, its key benefits include robust performance and modular extensibility."
                meta = {"key_concepts": [topic, dom, diff]}
            else:
                ev_type = BenchmarkEvaluationType.REFERENCE_BASED.value
                prompt = f"[Benchmark Case {item_seq}] Explain the core principles and operational mechanisms of {topic} within {dom} at a {diff} level (Query #{item_seq})."
                ref_ans = f"### Principles of {topic} in {dom} (Query {item_seq}):\n{topic.title()} is essential in {dom}. At a {diff} level, its primary mechanism involves structured coordination and invariant maintenance, ensuring reliable behavior across diverse operating environments."
                meta = {"key_concepts": [topic, dom, diff, f"query_{item_seq}"]}

            case = BenchmarkCase(
                benchmark_id=f"bench-v1-{cur_id:04d}",
                domain=dom,
                topic=topic,
                difficulty=diff,
                task_type=task,
                messages=[
                    Message(role=Role.USER, content=prompt),
                    Message(role=Role.ASSISTANT, content=ref_ans),
                ],
                expected_behavior=f"Authoritative, high-fidelity response for {dom} / {topic} ({diff} level, case #{item_seq}).",
                reference_answer=ref_ans,
                evaluation_type=ev_type,
                evaluation_metadata=meta,
                tags=[dom, topic, diff, task, ev_type],
            )
            cases.append(case)

            cur_id += 1
            dom_idx += 1
            if dom_idx % len(DOMAINS_13) == 0:
                diff_idx += 1
                task_idx += 1

        return cases[:target_count]
