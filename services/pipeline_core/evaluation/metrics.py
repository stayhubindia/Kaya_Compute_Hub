"""
Response Quality & Performance Metrics (Phase 4.4).
Implements deterministic quality metrics, n-gram repetition analysis,
formatting validation, truncation detection, and statistical aggregation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from pydantic import BaseModel, Field

from src.evaluation.config import MetricsConfig
from src.evaluation.inference import EvaluationInferenceResult


class SampleMetrics(BaseModel):
    """Metrics computed for a single evaluated response."""
    record_id: str
    is_valid: bool
    is_empty: bool
    char_length: int
    word_length: int
    token_length: int
    repetition_ratio_2gram: float
    repetition_ratio_3gram: float
    repetition_ratio_4gram: float
    repetition_ratio: float
    unique_word_ratio: float
    has_repeated_lines: bool
    is_truncated: bool
    formatting_score: float
    exact_match: bool
    keyword_overlap: float


class AggregatedMetrics(BaseModel):
    """Statistical summary across a collection of evaluated samples."""
    total_samples: int = 0
    valid_responses: int = 0
    validity_rate: float = 0.0
    empty_responses: int = 0
    empty_rate: float = 0.0
    avg_char_length: float = 0.0
    avg_word_length: float = 0.0
    avg_token_length: float = 0.0
    avg_repetition_ratio: float = 0.0
    avg_unique_word_ratio: float = 0.0
    repeated_lines_rate: float = 0.0
    truncation_rate: float = 0.0
    avg_formatting_score: float = 0.0
    exact_match_rate: float = 0.0
    avg_keyword_overlap: float = 0.0
    avg_latency_seconds: float = 0.0
    avg_tokens_per_second: float = 0.0
    p50_token_length: float = 0.0
    p90_token_length: float = 0.0
    p95_token_length: float = 0.0
    p99_token_length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class MetricCalculator:
    """Computes deterministic and heuristic metrics on evaluation outputs."""

    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()

    def _compute_ngram_repetition(self, tokens: List[str], n: int) -> float:
        """Calculate repetition ratio for n-grams in token stream."""
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        unique_ngrams = set(ngrams)
        # Ratio of repeated ngrams to total ngrams
        repeated_count = len(ngrams) - len(unique_ngrams)
        return float(repeated_count / len(ngrams))

    def _check_repeated_lines(self, text: str) -> bool:
        """Detect degenerate repetitive loops (consecutive duplicate lines)."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) < 2:
            return False
        for i in range(len(lines) - 1):
            if lines[i] == lines[i + 1]:
                return True
        return False

    def _check_truncation(self, text: str) -> bool:
        """Detect if generated text ends abruptly without terminal punctuation or closing tag."""
        stripped = text.strip()
        if not stripped:
            return False
        # Valid terminal markers
        valid_terminals = (".", "!", "?", "\n", "```", "'''", "}", "]", ")", ">", '"', "'", ";")
        return not stripped.endswith(valid_terminals)

    def _compute_formatting_score(self, text: str) -> float:
        """
        Evaluate structural balance of delimiters (code blocks, quotes, brackets).
        Returns a score between 0.0 (severely broken) and 1.0 (fully balanced).
        """
        if not text.strip():
            return 0.0

        score = 1.0
        # 1. Backtick code blocks balance
        backticks = text.count("```")
        if backticks % 2 != 0:
            score -= 0.3

        # 2. Bracket balances
        for open_char, close_char in (("{", "}"), ("[", "]"), ("(", ")")):
            open_count = text.count(open_char)
            close_count = text.count(close_char)
            if open_count != close_count:
                score -= 0.1

        return max(0.0, min(1.0, round(score, 2)))

    def _compute_keyword_overlap(self, generated: str, reference: str) -> float:
        """Compute Jaccard word token overlap between generated and reference text."""
        gen_words = set(re.findall(r"\w+", generated.lower()))
        ref_words = set(re.findall(r"\w+", reference.lower()))
        if not ref_words:
            return 1.0 if not gen_words else 0.0
        intersection = gen_words.intersection(ref_words)
        union = gen_words.union(ref_words)
        return float(len(intersection) / len(union)) if union else 0.0

    def calculate_sample_metrics(self, result: EvaluationInferenceResult) -> SampleMetrics:
        """Compute metrics for a single inference result."""
        text = result.generated_text
        ref = result.reference_text
        stripped = text.strip()

        is_empty = len(stripped) == 0
        is_valid = not is_empty and any(c.isalnum() for c in stripped)

        words = stripped.split()
        char_len = len(stripped)
        word_len = len(words)
        token_len = result.tokens_generated if result.tokens_generated > 0 else word_len

        # Repetition
        rep_2 = self._compute_ngram_repetition(words, 2)
        rep_3 = self._compute_ngram_repetition(words, 3)
        rep_4 = self._compute_ngram_repetition(words, 4)
        avg_rep = round(float(np.mean([rep_2, rep_3, rep_4])), 4)

        unique_word_ratio = round(float(len(set(words)) / word_len), 4) if word_len > 0 else 0.0
        has_repeated_lines = self._check_repeated_lines(text)
        is_truncated = self._check_truncation(text)
        fmt_score = self._compute_formatting_score(text)

        # Reference matching
        exact_match = (stripped.lower() == ref.strip().lower()) if ref else False
        keyword_overlap = round(self._compute_keyword_overlap(text, ref), 4)

        return SampleMetrics(
            record_id=result.record_id,
            is_valid=is_valid,
            is_empty=is_empty,
            char_length=char_len,
            word_length=word_len,
            token_length=token_len,
            repetition_ratio_2gram=round(rep_2, 4),
            repetition_ratio_3gram=round(rep_3, 4),
            repetition_ratio_4gram=round(rep_4, 4),
            repetition_ratio=avg_rep,
            unique_word_ratio=unique_word_ratio,
            has_repeated_lines=has_repeated_lines,
            is_truncated=is_truncated,
            formatting_score=fmt_score,
            exact_match=exact_match,
            keyword_overlap=keyword_overlap,
        )

    def aggregate_metrics(
        self,
        samples: List[SampleMetrics],
        inference_results: Optional[List[EvaluationInferenceResult]] = None,
    ) -> AggregatedMetrics:
        """Aggregate statistical summary across a list of SampleMetrics."""
        if not samples:
            return AggregatedMetrics()

        n = len(samples)
        valid_count = sum(1 for s in samples if s.is_valid)
        empty_count = sum(1 for s in samples if s.is_empty)
        repeated_lines_count = sum(1 for s in samples if s.has_repeated_lines)
        truncated_count = sum(1 for s in samples if s.is_truncated)
        exact_match_count = sum(1 for s in samples if s.exact_match)

        char_lens = [s.char_length for s in samples]
        word_lens = [s.word_length for s in samples]
        token_lens = [s.token_length for s in samples]
        rep_ratios = [s.repetition_ratio for s in samples]
        uniq_ratios = [s.unique_word_ratio for s in samples]
        fmt_scores = [s.formatting_score for s in samples]
        overlaps = [s.keyword_overlap for s in samples]

        # Latency & throughput
        latencies: List[float] = []
        tps_list: List[float] = []
        if inference_results:
            latencies = [r.latency_seconds for r in inference_results]
            tps_list = [r.tokens_per_second for r in inference_results]

        return AggregatedMetrics(
            total_samples=n,
            valid_responses=valid_count,
            validity_rate=round(valid_count / n, 4),
            empty_responses=empty_count,
            empty_rate=round(empty_count / n, 4),
            avg_char_length=round(float(np.mean(char_lens)), 2),
            avg_word_length=round(float(np.mean(word_lens)), 2),
            avg_token_length=round(float(np.mean(token_lens)), 2),
            avg_repetition_ratio=round(float(np.mean(rep_ratios)), 4),
            avg_unique_word_ratio=round(float(np.mean(uniq_ratios)), 4),
            repeated_lines_rate=round(repeated_lines_count / n, 4),
            truncation_rate=round(truncated_count / n, 4),
            avg_formatting_score=round(float(np.mean(fmt_scores)), 4),
            exact_match_rate=round(exact_match_count / n, 4),
            avg_keyword_overlap=round(float(np.mean(overlaps)), 4),
            avg_latency_seconds=round(float(np.mean(latencies)), 4) if latencies else 0.0,
            avg_tokens_per_second=round(float(np.mean(tps_list)), 2) if tps_list else 0.0,
            p50_token_length=round(float(np.percentile(token_lens, 50)), 2),
            p90_token_length=round(float(np.percentile(token_lens, 90)), 2),
            p95_token_length=round(float(np.percentile(token_lens, 95)), 2),
            p99_token_length=round(float(np.percentile(token_lens, 99)), 2),
        )
