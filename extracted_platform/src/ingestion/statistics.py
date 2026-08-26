"""
Ingestion Statistics and Reporting Engine (Phase 3.3).
Aggregates telemetry, counts, distributions, and produces formatted JSON/Markdown audit reports.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.ingestion.models import IngestionDocument, KnowledgeChunk


@dataclass
class IngestionStatistics:
    """Aggregated metrics from an ingestion execution run."""
    documents_discovered: int = 0
    documents_processed: int = 0
    documents_successful: int = 0
    documents_partial: int = 0
    documents_failed: int = 0
    documents_duplicate: int = 0
    pdf_count: int = 0
    html_count: int = 0
    json_count: int = 0
    total_pages: int = 0
    total_characters: int = 0
    total_sections: int = 0
    total_chunks: int = 0
    total_equations: int = 0
    total_tables: int = 0
    domain_distribution: Counter = field(default_factory=Counter)
    topic_distribution: Counter = field(default_factory=Counter)
    license_distribution: Counter = field(default_factory=Counter)
    quality_distribution: Counter = field(default_factory=Counter)
    processing_duration_seconds: float = 0.0

    def record_document(self, doc: IngestionDocument) -> None:
        """Updates metrics from a processed IngestionDocument."""
        self.documents_processed += 1
        if doc.format == "pdf":
            self.pdf_count += 1
        elif doc.format == "html":
            self.html_count += 1
        elif doc.format in ["json", "jsonl"]:
            self.json_count += 1

        self.total_pages += doc.telemetry.pages_total
        self.total_characters += doc.telemetry.characters_extracted
        self.total_sections += len(doc.sections)
        self.total_equations += sum(len(s.equations) for s in doc.sections)
        self.total_tables += sum(len(s.tables) for s in doc.sections)

        status = doc.telemetry.extraction_status.value
        if status == "SUCCESS":
            self.documents_successful += 1
        elif status == "PARTIAL":
            self.documents_partial += 1
        else:
            self.documents_failed += 1

        self.domain_distribution[doc.metadata.domain] += 1
        self.topic_distribution[doc.metadata.topic] += 1
        self.license_distribution[doc.metadata.license or "UNKNOWN"] += 1

    def record_chunks(self, chunks: List[KnowledgeChunk]) -> None:
        """Updates metrics from generated knowledge chunks."""
        self.total_chunks += len(chunks)
        for c in chunks:
            if c.quality_score >= 0.85:
                self.quality_distribution["high_quality (>=0.85)"] += 1
            elif c.quality_score >= 0.50:
                self.quality_distribution["medium_quality (0.50-0.84)"] += 1
            else:
                self.quality_distribution["low_quality (<0.50)"] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents_discovered": self.documents_discovered,
            "documents_processed": self.documents_processed,
            "documents_successful": self.documents_successful,
            "documents_partial": self.documents_partial,
            "documents_failed": self.documents_failed,
            "documents_duplicate": self.documents_duplicate,
            "format_breakdown": {
                "pdf": self.pdf_count,
                "html": self.html_count,
                "json": self.json_count,
            },
            "total_pages": self.total_pages,
            "total_characters": self.total_characters,
            "total_sections": self.total_sections,
            "total_chunks": self.total_chunks,
            "total_equations": self.total_equations,
            "total_tables": self.total_tables,
            "domain_distribution": dict(self.domain_distribution),
            "topic_distribution": dict(self.topic_distribution),
            "license_distribution": dict(self.license_distribution),
            "quality_distribution": dict(self.quality_distribution),
            "processing_duration_seconds": round(self.processing_duration_seconds, 2),
        }

    def generate_markdown_report(self, title: str = "Knowledge Ingestion Report") -> str:
        """Renders an executive Markdown summary report."""
        lines = [
            f"# {title}",
            "",
            "## 1. Document Extraction Summary",
            "",
            f"- **Documents Discovered:** {self.documents_discovered}",
            f"- **Documents Processed:** {self.documents_processed}",
            f"- **Successful:** {self.documents_successful}",
            f"- **Partial:** {self.documents_partial}",
            f"- **Failed:** {self.documents_failed}",
            f"- **Duplicates Filtered:** {self.documents_duplicate}",
            f"- **Format Breakdown:** PDF ({self.pdf_count}), HTML ({self.html_count}), JSON ({self.json_count})",
            "",
            "## 2. Content & Structural Metrics",
            "",
            f"- **Total Pages:** {self.total_pages}",
            f"- **Total Characters:** {self.total_characters:,}",
            f"- **Total Sections:** {self.total_sections}",
            f"- **Total Chunks:** {self.total_chunks}",
            f"- **Equations Detected:** {self.total_equations}",
            f"- **Tables Extracted:** {self.total_tables}",
            f"- **Processing Duration:** {self.processing_duration_seconds:.2f}s",
            "",
            "## 3. Domain & Topic Distribution",
            "",
            "| Domain / Topic | Count |",
            "|---|---|",
        ]
        for domain, cnt in self.domain_distribution.most_common():
            lines.append(f"| **Domain: {domain}** | {cnt} |")
        for topic, cnt in self.topic_distribution.most_common():
            lines.append(f"| Topic: {topic} | {cnt} |")

        lines.extend([
            "",
            "## 4. License & Rights Distribution",
            "",
            "| License | Count |",
            "|---|---|",
        ])
        for lic, cnt in self.license_distribution.most_common():
            lines.append(f"| {lic} | {cnt} |")

        lines.extend([
            "",
            "## 5. Quality Distribution",
            "",
            "| Tier | Count |",
            "|---|---|",
        ])
        for q, cnt in self.quality_distribution.items():
            lines.append(f"| {q} | {cnt} |")

        return "\n".join(lines)
