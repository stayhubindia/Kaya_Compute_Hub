"""
Domain Dataset Template Architecture.
Defines canonical TaskTemplate models, validation engines, and registry logic
for managing modular, domain-specific generation and training data templates (Phase 2.3.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.dataset.schema import DifficultyLevel, TaskType


class TaskTemplate(BaseModel):
    """Declarative task template for domain-specific conversational data synthesis and evaluation."""

    id: str
    domain: str
    topic: str
    task_type: str
    difficulty: str
    objective: str
    description: str
    supported_difficulties: List[str] = Field(default_factory=lambda: [d.value for d in DifficultyLevel])
    quality_requirements: Dict[str, Any] = Field(default_factory=dict)
    prompt_guidelines: Optional[List[str]] = None
    prompt_templates: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        clean = v.strip().lower()
        if not clean:
            raise ValueError("Template ID must be a non-empty string.")
        return clean

    @field_validator("domain", "topic", "objective", "description")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Field must be a non-empty string.")
        return clean

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, v: Any) -> str:
        if isinstance(v, DifficultyLevel):
            return v.value
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in [d.value for d in DifficultyLevel]:
                return clean
        raise ValueError(f"Invalid difficulty '{v}'. Allowed: {[d.value for d in DifficultyLevel]}")

    @field_validator("task_type", mode="before")
    @classmethod
    def normalize_task_type(cls, v: Any) -> str:
        if isinstance(v, TaskType):
            return v.value
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in [t.value for t in TaskType]:
                return clean
        raise ValueError(f"Invalid task_type '{v}'. Allowed: {[t.value for t in TaskType]}")

    @field_validator("supported_difficulties", mode="before")
    @classmethod
    def validate_supported_difficulties(cls, v: Any) -> List[str]:
        if not v:
            return [d.value for d in DifficultyLevel]
        allowed = {d.value for d in DifficultyLevel}
        cleaned = [str(item).strip().lower() for item in v]
        for item in cleaned:
            if item not in allowed:
                raise ValueError(f"Invalid difficulty '{item}' in supported_difficulties. Allowed: {sorted(allowed)}")
        return cleaned

    @model_validator(mode="after")
    def validate_difficulty_compatibility(self) -> TaskTemplate:
        if self.difficulty not in self.supported_difficulties:
            self.supported_difficulties.append(self.difficulty)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "topic": self.topic,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "objective": self.objective,
            "description": self.description,
            "supported_difficulties": self.supported_difficulties,
            "quality_requirements": self.quality_requirements,
            "prompt_guidelines": self.prompt_guidelines,
            "prompt_templates": self.prompt_templates,
            "metadata": self.metadata,
        }


class TemplateRegistry:
    """In-memory registry and query engine for TaskTemplate specifications."""

    def __init__(self, templates: Optional[List[TaskTemplate]] = None):
        self._templates: Dict[str, TaskTemplate] = {}
        if templates:
            for t in templates:
                self.register_template(t)

    def register_template(self, template: TaskTemplate, overwrite: bool = False) -> None:
        """Registers a TaskTemplate. Rejects duplicates unless overwrite=True."""
        if not isinstance(template, TaskTemplate):
            raise TypeError(f"Expected TaskTemplate, got {type(template).__name__}")

        if template.id in self._templates and not overwrite:
            raise ValueError(f"Template with ID '{template.id}' is already registered.")

        self._templates[template.id] = template

    def get_template(self, template_id: str) -> TaskTemplate:
        """Retrieves template by ID or raises KeyError."""
        clean_id = template_id.strip().lower()
        if clean_id not in self._templates:
            raise KeyError(f"Template with ID '{clean_id}' not found in registry.")
        return self._templates[clean_id]

    def lookup_template(self, template_id: str) -> Optional[TaskTemplate]:
        """Retrieves template by ID or returns None."""
        return self._templates.get(template_id.strip().lower())

    def list_templates(self) -> List[TaskTemplate]:
        """Returns all registered templates."""
        return list(self._templates.values())

    def list_by_domain(self, domain: str) -> List[TaskTemplate]:
        """Filters templates by domain."""
        target = domain.strip().lower()
        return [t for t in self._templates.values() if t.domain.lower() == target]

    def list_by_task_type(self, task_type: str) -> List[TaskTemplate]:
        """Filters templates by task_type."""
        target = task_type.strip().lower()
        return [t for t in self._templates.values() if t.task_type.lower() == target]

    def list_by_difficulty(self, difficulty: str) -> List[TaskTemplate]:
        """Filters templates that support or default to the given difficulty."""
        target = difficulty.strip().lower()
        return [
            t for t in self._templates.values()
            if target == t.difficulty.lower() or target in [d.lower() for d in t.supported_difficulties]
        ]

    def list_by_topic(self, domain: str, topic: str) -> List[TaskTemplate]:
        """Filters templates by domain and topic."""
        d_target = domain.strip().lower()
        t_target = topic.strip().lower()
        return [
            t for t in self._templates.values()
            if t.domain.lower() == d_target and t.topic.lower() == t_target
        ]

    def load_manifest(self, manifest_path: Union[str, Path]) -> int:
        """Loads and registers task templates from a YAML manifest file."""
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"Template manifest file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        templates_list = raw_data.get("templates", [])
        if not isinstance(templates_list, list):
            raise ValueError(f"Expected 'templates' key to contain a list, got {type(templates_list).__name__}")

        count = 0
        for item in templates_list:
            tmpl = TaskTemplate.model_validate(item)
            self.register_template(tmpl, overwrite=True)
            count += 1

        return count

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> TemplateRegistry:
        """Instantiates and populates a TemplateRegistry from a YAML manifest."""
        registry = cls()
        registry.load_manifest(yaml_path)
        return registry

    def validate(self, dataset_config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Validates all registered templates against global dataset taxonomy.
        Checks domain validity, task type validity, difficulty levels, and subtopics.
        """
        errors: List[str] = []
        known_domains: Set[str] = set()
        known_topics_by_domain: Dict[str, Set[str]] = {}
        known_task_types: Set[str] = {t.value for t in TaskType}
        known_difficulties: Set[str] = {d.value for d in DifficultyLevel}

        # If dataset config path provided, extract authoritative taxonomy
        if dataset_config_path:
            cfg_path = Path(dataset_config_path)
            if cfg_path.is_file():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                domains_dict = cfg.get("domains", {})
                known_domains = set(domains_dict.keys())
                for d_name, d_info in domains_dict.items():
                    if isinstance(d_info, dict):
                        known_topics_by_domain[d_name] = set(d_info.get("subtopics", []))

                cfg_tasks = cfg.get("task_types", [])
                if cfg_tasks:
                    known_task_types = set(cfg_tasks)

        for template_id, tmpl in self._templates.items():
            # Domain check
            if known_domains and tmpl.domain not in known_domains:
                errors.append(
                    f"Template '{template_id}': Unknown domain '{tmpl.domain}'. Allowed: {sorted(known_domains)}"
                )

            # Topic check
            if tmpl.domain in known_topics_by_domain:
                allowed_topics = known_topics_by_domain[tmpl.domain]
                if allowed_topics and tmpl.topic not in allowed_topics:
                    errors.append(
                        f"Template '{template_id}': Unknown topic '{tmpl.topic}' for domain '{tmpl.domain}'. Allowed: {sorted(allowed_topics)}"
                    )

            # Task type check
            if tmpl.task_type not in known_task_types:
                errors.append(
                    f"Template '{template_id}': Unknown task_type '{tmpl.task_type}'. Allowed: {sorted(known_task_types)}"
                )

            # Difficulty check
            if tmpl.difficulty not in known_difficulties:
                errors.append(
                    f"Template '{template_id}': Invalid difficulty '{tmpl.difficulty}'. Allowed: {sorted(known_difficulties)}"
                )

            for d in tmpl.supported_difficulties:
                if d not in known_difficulties:
                    errors.append(
                        f"Template '{template_id}': Invalid supported difficulty '{d}'. Allowed: {sorted(known_difficulties)}"
                    )

            # Description check
            if not tmpl.description or not tmpl.description.strip():
                errors.append(f"Template '{template_id}': Description must not be empty.")

            if not tmpl.objective or not tmpl.objective.strip():
                errors.append(f"Template '{template_id}': Objective must not be empty.")

        is_valid = len(errors) == 0
        return {
            "is_valid": is_valid,
            "template_count": len(self._templates),
            "errors": errors,
        }

    def template_statistics(self) -> Dict[str, Any]:
        """Calculates distribution statistics across domains, task types, and difficulty levels."""
        total = len(self._templates)
        by_domain: Dict[str, int] = {}
        by_task_type: Dict[str, int] = {}
        by_difficulty: Dict[str, int] = {}

        for tmpl in self._templates.values():
            by_domain[tmpl.domain] = by_domain.get(tmpl.domain, 0) + 1
            by_task_type[tmpl.task_type] = by_task_type.get(tmpl.task_type, 0) + 1
            by_difficulty[tmpl.difficulty] = by_difficulty.get(tmpl.difficulty, 0) + 1

        domain_percentages: Dict[str, float] = {
            k: round((v / total * 100.0), 2) if total > 0 else 0.0
            for k, v in by_domain.items()
        }

        return {
            "total_templates": total,
            "by_domain": by_domain,
            "by_task_type": by_task_type,
            "by_difficulty": by_difficulty,
            "domain_percentages": domain_percentages,
        }

    def __len__(self) -> int:
        return len(self._templates)

    def __repr__(self) -> str:
        return f"<TemplateRegistry(total_templates={len(self._templates)})>"
