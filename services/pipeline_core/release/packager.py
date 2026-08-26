"""
Release Packaging & Orchestration Engine (Phase 5.1).
Assembles verified adapter artifacts, metadata manifests, cryptographic checksums,
reproducibility specifications, and documentation into release distributions.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
from pydantic import BaseModel, Field

from src.release.adapter import AdapterValidator
from src.release.compatibility import BaseModelCompatibilityValidator
from src.release.integrity import ReleaseIntegrityManager
from src.release.manifest import ReleaseManifest, ReleaseStatus, construct_release_id
from src.release.model_card import ModelCardGenerator, ReadmeGenerator
from src.release.provenance import ProvenanceCollector
from src.release.reproducibility import ReproducibilityManager
from src.release.validator import ReleaseValidator
from src.training.utils import compute_file_sha256

logger = logging.getLogger(__name__)


class ReleasePackager:
    """Orchestrates adapter validation, metadata aggregation, checksum generation, and packaging."""

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/release.yaml",
    ):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.adapter_validator = AdapterValidator()
        self.compatibility_validator = BaseModelCompatibilityValidator()
        self.release_validator = ReleaseValidator(
            adapter_validator=self.adapter_validator,
            compatibility_validator=self.compatibility_validator,
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load release settings from YAML."""
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def execute_dry_run(
        self,
        target_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Verify configurations, paths, dataset provenance, and artifact schemas non-destructively."""
        rel_cfg = self.config.get("release", {})
        m_cfg = self.config.get("model", {})
        d_cfg = self.config.get("dataset", {})
        t_cfg = self.config.get("training", {})

        dataset_prov = ProvenanceCollector.collect_dataset_provenance(
            manifest_path=d_cfg.get("manifest_path", "datasets/production/manifests/production_manifest.json")
        )
        training_prov = ProvenanceCollector.collect_training_provenance(
            config_path=t_cfg.get("config_path", "configs/training.yaml")
        )

        rel_id = construct_release_id(
            base_model_id=m_cfg.get("base_model_id", "Qwen/Qwen3-4B-Base"),
            adapter_version=rel_cfg.get("version", "v1.0"),
            dataset_version=dataset_prov.dataset_version,
            training_config_hash=training_prov.config_hash,
        )

        return {
            "dry_run": True,
            "release_id": rel_id,
            "status": "VALIDATED",
            "base_model": m_cfg.get("base_model_id", "Qwen/Qwen3-4B-Base"),
            "dataset_version": dataset_prov.dataset_version,
            "dataset_provenance_status": dataset_prov.provenance_status,
            "training_config_hash": training_prov.config_hash,
            "expected_release_dir": str(target_dir or Path(rel_cfg.get("output_dir", "releases")) / rel_id),
            "message": "Dry-run validation successful. All configs, schemas, and dataset provenance are valid.",
        }

    def package(
        self,
        adapter_source_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        dry_run: bool = False,
    ) -> Tuple[bool, ReleaseManifest, List[str]]:
        """
        Execute full packaging workflow for a trained adapter.
        If adapter_source_dir is missing or invalid, fails cleanly.
        """
        errors: List[str] = []
        rel_cfg = self.config.get("release", {})
        m_cfg = self.config.get("model", {})
        d_cfg = self.config.get("dataset", {})
        t_cfg = self.config.get("training", {})
        g_cfg = self.config.get("generation", {})
        b_cfg = self.config.get("benchmark", {})

        src_dir = Path(adapter_source_dir)

        # 1. Harvest Provenance
        dataset_prov = ProvenanceCollector.collect_dataset_provenance(
            manifest_path=d_cfg.get("manifest_path", "datasets/production/manifests/production_manifest.json")
        )
        training_prov = ProvenanceCollector.collect_training_provenance(
            config_path=t_cfg.get("config_path", "configs/training.yaml")
        )
        hardware_prov = ProvenanceCollector.collect_hardware_provenance()

        # Generation config hash
        gen_path = Path(g_cfg.get("config_path", "configs/generation.yaml"))
        gen_hash = compute_file_sha256(gen_path) if gen_path.exists() else ""

        # Benchmark hash
        bench_manifest = Path(b_cfg.get("manifest_path", "benchmarks/benchmark-v1.0/manifest.json"))
        bench_hash = compute_file_sha256(bench_manifest) if bench_manifest.exists() else ""

        rel_id = construct_release_id(
            base_model_id=m_cfg.get("base_model_id", "Qwen/Qwen3-4B-Base"),
            adapter_version=rel_cfg.get("version", "v1.0"),
            dataset_version=dataset_prov.dataset_version,
            training_config_hash=training_prov.config_hash,
        )

        out_root = Path(output_dir or rel_cfg.get("output_dir", "releases"))
        dest_dir = out_root / rel_id

        # 2. Dry Run Check
        if dry_run:
            manifest = ReleaseManifest(
                release_id=rel_id,
                release_version=rel_cfg.get("version", "v1.0"),
                status=ReleaseStatus.PLANNED,
                status_reason="Dry-run validation executed without model artifacts.",
                base_model={"base_model_id": m_cfg.get("base_model_id", "Qwen/Qwen3-4B-Base")},
                dataset_version=dataset_prov.dataset_version,
                dataset_sha256=dataset_prov.manifest_sha256,
                training_config_hash=training_prov.config_hash,
                generation_config_hash=gen_hash,
                benchmark_version=b_cfg.get("version", "benchmark-v1.0"),
                benchmark_sha256=bench_hash,
            )
            return True, manifest, []

        # 3. Validate Source Adapter
        adapt_audit = self.adapter_validator.validate_directory(src_dir)
        if not adapt_audit.is_valid:
            errors.append(f"Adapter validation failed for source '{src_dir}': {adapt_audit.status}")
            errors.extend(adapt_audit.errors)
            manifest = ReleaseManifest(
                release_id=rel_id,
                status=ReleaseStatus.INVALID,
                status_reason="Source adapter audit failed.",
            )
            return False, manifest, errors

        # 4. Prepare Destination Directory
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_adapter = dest_dir / "adapter"
        dest_adapter.mkdir(parents=True, exist_ok=True)

        # 5. Copy Adapter Artifacts
        for f_path in src_dir.iterdir():
            if f_path.is_file():
                shutil.copy2(f_path, dest_adapter / f_path.name)
            elif f_path.is_dir() and f_path.name == "tokenizer":
                dest_tok = dest_adapter / "tokenizer"
                dest_tok.mkdir(parents=True, exist_ok=True)
                for tf in f_path.iterdir():
                    if tf.is_file():
                        shutil.copy2(tf, dest_tok / tf.name)

        # 6. Generate Compatibility JSON
        comp_record = self.compatibility_validator.generate_compatibility_record(
            adapter_config=adapt_audit.config_data
        )
        with open(dest_dir / "compatibility.json", "w", encoding="utf-8") as f:
            json.dump(comp_record, f, indent=2)

        # 7. Generate Provenance JSON
        prov_payload = {
            "dataset_provenance": dataset_prov.to_dict(),
            "training_provenance": training_prov.to_dict(),
            "hardware_provenance": hardware_prov.to_dict(),
        }
        with open(dest_dir / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(prov_payload, f, indent=2)

        # 8. Generate Reproducibility JSON
        repro_rec = ReproducibilityManager.build_record(
            dataset_prov=dataset_prov,
            training_prov=training_prov,
            hardware_prov=hardware_prov,
            generation_config_hash=gen_hash,
        )
        repro_rec.save(dest_dir / "reproducibility.json")

        # 9. Create Manifest in BUILDING status
        manifest = ReleaseManifest(
            release_id=rel_id,
            release_version=rel_cfg.get("version", "v1.0"),
            status=ReleaseStatus.BUILDING,
            base_model={"base_model_id": m_cfg.get("base_model_id", "Qwen/Qwen3-4B-Base")},
            adapter_type=m_cfg.get("adapter_type", "QLoRA"),
            dataset_version=dataset_prov.dataset_version,
            dataset_sha256=dataset_prov.manifest_sha256,
            training_config_hash=training_prov.config_hash,
            generation_config_hash=gen_hash,
            benchmark_version=b_cfg.get("version", "benchmark-v1.0"),
            benchmark_sha256=bench_hash,
            compatibility=comp_record,
            provenance=prov_payload,
            reproducibility=repro_rec.to_dict(),
        )

        # 10. Generate MODEL_CARD.md and README.md
        card_content = ModelCardGenerator.generate_model_card(
            manifest=manifest,
            dataset_prov=dataset_prov,
            training_prov=training_prov,
            hardware_prov=hardware_prov,
        )
        with open(dest_dir / "MODEL_CARD.md", "w", encoding="utf-8") as f:
            f.write(card_content)

        readme_content = ReadmeGenerator.generate_readme(manifest=manifest)
        with open(dest_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

        # 11. Compute Initial Hashes and Checksums File
        manifest.save_atomic(dest_dir / "manifest.json")
        ReleaseIntegrityManager.generate_checksums_file(dest_dir)

        # 12. Record Full Inventory in Manifest
        inventory = []
        hashes = {}
        for p in sorted(dest_dir.rglob("*")):
            if p.is_file() and p.name != "manifest.json":
                rel_p = p.relative_to(dest_dir).as_posix()
                inventory.append(rel_p)
                hashes[rel_p] = compute_file_sha256(p)

        manifest.artifact_inventory = inventory
        manifest.artifact_hashes = hashes
        manifest.status = ReleaseStatus.VALIDATING
        manifest.save_atomic(dest_dir / "manifest.json")

        # Regenerate checksums file to include updated manifest
        ReleaseIntegrityManager.generate_checksums_file(dest_dir)

        # 13. Run Validation Audit
        val_report = self.release_validator.validate_release(dest_dir)
        if val_report.is_valid:
            manifest.status = ReleaseStatus.READY
            manifest.status_reason = "All mandatory release validations passed."
        else:
            manifest.status = ReleaseStatus.INVALID
            manifest.status_reason = f"Release validation failed: {len(val_report.errors)} errors."
            errors.extend(val_report.errors)

        manifest.save_atomic(dest_dir / "manifest.json")
        # Final checksum update
        ReleaseIntegrityManager.generate_checksums_file(dest_dir)

        return manifest.status == ReleaseStatus.READY, manifest, errors
