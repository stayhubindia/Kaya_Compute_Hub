"""
Master Release Validation Engine (Phase 5.1).
Audits release directory bundles against manifest, adapter configs,
base model compatibility, provenance, integrity checksums, and documentation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from src.release.adapter import AdapterValidator
from src.release.compatibility import BaseModelCompatibilityValidator
from src.release.integrity import ReleaseIntegrityManager
from src.release.manifest import ReleaseManifest, ReleaseStatus

logger = logging.getLogger(__name__)


class ReleaseValidationReport(BaseModel):
    """Structured report evaluating complete release validity."""
    release_id: str
    release_dir: str
    is_valid: bool = False
    status: str = "INVALID"
    checks_passed: Dict[str, bool] = Field(default_factory=dict)
    manifest_valid: bool = False
    adapter_valid: bool = False
    compatibility_valid: bool = False
    provenance_valid: bool = False
    integrity_valid: bool = False
    reproducibility_valid: bool = False
    model_card_valid: bool = False
    readme_valid: bool = False
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ReleaseValidator:
    """Performs full end-to-end audit of a QLoRA release package."""

    def __init__(
        self,
        adapter_validator: Optional[AdapterValidator] = None,
        compatibility_validator: Optional[BaseModelCompatibilityValidator] = None,
    ):
        self.adapter_validator = adapter_validator or AdapterValidator()
        self.compatibility_validator = compatibility_validator or BaseModelCompatibilityValidator()

    def validate_release(self, release_dir: Union[str, Path]) -> ReleaseValidationReport:
        """Run comprehensive 8-stage audit on target release directory."""
        r_dir = Path(release_dir)
        report = ReleaseValidationReport(
            release_id=r_dir.name,
            release_dir=str(r_dir),
        )

        if not r_dir.exists() or not r_dir.is_dir():
            report.errors.append(f"Release directory not found: {r_dir}")
            return report

        # 1. Manifest Validation
        man_path = r_dir / "manifest.json"
        manifest: Optional[ReleaseManifest] = None
        if not man_path.exists():
            report.errors.append("Missing manifest.json")
            report.checks_passed["manifest"] = False
        else:
            try:
                manifest = ReleaseManifest.load(man_path)
                report.manifest_valid = True
                report.checks_passed["manifest"] = True
            except Exception as e:
                report.errors.append(f"Failed to load or parse manifest.json: {e}")
                report.checks_passed["manifest"] = False

        # 2. Adapter Artifact Validation
        adapter_dir = r_dir / "adapter"
        if not adapter_dir.exists():
            report.errors.append("Missing adapter/ directory")
            report.checks_passed["adapter"] = False
        else:
            adapt_res = self.adapter_validator.validate_directory(adapter_dir)
            report.adapter_valid = adapt_res.is_valid
            report.checks_passed["adapter"] = adapt_res.is_valid
            if not adapt_res.is_valid:
                report.errors.extend([f"Adapter error: {e}" for e in adapt_res.errors])
            if adapt_res.warnings:
                report.warnings.extend([f"Adapter warning: {w}" for w in adapt_res.warnings])

        # 3. Compatibility Validation
        comp_path = r_dir / "compatibility.json"
        if not comp_path.exists():
            report.errors.append("Missing compatibility.json")
            report.checks_passed["compatibility"] = False
        else:
            try:
                with open(comp_path, "r", encoding="utf-8") as f:
                    comp_data = json.load(f)
                base_id = comp_data.get("target_base_model")
                comp_res = self.compatibility_validator.validate_base_model_metadata(base_model_id=base_id)
                report.compatibility_valid = comp_res.is_compatible
                report.checks_passed["compatibility"] = comp_res.is_compatible
                if not comp_res.is_compatible:
                    report.errors.extend(comp_res.errors)
            except Exception as e:
                report.errors.append(f"Failed to parse compatibility.json: {e}")
                report.checks_passed["compatibility"] = False

        # 4. Provenance Validation
        prov_path = r_dir / "provenance.json"
        if not prov_path.exists():
            report.errors.append("Missing provenance.json")
            report.checks_passed["provenance"] = False
        else:
            try:
                with open(prov_path, "r", encoding="utf-8") as f:
                    prov_data = json.load(f)
                d_prov = prov_data.get("dataset_provenance", {})
                t_prov = prov_data.get("training_provenance", {})
                if d_prov.get("provenance_status") == "VERIFIED" and t_prov.get("config_hash"):
                    report.provenance_valid = True
                    report.checks_passed["provenance"] = True
                else:
                    report.provenance_valid = False
                    report.checks_passed["provenance"] = False
                    report.errors.append("Provenance status not verified or missing training config hash")
            except Exception as e:
                report.errors.append(f"Failed to parse provenance.json: {e}")
                report.checks_passed["provenance"] = False

        # 5. Integrity & Checksums Validation
        chk_path = r_dir / "checksums.sha256"
        if not chk_path.exists():
            report.errors.append("Missing checksums.sha256")
            report.checks_passed["integrity"] = False
        else:
            integ_res = ReleaseIntegrityManager.verify_release_integrity(r_dir)
            report.integrity_valid = integ_res.is_valid
            report.checks_passed["integrity"] = integ_res.is_valid
            if not integ_res.is_valid:
                for mf in integ_res.missing_files:
                    report.errors.append(f"Integrity missing file: {mf}")
                for mm in integ_res.mismatched_files:
                    report.errors.append(
                        f"Checksum mismatch for {mm['file']}: expected {mm['expected_sha256'][:8]}, got {mm['actual_sha256'][:8]}"
                    )
                for uf in integ_res.unexpected_files:
                    report.errors.append(f"Untracked / unexpected file found: {uf}")

        # 6. Reproducibility Record Validation
        rep_path = r_dir / "reproducibility.json"
        if not rep_path.exists():
            report.errors.append("Missing reproducibility.json")
            report.checks_passed["reproducibility"] = False
        else:
            try:
                with open(rep_path, "r", encoding="utf-8") as f:
                    rep_data = json.load(f)
                if rep_data.get("training_config_hash") and rep_data.get("random_seed") is not None:
                    report.reproducibility_valid = True
                    report.checks_passed["reproducibility"] = True
                else:
                    report.checks_passed["reproducibility"] = False
                    report.errors.append("reproducibility.json missing seed or config hash")
            except Exception as e:
                report.errors.append(f"Failed to parse reproducibility.json: {e}")
                report.checks_passed["reproducibility"] = False

        # 7. Model Card & README Validation
        card_path = r_dir / "MODEL_CARD.md"
        if not card_path.exists():
            report.errors.append("Missing MODEL_CARD.md")
            report.checks_passed["model_card"] = False
        else:
            with open(card_path, "r", encoding="utf-8") as f:
                card_content = f.read()
            # Check mandatory sections
            required_sections = [
                "1. Model Summary", "2. Base Model", "3. Adapter Type",
                "4. Intended Use", "5. Training Method", "6. Dataset",
                "7. Dataset Provenance", "8. Training Configuration",
                "9. Quantization", "10. Hardware", "11. Evaluation",
                "12. Limitations", "13. Reproducibility", "14. Version",
                "15. Integrity", "16. Change Log",
            ]
            missing_sec = [sec for sec in required_sections if sec not in card_content]
            if missing_sec:
                report.errors.append(f"MODEL_CARD.md missing required sections: {missing_sec}")
                report.checks_passed["model_card"] = False
            else:
                report.model_card_valid = True
                report.checks_passed["model_card"] = True

        readme_path = r_dir / "README.md"
        if not readme_path.exists():
            report.errors.append("Missing README.md")
            report.checks_passed["readme"] = False
        else:
            report.readme_valid = True
            report.checks_passed["readme"] = True

        # Determine overall validity
        all_passed = (
            report.manifest_valid
            and report.adapter_valid
            and report.compatibility_valid
            and report.provenance_valid
            and report.integrity_valid
            and report.reproducibility_valid
            and report.model_card_valid
            and report.readme_valid
            and len(report.errors) == 0
        )
        report.is_valid = all_passed
        report.status = "READY" if all_passed else "INVALID"

        return report
