"""
License and Rights Management (Phase 3.3).
Detects and audits license statements across NPTEL, arXiv, and scientific documents.
Enforces internal_only gating on unverified or restricted content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from src.ingestion.models import LicenseStatus


@dataclass
class LicenseEvaluationResult:
    license_name: Optional[str]
    license_status: LicenseStatus
    license_url: Optional[str]
    license_evidence: Optional[str]
    internal_only: bool
    rights_notes: str


class LicenseHandler:
    """Evaluates document text and metadata to classify licensing terms and rights restrictions."""

    # Explicit permissive licenses permitted for external training/redistribution
    PERMISSIVE_LICENSES = {
        "CC-BY-4.0",
        "CC-BY-3.0",
        "CC-BY",
        "CC0-1.0",
        "CC0",
        "MIT",
        "Apache-2.0",
        "BSD-3-Clause",
        "BSD-2-Clause",
    }

    # License detection regex rules
    PATTERNS = [
        (r"(?i)creative\s+commons\s+attribution\s+4\.0\s+international", "CC-BY-4.0", "http://creativecommons.org/licenses/by/4.0/"),
        (r"(?i)creative\s+commons\s+attribution-noncommercial-sharealike", "CC-BY-NC-SA-4.0", "http://creativecommons.org/licenses/by-nc-sa/4.0/"),
        (r"(?i)creative\s+commons\s+attribution-sharealike", "CC-BY-SA-4.0", "http://creativecommons.org/licenses/by-sa/4.0/"),
        (r"(?i)creative\s+commons\s+attribution-noncommercial", "CC-BY-NC-4.0", "http://creativecommons.org/licenses/by-nc/4.0/"),
        (r"(?i)cc0\s+1\.0\s+universal|public\s+domain\s+dedication", "CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"),
        (r"(?i)arxiv\.org\s+non-exclusive\s+license\s+to\s+distribute", "arXiv Non-exclusive License", "https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html"),
        (r"(?i)mit\s+license", "MIT", "https://opensource.org/licenses/MIT"),
        (r"(?i)apache\s+license,?\s+version\s+2\.0", "Apache-2.0", "https://www.apache.org/licenses/LICENSE-2.0"),
        (r"(?i)nptel\s+open\s+access|nptel.*?license", "NPTEL Open Access", "https://nptel.ac.in"),
    ]

    def evaluate_license(
        self,
        text: str,
        declared_license: Optional[str] = None,
        source: str = "unknown",
    ) -> LicenseEvaluationResult:
        """
        Determines the license and internal_only safety gating for a document.
        """
        # 1. Use declared license if valid
        if declared_license and declared_license.strip() and declared_license.upper() != "UNKNOWN":
            dec = declared_license.strip()
            is_internal = dec not in self.PERMISSIVE_LICENSES
            return LicenseEvaluationResult(
                license_name=dec,
                license_status=LicenseStatus.KNOWN,
                license_url=None,
                license_evidence=f"Explicitly declared by source '{source}'",
                internal_only=is_internal,
                rights_notes="License provided in source metadata.",
            )

        # 2. Heuristic text search for license statements
        sample_text = text[:4000] + "\n" + text[-2000:] if len(text) > 4000 else text
        for pat, lic_name, lic_url in self.PATTERNS:
            match = re.search(pat, sample_text)
            if match:
                is_internal = lic_name not in self.PERMISSIVE_LICENSES
                return LicenseEvaluationResult(
                    license_name=lic_name,
                    license_status=LicenseStatus.KNOWN,
                    license_url=lic_url,
                    license_evidence=match.group(0),
                    internal_only=is_internal,
                    rights_notes=f"Detected license '{lic_name}' in document text.",
                )

        # 3. Source-specific defaults if known
        if "nptel" in source.lower():
            return LicenseEvaluationResult(
                license_name="CC-BY-NC-SA-4.0",
                license_status=LicenseStatus.KNOWN,
                license_url="http://creativecommons.org/licenses/by-nc-sa/4.0/",
                license_evidence="NPTEL standard educational license policy",
                internal_only=True,
                rights_notes="NPTEL material is non-commercial/share-alike; gated internal_only.",
            )
        elif "arxiv" in source.lower():
            return LicenseEvaluationResult(
                license_name="arXiv Non-exclusive License",
                license_status=LicenseStatus.KNOWN,
                license_url="https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html",
                license_evidence="arXiv standard distribution agreement",
                internal_only=True,
                rights_notes="arXiv preprints default to non-exclusive distribution; gated internal_only.",
            )

        # 4. Unknown / unverified license -> Gated internal_only: true
        return LicenseEvaluationResult(
            license_name="UNKNOWN",
            license_status=LicenseStatus.UNKNOWN,
            license_url=None,
            license_evidence=None,
            internal_only=True,
            rights_notes="License terms unestablished. Content strictly gated as internal_only.",
        )
