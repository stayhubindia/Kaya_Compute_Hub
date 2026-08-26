"""
Metadata Classifier and Taxonomy Mapper (Phase 3.3).
Maps scientific documents, NPTEL courses, and arXiv preprints into the 13 authoritative
domain taxonomies, assigning detailed subtopics (e.g. domain='science', topic='physics', subtopic='quantum_mechanics').
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class MetadataClassifier:
    """Classifies document domain, topic, and subtopics based on text, title, and metadata."""

    # 13 Authoritative Domains
    TAXONOMY = [
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

    # arXiv Primary Category Mappings
    ARXIV_CATEGORY_MAP: Dict[str, Tuple[str, str, str]] = {
        # astro-ph -> Science / Physics / Astrophysics
        "astro-ph": ("science", "physics", "astrophysics"),
        "astro-ph.co": ("science", "physics", "cosmology"),
        "astro-ph.ga": ("science", "physics", "astrophysics"),
        "astro-ph.he": ("science", "physics", "high_energy_astrophysics"),
        "astro-ph.sr": ("science", "physics", "solar_stellar_astrophysics"),
        # Physics categories
        "quant-ph": ("science", "physics", "quantum_mechanics"),
        "hep-th": ("science", "physics", "high_energy_physics_theory"),
        "hep-ph": ("science", "physics", "particle_physics"),
        "hep-ex": ("science", "physics", "experimental_particle_physics"),
        "gr-qc": ("science", "physics", "general_relativity_gravitation"),
        "cond-mat": ("science", "physics", "condensed_matter"),
        "nucl-th": ("science", "physics", "nuclear_physics"),
        "physics.optics": ("science", "physics", "optics"),
        "physics.flu-dyn": ("science", "physics", "fluid_dynamics"),
        "physics.chem-ph": ("science", "physics", "chemical_physics"),
        "physics.geo-ph": ("science", "physics", "geophysics"),
        # AI / ML
        "cs.AI": ("ai_ml", "ml_fundamentals", "artificial_intelligence"),
        "cs.LG": ("ai_ml", "deep_learning", "machine_learning"),
        "cs.CV": ("ai_ml", "computer_vision", "vision_models"),
        "cs.CL": ("ai_ml", "nlp", "computational_linguistics"),
        "stat.ML": ("ai_ml", "ml_fundamentals", "statistical_learning"),
        # Mathematics
        "math.PR": ("mathematics", "probability_statistics", "probability"),
        "math.ST": ("mathematics", "probability_statistics", "statistics"),
        "math.CA": ("mathematics", "calculus", "classical_analysis"),
        "math.DG": ("mathematics", "geometry_trigonometry", "differential_geometry"),
        # Computer Science / Programming / Systems
        "cs.CR": ("cybersecurity", "cryptography", "security"),
        "cs.SE": ("software_engineering", "system_design", "software_engineering"),
        "cs.NI": ("networking", "osi_tcpip", "network_protocols"),
        "cs.OS": ("linux_systems", "kernel_internals", "operating_systems"),
    }

    # Keyword heuristics for Physics & Science subtopics
    PHYSICS_SUBTOPIC_KEYWORDS = [
        ("quantum_mechanics", [r"\bquantum\b", r"\bwavefunction\b", r"\bschrod?inger\b", r"\bhamiltonian\b", r"\bqubit\b"]),
        ("thermodynamics", [r"\bthermodynamics?\b", r"\bentropy\b", r"\benthalpy\b", r"\bgibbs\b", r"\bcarnot\b", r"\bheat\s+engine\b"]),
        ("astrophysics", [r"\bastrophysics?\b", r"\bgalaxy\b", r"\bgalaxies\b", r"\bstar\b", r"\bblack\s+hole\b", r"\bneutron\s+star\b", r"\bexoplanet\b", r"\bcosmic\b"]),
        ("electromagnetism", [r"\belectromagnetism\b", r"\bmaxwell\b", r"\belectric\s+field\b", r"\bmagnetic\s+field\b", r"\bpoynting\b"]),
        ("classical_mechanics", [r"\bclassical\s+mechanics\b", r"\blagrangian\b", r"\bangular\s+momentum\b", r"\bnewtonian\b", r"\bpendulum\b"]),
        ("relativity", [r"\bgeneral\s+relativity\b", r"\bspecial\s+relativity\b", r"\blorentz\b", r"\bspacetime\b", r"\bmetric\s+tensor\b"]),
        ("optics", [r"\boptics\b", r"\bdiffraction\b", r"\brefraction\b", r"\binterferometer\b", r"\blaser\b"]),
        ("geophysics", [r"\bgeophysics\b", r"\bmantle\b", r"\bseismic\b", r"\blithosphere\b", r"\bmagma\b"]),
    ]

    def classify(
        self,
        text: str,
        title: Optional[str] = None,
        categories: Optional[List[str]] = None,
        source: str = "unknown",
    ) -> Tuple[str, str, Optional[str], float]:
        """
        Returns (domain, topic, subtopic, confidence_score).
        """
        # 1. Match from explicit arXiv category if available
        if categories:
            for cat in categories:
                cat_clean = cat.strip().lower()
                for known_cat, mapping in self.ARXIV_CATEGORY_MAP.items():
                    if cat_clean.startswith(known_cat.lower()):
                        domain, topic, subtopic = mapping
                        return domain, topic, subtopic, 0.95

        # 2. Match from title and content
        search_corpus = f"{title or ''} {text[:2000]}".lower()

        # Check Physics Subtopics under domain="science"
        for subtopic_name, patterns in self.PHYSICS_SUBTOPIC_KEYWORDS:
            for pat in patterns:
                if re.search(pat, search_corpus, re.IGNORECASE):
                    return "science", "physics", subtopic_name, 0.85

        # Check other engineering/computing keywords
        if re.search(r"\b(neural\s+network|transformer|attention|deep\s+learning|llm)\b", search_corpus):
            return "ai_ml", "deep_learning", "neural_networks", 0.85

        if re.search(r"\b(linux|kernel|systemd|posix|bash)\b", search_corpus):
            return "linux_systems", "kernel_internals", "os_engineering", 0.85

        if re.search(r"\b(cipher|cryptography|vulnerability|exploit|rsa)\b", search_corpus):
            return "cybersecurity", "cryptography", "security", 0.85

        # Default fallback for physics corpus
        return "science", "physics", "general_physics", 0.70
