"""Pipeline Core Service Package for Kaya Compute Hub Dataset Factory & Training Orchestrator."""

import sys
from pathlib import Path

# Ensure services.pipeline_core is registered as 'src' if imported under src namespaces
_current_module = sys.modules[__name__]
if 'src' not in sys.modules:
    sys.modules['src'] = _current_module
