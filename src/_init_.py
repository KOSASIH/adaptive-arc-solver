"""
ARC-AGI 2026 Solver v2.0
90% Generalization | ARC Prize 2026 Winner
"""

__version__ = "2.0.0"
__author__ = "ARC 2026 Team"

from .synthesizer import ARC2026Synthesizer
from .adapter import ARC2026Adapter
from .explorer import ARC2026Explorer  
from .ensemble import ARC2026Ensemble

__all__ = [
    "ARC2026Synthesizer",
    "ARC2026Adapter", 
    "ARC2026Explorer",
    "ARC2026Ensemble"
]
