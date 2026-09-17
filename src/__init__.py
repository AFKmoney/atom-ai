"""ATOM (atom-ai) — atom-native continuous structured learning.

Keep imports as ``from src...`` for the flat src layout.
"""

__version__ = "0.2.0"

try:
    from src.toroidal.model import ToroidalFractalIntelligence
    from src.atom_native import AtomNativeModel
    from src.io.atomizer import Atomizer, AtomPacket
except ImportError:  # pragma: no cover
    ToroidalFractalIntelligence = None  # type: ignore
    AtomNativeModel = None  # type: ignore
    Atomizer = None  # type: ignore
    AtomPacket = None  # type: ignore

__all__ = [
    "ToroidalFractalIntelligence",
    "AtomNativeModel",
    "Atomizer",
    "AtomPacket",
]
