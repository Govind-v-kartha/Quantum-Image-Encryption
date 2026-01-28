"""
Bridge Controller Package
Integrates FlexiMo (AI Segmentation) with Quantum-Image-Encryption

Main Components:
- splitter: Image splitting logic (ROI vs Background)
- quantum_handler: Quantum encryption (NEQR + Arnold Scrambling)
- classical_handler: Classical encryption (Chaos Maps)
- pipeline: Main Bridge Controller orchestration
"""

from .splitter import ImageSplitter, load_image, load_mask
from .quantum_handler import QuantumEncryptionHandler, QuantumKeyManager
from .classical_handler import ClassicalEncryptionHandler
from .pipeline import BridgeController

__version__ = "1.0.0"
__author__ = "Secure Image Encryption Project"

__all__ = [
    "BridgeController",
    "ImageSplitter",
    "QuantumEncryptionHandler",
    "QuantumKeyManager",
    "ClassicalEncryptionHandler",
    "load_image",
    "load_mask",
]
