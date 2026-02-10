"""
Main Entry Point - 6-Layer Hybrid Quantum-Classical Image Encryption

Integrates:
  - FlexiMo (Repo A): Semantic segmentation for ROI detection
  - Quantum-Image-Encryption (Repo B): NEQR quantum encoding for ROI
  - AES-256-GCM: Classical encryption for background
"""

import sys
from pathlib import Path

# Expose cloned repos to Python path
repos_path = Path(__file__).parent / "repos"
quantum_repo_path = repos_path / "quantum_repo"

for p in (str(repos_path), str(quantum_repo_path)):
    if p not in sys.path:
        sys.path.insert(0, p)

print("=" * 80)
print("6-LAYER HYBRID QUANTUM-CLASSICAL IMAGE ENCRYPTION SYSTEM")
print("=" * 80)
print("  Repo A: FlexiMo Vision Transformer (ROI detection)")
print("  Repo B: NEQR Quantum Encryption     (ROI blocks)")
print("  Repo C: AES-256-GCM                 (background)")
print("=" * 80)

from workflows.encrypt import main as encrypt_main

if __name__ == "__main__":
    sys.exit(encrypt_main())
