"""
Main Entry Point - Hybrid Quantum-Classical Image Encryption System

This integrates:
- FlexiMo: Semantic segmentation and ROI detection
- Quantum-Image-Encryption: NEQR quantum encoding + AES-256-GCM
"""

import sys
from pathlib import Path

# ===== EXPOSE CLONED REPOS TO PYTHON =====
# Add repos folder to Python path so engines can import from them
repos_path = Path(__file__).parent / "repos"
if str(repos_path) not in sys.path:
    sys.path.insert(0, str(repos_path))

# Import repos to make them available as packages
print("=" * 80)
print("LOADING REPOSITORY INTEGRATIONS...")
print("=" * 80)

try:
    import quantum_repo
    print("✓ Quantum Image Encryption repository loaded")
except ImportError as e:
    print(f"⚠ Warning: Could not import quantum_repo: {e}")

try:
    import fleximo_repo
    print("✓ FlexiMo repository loaded")
except ImportError as e:
    print(f"⚠ Warning: Could not import fleximo_repo: {e}")

print("=" * 80)

# ===== IMPORT MAIN WORKFLOWS =====
from workflows.encrypt import main as encrypt_main

if __name__ == "__main__":
    sys.exit(encrypt_main())
