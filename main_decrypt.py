"""
Decryption Entry Point

Usage:
  python main_decrypt.py <encrypted_image> <metadata_json> [config] [output_dir]
"""

import sys
from pathlib import Path

# Expose cloned repos to Python path
repos_path = Path(__file__).parent / "repos"
quantum_repo_path = repos_path / "quantum_repo"

for p in (str(repos_path), str(quantum_repo_path)):
    if p not in sys.path:
        sys.path.insert(0, p)

from workflows.decrypt import main as decrypt_main

if __name__ == "__main__":
    sys.exit(decrypt_main())
