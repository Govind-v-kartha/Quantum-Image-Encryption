"""
Main Entry Point - Decryption Orchestrator

This is a simple entry point that imports and runs the decryption workflow.
The actual implementation is in workflows/decrypt.py
"""

import sys
from workflows.decrypt import main as decrypt_main

if __name__ == "__main__":
    sys.exit(decrypt_main())
