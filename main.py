"""
Main Entry Point - Encryption Orchestrator

This is a simple entry point that imports and runs the encryption workflow.
The actual implementation is in workflows/encrypt.py
"""

import sys
from workflows.encrypt import main as encrypt_main

if __name__ == "__main__":
    sys.exit(encrypt_main())
