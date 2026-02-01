#!/usr/bin/env python3
"""
Quick test of quantum encryption with a single block
"""

import numpy as np
from pathlib import Path
import sys

# Add repo paths
repo_quantum = Path(__file__).parent / "repos" / "Quantum-Image-Encryption"
sys.path.insert(0, str(repo_quantum))

# Test imports
print("[TEST] Importing quantum modules...")
try:
    from quantum.neqr import encode_neqr, reconstruct_neqr_image
    from quantum.scrambling import quantum_scramble
    print("  [OK] NEQR and scrambling imported")
except Exception as e:
    print(f"  [ERROR] Import failed: {e}")
    sys.exit(1)

# Create a test block
print("\n[TEST] Creating 8x8 test block...")
test_block = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
print(f"  Block shape: {test_block.shape}")
print(f"  Block min/max: {test_block.min()}/{test_block.max()}")

# Test NEQR encoding
print("\n[TEST] NEQR Encoding...")
try:
    qc = encode_neqr(test_block)
    print(f"  [OK] Quantum circuit created with {qc.num_qubits} qubits")
except Exception as e:
    print(f"  [ERROR] NEQR encoding failed: {e}")
    sys.exit(1)

# Test quantum scrambling
print("\n[TEST] Quantum Scrambling...")
try:
    num_position_qubits = 6  # log2(8)*2 = 6 for 8x8
    seed = 12345
    key = np.random.RandomState(seed).randint(0, 256, num_position_qubits, dtype=np.uint8)
    qc_scrambled = quantum_scramble(qc, key, num_position_qubits)
    print(f"  [OK] Scrambling applied with {len(key)} key bytes")
except Exception as e:
    print(f"  [ERROR] Scrambling failed: {e}")
    sys.exit(1)

# Test reconstruction (this is SLOW with Qiskit simulator!)
print("\n[TEST] Reconstruction (WARNING: This may take 10-30 seconds)...")
try:
    import time
    t0 = time.time()
    encrypted_block = reconstruct_neqr_image(qc_scrambled, 8, 8)
    elapsed = time.time() - t0
    print(f"  [OK] Block reconstructed in {elapsed:.2f}s")
    print(f"  Encrypted block shape: {encrypted_block.shape}")
    print(f"  Encrypted block range: {encrypted_block.min()}-{encrypted_block.max()}")
except Exception as e:
    print(f"  [ERROR] Reconstruction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All quantum encryption tests passed!")
print("NOTE: Full encryption of 13,163 blocks would take ~30-60 minutes")
