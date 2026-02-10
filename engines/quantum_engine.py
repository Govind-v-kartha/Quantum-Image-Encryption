"""
Layer 4: Quantum Engine
Encrypts/decrypts ROI blocks using NEQR quantum encoding from Repo B
(repos/quantum_repo/encryption_pipeline.py).

Each 8x8 RGB block is passed through the full quantum-chaos-DNA pipeline:
  NEQR encode -> quantum scramble -> quantum permutation -> DNA encode -> XOR diffusion
"""

import numpy as np
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Ensure quantum_repo is importable
_REPO_ROOT = str(Path(__file__).parent.parent / 'repos' / 'quantum_repo')
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class QuantumEngine:
    """Encrypt/decrypt ROI blocks via Repo-B NEQR quantum pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('quantum_engine', {})
        self.logger = logging.getLogger('quantum_engine')
        self.is_initialized = False
        self.shots = self.config.get('shots', 65536)

    def initialize(self):
        """Import the Repo-B pipeline (deferred to avoid import-time crashes)."""
        try:
            from encryption_pipeline import encrypt_image, decrypt_image
            self._encrypt_fn = encrypt_image
            self._decrypt_fn = decrypt_image
            self.is_initialized = True
            self.logger.info(
                f"  Quantum Engine initialized  (NEQR pipeline, shots={self.shots})"
            )
        except ImportError as e:
            self.logger.error(f"  Failed to import quantum_repo pipeline: {e}")
            raise RuntimeError(
                "Cannot import repos.quantum_repo.encryption_pipeline. "
                "Ensure the quantum_repo submodule is present."
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encrypt_blocks(
        self,
        roi_blocks: List[np.ndarray],
        master_seed: int,
        *,
        progress_callback=None,
    ) -> List[np.ndarray]:
        """
        Encrypt a list of 8x8 RGB blocks using Repo-B NEQR pipeline.

        Parameters
        ----------
        roi_blocks      : list of (8,8,3) uint8
        master_seed     : int  – shared secret for key derivation
        progress_callback : callable(current, total)  optional

        Returns
        -------
        encrypted_blocks : list of (8,8,3) uint8
        """
        encrypted: List[np.ndarray] = []
        total = len(roi_blocks)

        for idx, block in enumerate(roi_blocks):
            assert block.shape == (8, 8, 3) and block.dtype == np.uint8, \
                f"Block {idx} shape/dtype mismatch: {block.shape}, {block.dtype}"

            # Per-block seed derived from master + block index for uniqueness
            block_seed = (master_seed + idx * 7919) % (2**31)

            enc_block = self._encrypt_fn(
                block,
                master_seed=block_seed,
                shots=self.shots,
                use_quantum_encoding=True,
                pad_mode='edge',
            )
            encrypted.append(enc_block.astype(np.uint8))

            if progress_callback and (idx + 1) % max(1, total // 20) == 0:
                progress_callback(idx + 1, total)

        self.logger.info(f"  Encrypted {total} ROI blocks (NEQR)")
        return encrypted

    def decrypt_blocks(
        self,
        encrypted_blocks: List[np.ndarray],
        master_seed: int,
        *,
        progress_callback=None,
    ) -> List[np.ndarray]:
        """
        Decrypt a list of 8x8 RGB blocks using Repo-B NEQR pipeline.

        Parameters
        ----------
        encrypted_blocks : list of (8,8,3) uint8
        master_seed      : int
        progress_callback : callable(current, total)

        Returns
        -------
        decrypted_blocks : list of (8,8,3) uint8
        """
        decrypted: List[np.ndarray] = []
        total = len(encrypted_blocks)

        for idx, block in enumerate(encrypted_blocks):
            block_seed = (master_seed + idx * 7919) % (2**31)

            dec_block = self._decrypt_fn(
                block.astype(np.uint8),
                master_seed=block_seed,
                shots=self.shots,
                use_quantum_encoding=True,
            )
            decrypted.append(dec_block.astype(np.uint8))

            if progress_callback and (idx + 1) % max(1, total // 20) == 0:
                progress_callback(idx + 1, total)

        self.logger.info(f"  Decrypted {total} ROI blocks (NEQR)")
        return decrypted
