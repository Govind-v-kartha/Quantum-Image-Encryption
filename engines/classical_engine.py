"""
Layer 5: Classical Engine
AES-256-GCM encryption/decryption for background image data.
Uses PBKDF2 for key derivation with random salt.
"""

import numpy as np
import logging
import os
import hashlib
import struct
from typing import Dict, Any, Tuple

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class ClassicalEngine:
    """AES-256-GCM encryption/decryption for background image."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('classical_engine', {})
        self.logger = logging.getLogger('classical_engine')
        self.is_initialized = False
        self.key_size = self.config.get('key_size', 256) // 8  # bytes
        self.nonce_size = self.config.get('nonce_size', 96) // 8  # 12 bytes
        self.pbkdf2_iterations = self.config.get('pbkdf2_iterations', 100000)

    def initialize(self):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "cryptography package is required for AES-256-GCM. "
                "Install with: pip install cryptography"
            )
        self.is_initialized = True
        self.logger.info("  Classical Engine initialized (AES-256-GCM)")

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    def _derive_key(self, password: bytes, salt: bytes) -> bytes:
        """Derive a 256-bit AES key from a password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.key_size,
            salt=salt,
            iterations=self.pbkdf2_iterations,
        )
        return kdf.derive(password)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encrypt(
        self,
        bg_image: np.ndarray,
        master_seed: int,
    ) -> Dict[str, Any]:
        """
        Encrypt background image using AES-256-GCM.

        Parameters
        ----------
        bg_image    : (H, W, 3) uint8 – background image (ROI positions zeroed)
        master_seed : int

        Returns
        -------
        dict with keys:
            encrypted_data : bytes   – AES-256-GCM ciphertext
            salt           : bytes   – 16-byte salt
            nonce          : bytes   – 12-byte nonce
            tag            : bytes   – included in ciphertext by AESGCM
            shape          : (H, W, 3)
            dtype          : str
        """
        # Serialise the array
        plaintext = bg_image.tobytes()
        shape = bg_image.shape
        dtype_str = str(bg_image.dtype)

        # Key derivation
        salt = os.urandom(16)
        password = master_seed.to_bytes(8, 'big')
        key = self._derive_key(password, salt)

        # Encrypt
        nonce = os.urandom(self.nonce_size)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)  # includes 16-byte tag

        self.logger.info(
            f"  BG encrypted: {len(plaintext)} bytes -> "
            f"{len(ciphertext)} bytes (AES-256-GCM)"
        )

        return {
            'encrypted_data': ciphertext,
            'salt': salt,
            'nonce': nonce,
            'shape': shape,
            'dtype': dtype_str,
        }

    def decrypt(
        self,
        encrypted_bundle: Dict[str, Any],
        master_seed: int,
    ) -> np.ndarray:
        """
        Decrypt background image from AES-256-GCM bundle.

        Parameters
        ----------
        encrypted_bundle : dict from encrypt()
        master_seed      : int  (must match encryption)

        Returns
        -------
        bg_image : (H, W, 3) uint8
        """
        ciphertext = encrypted_bundle['encrypted_data']
        salt       = encrypted_bundle['salt']
        nonce      = encrypted_bundle['nonce']
        shape      = tuple(encrypted_bundle['shape'])
        dtype_str  = encrypted_bundle['dtype']

        # Re-derive key
        password = master_seed.to_bytes(8, 'big')
        key = self._derive_key(password, salt)

        # Decrypt
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        bg_image = np.frombuffer(plaintext, dtype=np.dtype(dtype_str)).reshape(shape)
        self.logger.info(f"  BG decrypted: shape={shape}")
        return bg_image.copy()  # writable copy
