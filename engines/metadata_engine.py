"""
Metadata Engine
Saves and loads all metadata required for lossless decryption:
  - master_seed, shots, block_map, image_shape, roi_mask, AES params, etc.
"""

import json
import numpy as np
import logging
import base64
import zlib
from pathlib import Path
from typing import Dict, Any, Optional


class MetadataEngine:
    """Serialize / deserialize all encryption metadata."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('metadata_engine', {})
        self.logger = logging.getLogger('metadata_engine')
        self.is_initialized = False
        self.use_compression = self.config.get('compression', 'zlib') == 'zlib'

    def initialize(self):
        self.is_initialized = True
        self.logger.info("  Metadata Engine initialized")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bytes_to_str(b: bytes) -> str:
        return base64.b64encode(b).decode('ascii')

    @staticmethod
    def _str_to_bytes(s: str) -> bytes:
        return base64.b64decode(s.encode('ascii'))

    @staticmethod
    def _ndarray_to_str(arr: np.ndarray) -> str:
        raw = arr.tobytes()
        compressed = zlib.compress(raw)
        return base64.b64encode(compressed).decode('ascii')

    @staticmethod
    def _str_to_ndarray(s: str, dtype, shape) -> np.ndarray:
        compressed = base64.b64decode(s.encode('ascii'))
        raw = zlib.decompress(compressed)
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        metadata_path: str,
        *,
        master_seed: int,
        shots: int,
        image_shape: tuple,
        block_map: list,
        roi_mask: np.ndarray,
        bg_encrypted_bundle: Dict[str, Any],
        encrypted_roi_blocks: list,
    ) -> str:
        """
        Save all encryption metadata to a JSON file.

        Returns the path where metadata was saved.
        """
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        # Serialise ROI mask
        roi_mask_data = self._ndarray_to_str(roi_mask)

        # Serialise AES bundle
        aes_bundle = {
            'encrypted_data': self._bytes_to_str(bg_encrypted_bundle['encrypted_data']),
            'salt': self._bytes_to_str(bg_encrypted_bundle['salt']),
            'nonce': self._bytes_to_str(bg_encrypted_bundle['nonce']),
            'shape': list(bg_encrypted_bundle['shape']),
            'dtype': bg_encrypted_bundle['dtype'],
        }

        # Serialise encrypted ROI blocks (list of 8x8x3 uint8)
        roi_data = []
        for block in encrypted_roi_blocks:
            roi_data.append(self._ndarray_to_str(block))

        doc = {
            'version': '4.0',
            'master_seed': master_seed,
            'shots': shots,
            'image_shape': list(image_shape),
            'block_map': block_map,
            'roi_mask': roi_mask_data,
            'roi_mask_shape': list(roi_mask.shape),
            'roi_mask_dtype': str(roi_mask.dtype),
            'aes_bundle': aes_bundle,
            'roi_blocks_encrypted': roi_data,
        }

        with open(metadata_path, 'w') as f:
            json.dump(doc, f, indent=2)

        size_kb = Path(metadata_path).stat().st_size / 1024
        self.logger.info(f"  Metadata saved: {metadata_path} ({size_kb:.1f} KB)")
        return metadata_path

    def load(self, metadata_path: str) -> Dict[str, Any]:
        """
        Load encryption metadata from JSON.

        Returns dict with same keys as save() parameters.
        """
        with open(metadata_path, 'r') as f:
            doc = json.load(f)

        # Deserialise ROI mask
        roi_mask = self._str_to_ndarray(
            doc['roi_mask'],
            dtype=np.dtype(doc['roi_mask_dtype']),
            shape=tuple(doc['roi_mask_shape']),
        )

        # Deserialise AES bundle
        aes_raw = doc['aes_bundle']
        bg_encrypted_bundle = {
            'encrypted_data': self._str_to_bytes(aes_raw['encrypted_data']),
            'salt': self._str_to_bytes(aes_raw['salt']),
            'nonce': self._str_to_bytes(aes_raw['nonce']),
            'shape': tuple(aes_raw['shape']),
            'dtype': aes_raw['dtype'],
        }

        # Deserialise encrypted ROI blocks
        encrypted_roi_blocks = []
        for s in doc['roi_blocks_encrypted']:
            block = self._str_to_ndarray(s, dtype=np.uint8, shape=(8, 8, 3))
            encrypted_roi_blocks.append(block)

        self.logger.info(
            f"  Metadata loaded: {metadata_path}  "
            f"({len(encrypted_roi_blocks)} ROI blocks)"
        )

        return {
            'master_seed': doc['master_seed'],
            'shots': doc['shots'],
            'image_shape': tuple(doc['image_shape']),
            'block_map': doc['block_map'],
            'roi_mask': roi_mask,
            'bg_encrypted_bundle': bg_encrypted_bundle,
            'encrypted_roi_blocks': encrypted_roi_blocks,
        }
