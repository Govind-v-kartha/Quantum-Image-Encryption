"""
Layer 6: Fusion Engine
Merges encrypted ROI blocks and encrypted BG into a final output image.
Also handles the reverse operation for decryption.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple


class FusionEngine:
    """Place encrypted ROI blocks at their original positions over encrypted BG."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('fusion_engine', {})
        self.logger = logging.getLogger('fusion_engine')
        self.is_initialized = False

    def initialize(self):
        self.is_initialized = True
        self.logger.info("  Fusion Engine initialized")

    # ------------------------------------------------------------------
    # Encryption path
    # ------------------------------------------------------------------

    def fuse_encrypted(
        self,
        encrypted_roi_blocks: List[np.ndarray],
        encrypted_bg_visual: np.ndarray,
        block_map: List[Dict[str, Any]],
        image_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Build the final encrypted image by placing encrypted ROI blocks
        over a visual representation of the encrypted BG.

        Parameters
        ----------
        encrypted_roi_blocks : list of (8,8,3) uint8  – quantum-encrypted
        encrypted_bg_visual  : (H,W,3) uint8          – noise-like visual for BG
        block_map            : position metadata
        image_shape          : original (H, W, 3)

        Returns
        -------
        fused : (H, W, 3) uint8
        """
        h, w, c = image_shape
        fused = encrypted_bg_visual.copy()

        for i, bm in enumerate(block_map):
            r, col = bm['row'], bm['col']
            ah, aw = bm['actual_h'], bm['actual_w']
            block = encrypted_roi_blocks[i]
            fused[r:r+ah, col:col+aw] = block[:ah, :aw]

        self.logger.info(
            f"  Fused encrypted image: {len(encrypted_roi_blocks)} ROI blocks "
            f"+ BG -> {fused.shape}"
        )
        return fused

    def create_bg_visual(
        self,
        bg_image: np.ndarray,
        master_seed: int,
    ) -> np.ndarray:
        """
        Create a visual noise representation for the BG region.
        The actual ciphertext is stored in metadata; this is only for the
        output image so no original content is visible.

        Parameters
        ----------
        bg_image    : (H,W,3) uint8 – original BG (ROI zeroed)
        master_seed : int

        Returns
        -------
        bg_noise : (H,W,3) uint8
        """
        rng = np.random.RandomState(master_seed & 0xFFFFFFFF)
        noise = rng.randint(0, 256, size=bg_image.shape, dtype=np.uint8)

        # Keep ROI-block positions at zero (they'll be overwritten during fuse)
        # Identify BG pixels (originally non-zero) and non-BG pixels
        bg_mask = bg_image.sum(axis=2) > 0
        result = np.zeros_like(bg_image)
        result[bg_mask] = noise[bg_mask]

        return result

    # ------------------------------------------------------------------
    # Decryption path
    # ------------------------------------------------------------------

    def split_encrypted(
        self,
        encrypted_image: np.ndarray,
        block_map: List[Dict[str, Any]],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Extract encrypted ROI blocks and BG region from fused encrypted image.
        (Used for visual reference only; actual BG decryption uses stored ciphertext.)

        Parameters
        ----------
        encrypted_image : (H, W, 3) uint8
        block_map       : block position metadata

        Returns
        -------
        roi_blocks : list of (8,8,3) uint8
        bg_image   : (H,W,3) uint8 with ROI positions zeroed
        """
        B = 8
        h, w, c = encrypted_image.shape
        bg_image = encrypted_image.copy()
        roi_blocks: List[np.ndarray] = []

        for bm in block_map:
            r, col = bm['row'], bm['col']
            ah, aw = bm['actual_h'], bm['actual_w']

            block = np.zeros((B, B, c), dtype=np.uint8)
            block[:ah, :aw] = encrypted_image[r:r+ah, col:col+aw]
            roi_blocks.append(block)

            bg_image[r:r+ah, col:col+aw] = 0

        return roi_blocks, bg_image
