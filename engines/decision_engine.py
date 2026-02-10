"""
Layer 3: Decision Engine
Separates image into ROI and Background using the ROI mask.
Divides ROI into 8x8 blocks, pads under-sized blocks, records positions.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple


class DecisionEngine:
    """Split image into ROI 8x8 blocks and BG region using a binary mask."""

    BLOCK_SIZE = 8  # fixed 8x8

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('decision_engine', {})
        self.logger = logging.getLogger('decision_engine')
        self.is_initialized = False

    def initialize(self):
        self.is_initialized = True
        self.logger.info("Decision Engine initialized (block_size=8x8)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def separate_roi_bg(
        self,
        image: np.ndarray,
        roi_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Separate image into ROI 8x8 blocks and background.

        Parameters
        ----------
        image    : (H, W, 3) uint8 RGB
        roi_mask : (H, W) uint8, {0, 255}

        Returns
        -------
        dict with keys:
            roi_blocks   : list[np.ndarray]  – each (8,8,3) uint8
            block_map    : list[dict]         – position/padding info per block
            bg_image     : np.ndarray         – (H,W,3) uint8, ROI pixels zeroed
            roi_mask     : np.ndarray         – original mask (kept for metadata)
            image_shape  : (H, W, 3)
        """
        h, w, c = image.shape
        B = self.BLOCK_SIZE

        # 1. Identify which 8x8 grid cells contain ROI
        roi_blocks: List[np.ndarray] = []
        block_map: List[Dict[str, Any]] = []

        # Walk the grid
        for row in range(0, h, B):
            for col in range(0, w, B):
                # Actual region (may be < 8 at edges)
                r_end = min(row + B, h)
                c_end = min(col + B, w)

                block_mask = roi_mask[row:r_end, col:c_end]

                # A block is ROI if ANY pixel in it is ROI
                if np.any(block_mask > 0):
                    block_data = image[row:r_end, col:c_end].copy()  # (bh, bw, 3)
                    bh, bw = block_data.shape[:2]

                    # Pad to 8x8 if needed
                    padded = False
                    if bh < B or bw < B:
                        padded_block = np.zeros((B, B, c), dtype=np.uint8)
                        padded_block[:bh, :bw, :] = block_data
                        block_data = padded_block
                        padded = True

                    roi_blocks.append(block_data)  # always (8,8,3)
                    block_map.append({
                        'index': len(block_map),
                        'row': int(row),
                        'col': int(col),
                        'actual_h': int(bh),
                        'actual_w': int(bw),
                        'padded': padded,
                    })

        # 2. Build background image – ROI pixel positions zeroed
        bg_image = image.copy()
        # Zero out every ROI block position
        for bm in block_map:
            r, c_ = bm['row'], bm['col']
            ah, aw = bm['actual_h'], bm['actual_w']
            bg_image[r:r+ah, c_:c_+aw] = 0

        n_roi = len(roi_blocks)
        n_total = ((h + B - 1) // B) * ((w + B - 1) // B)
        self.logger.info(
            f"  ROI blocks: {n_roi}/{n_total}  "
            f"({n_roi / max(n_total, 1) * 100:.1f}%)"
        )
        self.logger.info(
            f"  BG pixels: {np.count_nonzero(bg_image.sum(axis=2) > 0)} non-zero"
        )

        return {
            'roi_blocks': roi_blocks,
            'block_map': block_map,
            'bg_image': bg_image,
            'roi_mask': roi_mask,
            'image_shape': image.shape,
        }

    # ------------------------------------------------------------------
    # Reconstruction (for decryption)
    # ------------------------------------------------------------------

    def reconstruct_image(
        self,
        decrypted_roi_blocks: List[np.ndarray],
        decrypted_bg_image: np.ndarray,
        block_map: List[Dict[str, Any]],
        image_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Merge decrypted ROI blocks and decrypted BG into a full image.

        Parameters
        ----------
        decrypted_roi_blocks : list[(8,8,3) uint8]
        decrypted_bg_image   : (H,W,3) uint8
        block_map            : list[dict]
        image_shape          : (H, W, 3)

        Returns
        -------
        image : (H, W, 3) uint8
        """
        h, w, c = image_shape
        result = decrypted_bg_image.copy()

        for i, bm in enumerate(block_map):
            r, col = bm['row'], bm['col']
            ah, aw = bm['actual_h'], bm['actual_w']
            block = decrypted_roi_blocks[i]
            result[r:r+ah, col:col+aw] = block[:ah, :aw]

        self.logger.info(f"  Reconstructed image: shape={result.shape}")
        return result

