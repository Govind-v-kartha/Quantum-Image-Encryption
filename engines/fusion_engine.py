"""
PHASE 7: Fusion Engine V2

Merges encrypted ROI and encrypted background with:
- Random overlay strategies
- Pixel scrambling
- Boundary blending
- Integrity watermarking

Independent module - Input → Process → Output
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger("fusion_engine")


class FusionEngine:
    """Merge encrypted blocks into final image."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fusion engine.
        
        Args:
            config: Configuration dict from config.json
        """
        self.config = config.get('fusion_engine', {})
        self.overlay_strategy = self.config.get('overlay_strategy', 'random')
        self.pixel_scrambling = self.config.get('pixel_scrambling', True)
        self.boundary_blending = self.config.get('boundary_blending', True)
        self.integrity_watermark = self.config.get('integrity_watermark', True)
        
        logger.info("Fusion Engine initialized")
    
    def initialize(self):
        """Initialize engine."""
        logger.info("Fusion Engine ready")
    
    def fuse(self, encrypted_blocks: np.ndarray, 
             original_shape: Tuple[int, int, int],
             block_assignments: Optional[Dict] = None,
             block_size: int = 8) -> np.ndarray:
        """
        Fuse encrypted blocks into single image.
        
        Args:
            encrypted_blocks: Array of encrypted blocks (num_blocks, block_size, block_size, C)
            original_shape: (H, W, C) original image shape
            block_assignments: Per-block encryption info
            block_size: Size of blocks
            
        Returns:
            Fused encrypted image (H, W, C)
        """
        try:
            h, w, c = original_shape
            
            # Crop to multiple of block_size
            h = (h // block_size) * block_size
            w = (w // block_size) * block_size
            
            # Initialize output
            fused_image = np.zeros((h, w, c), dtype=np.uint8)
            
            # Reassemble blocks
            num_h = h // block_size
            num_w = w // block_size
            
            block_idx = 0
            for row in range(num_h):
                for col in range(num_w):
                    if block_idx >= len(encrypted_blocks):
                        break
                    
                    block = encrypted_blocks[block_idx]
                    
                    # Apply overlay strategy
                    block = self._apply_overlay_strategy(
                        block, block_idx, num_h, num_w,
                        block_assignments
                    )
                    
                    # Place block
                    r_start = row * block_size
                    r_end = r_start + block_size
                    c_start = col * block_size
                    c_end = c_start + block_size
                    
                    fused_image[r_start:r_end, c_start:c_end, :] = block
                    block_idx += 1
            
            # Apply boundary blending if enabled
            if self.boundary_blending:
                fused_image = self._apply_boundary_blending(fused_image, block_size)
            
            # Apply integrity watermark if enabled
            if self.integrity_watermark:
                fused_image = self._apply_integrity_watermark(fused_image)
            
            logger.info(f"Fused encrypted image: {fused_image.shape}")
            return fused_image
        
        except Exception as e:
            logger.error(f"Failed to fuse blocks: {str(e)}")
            raise
    
    def _apply_overlay_strategy(self, block: np.ndarray, block_id: int,
                               num_h: int, num_w: int,
                               block_assignments: Optional[Dict] = None) -> np.ndarray:
        """
        Apply overlay strategy to block.
        
        Strategies:
        - 'random': Random pixel permutation
        - 'spiral': Spiral pattern permutation
        - 'diagonal': Diagonal pattern
        """
        if self.overlay_strategy == 'random':
            return self._random_overlay(block, block_id)
        elif self.overlay_strategy == 'spiral':
            return self._spiral_overlay(block)
        elif self.overlay_strategy == 'diagonal':
            return self._diagonal_overlay(block)
        else:
            return block
    
    def _random_overlay(self, block: np.ndarray, seed: int) -> np.ndarray:
        """Apply random permutation to block."""
        if not self.pixel_scrambling:
            return block
        
        np.random.seed(seed)
        h, w, c = block.shape
        
        # Create permutation
        indices = np.arange(h * w)
        np.random.shuffle(indices)
        
        # Apply permutation
        permuted = block.reshape(-1, c)[indices].reshape(h, w, c)
        return permuted
    
    def _spiral_overlay(self, block: np.ndarray) -> np.ndarray:
        """Apply spiral pattern permutation."""
        if not self.pixel_scrambling:
            return block
        
        h, w, c = block.shape
        
        # Create spiral indices
        spiral_indices = self._create_spiral_indices(h, w)
        
        # Reshape and permute
        flat_block = block.reshape(-1, c)
        spiral_block = flat_block[spiral_indices].reshape(h, w, c)
        
        return spiral_block
    
    def _diagonal_overlay(self, block: np.ndarray) -> np.ndarray:
        """Apply diagonal pattern permutation."""
        if not self.pixel_scrambling:
            return block
        
        h, w, c = block.shape
        
        # Create diagonal indices
        diag_indices = []
        for d in range(h + w - 1):
            for i in range(max(0, d - w + 1), min(d + 1, h)):
                diag_indices.append(i * w + (d - i))
        
        # Reshape and permute
        flat_block = block.reshape(-1, c)
        diag_block = flat_block[diag_indices[:len(flat_block)]].reshape(h, w, c)
        
        return diag_block
    
    def _create_spiral_indices(self, h: int, w: int) -> np.ndarray:
        """Create spiral traversal indices."""
        indices = []
        top, bottom, left, right = 0, h - 1, 0, w - 1
        
        while top <= bottom and left <= right:
            # Right
            for col in range(left, right + 1):
                indices.append(top * w + col)
            top += 1
            
            # Down
            for row in range(top, bottom + 1):
                indices.append(row * w + right)
            right -= 1
            
            # Left
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    indices.append(bottom * w + col)
                bottom -= 1
            
            # Up
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    indices.append(row * w + left)
                left += 1
        
        return np.array(indices)
    
    def _apply_boundary_blending(self, fused_image: np.ndarray,
                                block_size: int = 8) -> np.ndarray:
        """
        Apply boundary blending between blocks.
        Smooths transitions between block boundaries.
        """
        try:
            result = fused_image.copy().astype(np.float32)
            h, w, c = result.shape
            
            # Apply soft blending at block boundaries
            for row in range(1, h // block_size):
                boundary = row * block_size
                
                # Blend vertical boundaries
                for col in range(w):
                    # Average neighboring pixels
                    if boundary > 0 and boundary < h:
                        alpha = 0.5
                        for ch in range(c):
                            result[boundary, col, ch] = (
                                alpha * result[boundary - 1, col, ch] +
                                (1 - alpha) * result[boundary, col, ch]
                            )
            
            # Clamp to valid range
            result = np.clip(result, 0, 255).astype(np.uint8)
            logger.info("Applied boundary blending")
            
            return result
        
        except Exception as e:
            logger.warning(f"Boundary blending failed: {str(e)}")
            return fused_image
    
    def _apply_integrity_watermark(self, fused_image: np.ndarray) -> np.ndarray:
        """
        Apply integrity watermark.
        Embeds verification information in LSBs.
        """
        try:
            import hashlib
            
            # Compute hash of image
            image_hash = hashlib.sha256(fused_image.tobytes()).digest()
            
            # Extract first few bytes for watermark
            watermark = image_hash[:4]
            watermark_bits = ''.join(format(byte, '08b') for byte in watermark)
            
            # Embed in LSBs of first 32 pixels
            result = fused_image.copy()
            flat = result.reshape(-1)
            
            for i, bit in enumerate(watermark_bits):
                if i < len(flat):
                    # Clear LSB and set to watermark bit
                    flat[i] = (flat[i] & 0xFE) | int(bit)
            
            result = flat.reshape(fused_image.shape)
            logger.info("Applied integrity watermark")
            
            return result
        
        except Exception as e:
            logger.warning(f"Integrity watermark failed: {str(e)}")
            return fused_image
    
    def verify_watermark(self, fused_image: np.ndarray) -> bool:
        """
        Verify integrity watermark.
        
        Args:
            fused_image: Image with embedded watermark
            
        Returns:
            True if watermark valid
        """
        try:
            import hashlib
            
            flat = fused_image.reshape(-1)
            
            # Extract watermark bits
            watermark_bits = ''.join(
                str(flat[i] & 1) for i in range(32)
            )
            
            # Reconstruct original image (clear LSBs)
            original = fused_image.copy()
            original.reshape(-1)[:32] = original.reshape(-1)[:32] & 0xFE
            
            # Compute hash
            actual_hash = hashlib.sha256(original.tobytes()).digest()
            expected_bits = ''.join(format(byte, '08b') for byte in actual_hash[:4])
            
            # Compare
            match = watermark_bits == expected_bits
            logger.info(f"Watermark verification: {'✓' if match else '✗'}")
            
            return match
        
        except Exception as e:
            logger.warning(f"Watermark verification failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    config = {'fusion_engine': {}}
    engine = FusionEngine(config)
    engine.initialize()
    
    # Create test blocks
    blocks = np.random.randint(0, 256, (256, 8, 8, 3), dtype=np.uint8)
    
    # Fuse
    fused = engine.fuse(blocks, (256, 256, 3))
    
    print(f"Fused image shape: {fused.shape}")
    print("Fusion Engine OK")
