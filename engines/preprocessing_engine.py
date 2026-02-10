"""
Layer 1: Preprocessing Engine
Loads, validates, and prepares input images for the encryption pipeline.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from PIL import Image


class PreprocessingEngine:
    """Load and validate input images. Ensure dtype=uint8, RGB format."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('preprocessing_engine', {})
        self.logger = logging.getLogger('preprocessing_engine')
        self.is_initialized = False

    def initialize(self):
        self.is_initialized = True
        self.logger.info("Preprocessing Engine initialized")

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from disk, convert to RGB uint8.

        Returns:
            (H, W, 3) uint8 ndarray
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(str(path))

        # Convert to RGB
        if img.mode == 'L':
            img = img.convert('RGB')
            self.logger.info("  Converted grayscale -> RGB")
        elif img.mode == 'RGBA':
            img = img.convert('RGB')
            self.logger.info("  Converted RGBA -> RGB (alpha dropped)")
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            self.logger.info(f"  Converted {img.mode} -> RGB")

        image = np.array(img, dtype=np.uint8)
        self.logger.info(f"  Loaded image: shape={image.shape}, dtype={image.dtype}")
        return image

    def validate(self, image: np.ndarray) -> bool:
        """Validate image is (H, W, 3) uint8."""
        if not isinstance(image, np.ndarray):
            return False
        if image.dtype != np.uint8:
            return False
        if image.ndim != 3 or image.shape[2] != 3:
            return False
        return True

    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        """Save image to disk as PNG (lossless)."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(image)
            img.save(output_path, format='PNG')
            self.logger.info(f"  Saved image: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"  Failed to save image: {e}")
            return False
