"""
AI Engine - Phase 2
Semantic Segmentation for ROI Detection

Integrates with FlexiMo from cloned repository
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging
from pathlib import Path
import sys


class AIEngine:
    """
    AI-powered semantic segmentation engine.
    Detects regions of interest (ROI) in satellite/aerial imagery.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AI Engine.
        
        Args:
            config: Configuration dict with ai_engine settings
        """
        self.config = config.get('ai_engine', {})
        self.logger = logging.getLogger('ai_engine')
        self.is_initialized = False
        
        # Try to import FlexiMo from cloned repository
        self.use_fleximo = False
        self.fleximo_module = None
        
        try:
            # Try to import from fleximo_repo that was cloned
            import fleximo_repo
            self.fleximo_module = fleximo_repo
            self.logger.info("✓ FlexiMo repository module imported successfully")
            self.use_fleximo = True
        except ImportError as e:
            self.logger.warning(f"Could not import fleximo_repo: {e}")
            self.logger.warning("Using fallback dummy segmentation")
            self.use_fleximo = False
    
    def initialize(self):
        """Initialize engine and prepare for processing."""
        self.is_initialized = True
        self.logger.info("AI Engine initialized")
    
    def validate_input(self, image: np.ndarray) -> bool:
        """Validate image input."""
        if not isinstance(image, np.ndarray):
            return False
        if len(image.shape) not in [2, 3]:
            return False
        if image.dtype != np.uint8:
            return False
        return True
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform semantic segmentation on image to detect ROI.
        
        Args:
            image: Input image (H, W, 3) or (H, W) uint8 array
            
        Returns:
            ROI mask (H, W) uint8: 255 for ROI, 0 for background
        """
        if not self.is_initialized:
            self.initialize()
        
        if not self.validate_input(image):
            self.logger.error("Invalid image input")
            # Return fallback: all pixels as ROI
            return np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        
        try:
            if self.use_fleximo and self.fleximo_module:
                # Use FlexiMo from cloned repository
                self.logger.info(f"Running FlexiMo segmentation (from cloned repo) on {image.shape}")
                
                # Call FlexiMo function from repo if available
                try:
                    from fleximo_repo import segment_image_fleximo
                    roi_mask = segment_image_fleximo(image)
                    self.logger.info(f"✓ FlexiMo segmentation completed")
                    return roi_mask
                except (ImportError, AttributeError):
                    self.logger.warning("FlexiMo function not directly callable, using fallback")
                    return self._fallback_segmentation(image)
            else:
                # Fallback: simple edge-based segmentation
                return self._fallback_segmentation(image)
        
        except Exception as e:
            self.logger.error(f"Segmentation failed: {str(e)}")
            return self._fallback_segmentation(image)
    
    def _fallback_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback segmentation when FlexiMo is unavailable.
        Uses simple contrast-based detection.
        """
        self.logger.info("Using fallback segmentation (contrast-based)")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(np.float32)
        
        # Apply simple thresholding based on standard deviation
        threshold = np.mean(gray) + self.config.get('threshold_std_factor', 1.0) * np.std(gray)
        roi_mask = (gray > threshold).astype(np.uint8) * 255
        
        # Apply slight dilation to expand ROI regions
        from scipy import ndimage
        roi_mask = ndimage.binary_dilation(roi_mask, iterations=2).astype(np.uint8) * 255
        
        self.logger.info(f"Fallback ROI Detection: {np.count_nonzero(roi_mask > 0)} / {roi_mask.size} pixels")
        return roi_mask
    
    def get_summary(self) -> Dict[str, Any]:
        """Get engine summary."""
        return {
            'engine': 'AI Engine (Phase 2)',
            'model': 'FlexiMo (from cloned repo)' if self.use_fleximo else 'Fallback (Contrast-based)',
            'status': 'initialized' if self.is_initialized else 'not_initialized',
            'repo_loaded': self.fleximo_module is not None,
            'config': self.config
        }
