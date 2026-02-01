"""
AI Engine - Phase 2
Semantic Segmentation for ROI Detection

Integrates with existing FlexiMo ViT model from /core/ai_engine/
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging
from pathlib import Path


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
        
        # Try to import FlexiMo (optional, fallback to dummy if unavailable)
        self.use_fleximo = False
        try:
            from fleximo_integration import FlexiMoSegmentor
            self.fleximo_weights = Path(__file__).parent.parent / "models" / "DOFA_ViT_base_e100.pth"
            
            if self.fleximo_weights.exists():
                self.segmentor = FlexiMoSegmentor(
                    model_path=str(self.fleximo_weights),
                    device=self.config.get('device', 'cpu')
                )
                self.use_fleximo = True
                self.logger.info("FlexiMo ViT model loaded successfully")
            else:
                self.logger.warning(f"FlexiMo weights not found at {self.fleximo_weights}")
                self.logger.warning("Using fallback dummy segmentation")
                self.segmentor = None
        except ImportError:
            self.logger.warning("FlexiMo not available - using fallback segmentation")
            self.segmentor = None
    
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
            if self.use_fleximo and self.segmentor:
                # Use actual FlexiMo segmentation
                self.logger.info(f"Running FlexiMo ViT segmentation on {image.shape}")
                roi_mask = self.segmentor.segment(image)
                
                # Ensure output is (H, W) uint8
                if len(roi_mask.shape) == 3:
                    roi_mask = roi_mask[:, :, 0]
                roi_mask = roi_mask.astype(np.uint8)
                
                self.logger.info(f"ROI Detection: {np.count_nonzero(roi_mask > 127)} / {roi_mask.size} pixels")
                return roi_mask
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
            'model': 'FlexiMo ViT' if self.use_fleximo else 'Fallback (Contrast-based)',
            'status': 'initialized' if self.is_initialized else 'not_initialized',
            'config': self.config
        }
