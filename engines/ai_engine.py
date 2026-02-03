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
    Uses FlexiMo Vision Transformer for true semantic understanding.
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
        
        # FlexiMo model and weights
        self.use_fleximo = False
        self.fleximo_module = None
        self.fleximo_model = None
        self.device = 'cpu'
        
        try:
            # Try to import from fleximo_repo that was cloned
            import fleximo_repo
            self.fleximo_module = fleximo_repo
            self.logger.info("✓ FlexiMo repository module imported successfully")
            self.use_fleximo = True
            
            # Try to load FlexiMo model
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.logger.info(f"  Using device: {self.device}")
                
                # Attempt to load FlexiMo model
                self._load_fleximo_model()
            except ImportError:
                self.logger.warning("PyTorch not available, will use fallback segmentation")
                self.use_fleximo = False
                
        except ImportError as e:
            self.logger.warning(f"Could not import fleximo_repo: {e}")
            self.logger.warning("Using fallback contrast-based segmentation")
            self.use_fleximo = False
    
    def _load_fleximo_model(self):
        """Load FlexiMo pretrained model."""
        try:
            import torch
            from fleximo_repo.fleximo.models_dwv import get_model
            
            # Try to load model (this requires weights to be available)
            model_config = self.config.get('model_config', 'dwv_vit_base')
            pretrained = self.config.get('pretrained', True)
            
            self.logger.info(f"Loading FlexiMo model: {model_config}")
            self.fleximo_model = get_model(model_config, pretrained=pretrained)
            self.fleximo_model = self.fleximo_model.to(self.device)
            self.fleximo_model.eval()
            self.logger.info("✓ FlexiMo model loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not load FlexiMo model: {e}")
            self.logger.warning("Will use fallback segmentation")
            self.fleximo_model = None
    
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
            # Try FlexiMo semantic segmentation first
            if self.use_fleximo and self.fleximo_model is not None:
                return self._segment_with_fleximo(image)
            else:
                # Fallback: contrast-based segmentation
                return self._fallback_segmentation(image)
        
        except Exception as e:
            self.logger.error(f"Segmentation failed: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return self._fallback_segmentation(image)
    
    def _segment_with_fleximo(self, image: np.ndarray) -> np.ndarray:
        """
        Semantic segmentation using FlexiMo Vision Transformer.
        
        Args:
            image: Input image (H, W, 3) uint8
            
        Returns:
            ROI mask (H, W) uint8: 255 for ROI, 0 for background
        """
        try:
            import torch
            from torchvision import transforms
            
            self.logger.info(f"Running FlexiMo semantic segmentation on {image.shape}")
            
            # Normalize image to [0, 1] and prepare for model
            if len(image.shape) == 3:
                img_tensor = torch.from_numpy(image).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
            else:
                img_tensor = torch.from_numpy(image).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # Add channel
            
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)  # BCHW
            img_tensor = img_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.fleximo_model(img_tensor)
            
            # Convert output to mask
            if isinstance(output, torch.Tensor):
                if len(output.shape) == 4:  # BCHW
                    output = output.squeeze(0).squeeze(0)  # Remove batch and channel
                elif len(output.shape) == 3:  # CHW
                    output = output.squeeze(0)  # Remove channel
                
                # Convert to numpy and threshold
                mask = output.cpu().numpy().astype(np.float32)
                
                # Normalize to 0-1 range
                mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                
                # Apply threshold to get binary mask
                threshold = self.config.get('fleximo_threshold', 0.5)
                roi_mask = (mask > threshold).astype(np.uint8) * 255
                
                self.logger.info(f"✓ FlexiMo ROI Detection: {np.count_nonzero(roi_mask > 0)} / {roi_mask.size} pixels")
                return roi_mask
            else:
                self.logger.warning("FlexiMo output format unexpected")
                return self._fallback_segmentation(image)
        
        except Exception as e:
            self.logger.warning(f"FlexiMo segmentation failed: {str(e)}, using fallback")
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
