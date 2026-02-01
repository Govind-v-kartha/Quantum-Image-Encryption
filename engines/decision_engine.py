"""
Decision Engine - Phase 3
Adaptive Encryption Level Assignment

Makes decisions about which blocks get which encryption level.
Integrates with existing adaptive decision logic from /core/decision_engine/
"""

import numpy as np
from typing import Dict, Any, List
import logging


class DecisionEngine:
    """
    Makes adaptive encryption decisions based on image content and ROI analysis.
    Determines which blocks use quantum vs classical encryption.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Decision Engine.
        
        Args:
            config: Configuration dict with decision_engine settings
        """
        self.config = config.get('decision_engine', {})
        self.logger = logging.getLogger('decision_engine')
        self.is_initialized = False
        
        # Load thresholds from config
        self.roi_thresholds = self.config.get('roi_thresholds', {
            'small': 100,
            'medium': 500,
            'large': 2000,
            'huge': 5000
        })
        
        # Encryption level assignments
        self.encryption_levels = {
            'FULL_QUANTUM': 3,     # Maximum security
            'HYBRID': 2,           # Mixed quantum-classical
            'CLASSICAL_ONLY': 1    # Classical only
        }
    
    def initialize(self):
        """Initialize engine and prepare for processing."""
        self.is_initialized = True
        self.logger.info("Decision Engine initialized")
    
    def validate_input(self, roi_mask: np.ndarray) -> bool:
        """Validate ROI mask input."""
        if not isinstance(roi_mask, np.ndarray):
            return False
        if len(roi_mask.shape) != 2:
            return False
        if roi_mask.dtype != np.uint8:
            return False
        return True
    
    def decide(self, roi_mask: np.ndarray, image_shape: tuple) -> Dict[str, Any]:
        """
        Make encryption decisions based on ROI analysis.
        
        Args:
            roi_mask: ROI detection mask (H, W) uint8
            image_shape: Original image shape (H, W, C)
            
        Returns:
            Decision dict with encryption allocations
        """
        if not self.is_initialized:
            self.initialize()
        
        if not self.validate_input(roi_mask):
            self.logger.error("Invalid ROI mask")
            # Default decision: full quantum encryption
            return {
                'roi_category': 'unknown',
                'primary_encryption_level': 'FULL_QUANTUM',
                'adaptive_key_length': 256,
                'block_assignments': {}
            }
        
        try:
            # Analyze ROI size
            roi_pixels = np.count_nonzero(roi_mask > 127)
            total_pixels = roi_mask.size
            roi_percentage = (roi_pixels / total_pixels) * 100
            
            self.logger.info(f"ROI Analysis: {roi_pixels}/{total_pixels} pixels ({roi_percentage:.1f}%)")
            
            # Categorize ROI
            roi_category = self._categorize_roi(roi_pixels)
            
            # Determine encryption level
            encryption_level = self._determine_encryption_level(roi_category)
            
            # Determine adaptive key length
            key_length = self._determine_key_length(roi_category)
            
            # Create block assignments (simplified - full image treatment)
            block_assignments = {
                'default': encryption_level,
                'roi_category': roi_category,
                'roi_pixels': roi_pixels,
                'roi_percentage': roi_percentage
            }
            
            decision = {
                'roi_category': roi_category,
                'primary_encryption_level': encryption_level,
                'adaptive_key_length': key_length,
                'block_assignments': block_assignments,
                'roi_analysis': {
                    'roi_pixels': roi_pixels,
                    'total_pixels': total_pixels,
                    'roi_percentage': roi_percentage
                }
            }
            
            self.logger.info(f"Decision: {encryption_level} with {key_length}-bit keys")
            return decision
        
        except Exception as e:
            self.logger.error(f"Decision making failed: {str(e)}")
            # Default fallback
            return {
                'roi_category': 'error',
                'primary_encryption_level': 'FULL_QUANTUM',
                'adaptive_key_length': 256,
                'block_assignments': {}
            }
    
    def _categorize_roi(self, roi_pixels: int) -> str:
        """Categorize ROI based on size."""
        thresholds = self.roi_thresholds
        
        if roi_pixels < thresholds.get('small', 100):
            return 'tiny'
        elif roi_pixels < thresholds.get('medium', 500):
            return 'small'
        elif roi_pixels < thresholds.get('large', 2000):
            return 'medium'
        elif roi_pixels < thresholds.get('huge', 5000):
            return 'large'
        else:
            return 'huge'
    
    def _determine_encryption_level(self, roi_category: str) -> str:
        """Determine encryption level based on ROI category."""
        level_map = self.config.get('encryption_level_map', {
            'tiny': 'CLASSICAL_ONLY',
            'small': 'CLASSICAL_ONLY',
            'medium': 'HYBRID',
            'large': 'FULL_QUANTUM',
            'huge': 'FULL_QUANTUM',
            'error': 'FULL_QUANTUM'
        })
        
        return level_map.get(roi_category, 'FULL_QUANTUM')
    
    def _determine_key_length(self, roi_category: str) -> int:
        """Determine adaptive key length based on ROI category."""
        key_map = self.config.get('key_length_map', {
            'tiny': 128,
            'small': 192,
            'medium': 256,
            'large': 256,
            'huge': 512,
            'error': 256
        })
        
        return key_map.get(roi_category, 256)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get engine summary."""
        return {
            'engine': 'Decision Engine (Phase 3)',
            'status': 'initialized' if self.is_initialized else 'not_initialized',
            'thresholds': self.roi_thresholds,
            'config': self.config
        }
