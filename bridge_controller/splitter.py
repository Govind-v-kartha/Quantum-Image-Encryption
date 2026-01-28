"""
Image Splitting Module
Splits satellite image into ROI (Region of Interest) and Background matrices
based on semantic segmentation mask.
"""

import numpy as np
import cv2
from typing import Tuple
from pathlib import Path


class ImageSplitter:
    """
    Splits an image into ROI and Background components based on a binary mask.
    
    Mathematical Operations:
    - ROI Matrix: I_ROI = I × M (where M is binary mask)
    - Background Matrix: I_BG = I × (1-M)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the ImageSplitter.
        
        Args:
            verbose: Print processing information
        """
        self.verbose = verbose
    
    def split_image(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split image into ROI and Background matrices.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            mask: Binary segmentation mask (H, W) with values 0 or 1
        
        Returns:
            roi_image: ROI matrix (I_ROI = I × M)
            bg_image: Background matrix (I_BG = I × (1-M))
        """
        # Ensure mask is binary
        mask = (mask > 0.5).astype(np.float32)
        
        # Handle different image formats
        if len(image.shape) == 3:
            # Multi-channel image: expand mask to match channels
            mask_expanded = np.stack([mask] * image.shape[2], axis=2)
        else:
            # Single channel image
            mask_expanded = mask
        
        # Split according to formulas
        roi_image = image * mask_expanded
        bg_image = image * (1 - mask_expanded)
        
        if self.verbose:
            print(f"Image split completed:")
            print(f"  Input shape: {image.shape}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  ROI pixels (non-zero): {np.count_nonzero(roi_image)}")
            print(f"  Background pixels (non-zero): {np.count_nonzero(bg_image)}")
        
        return roi_image, bg_image
    
    def validate_split(
        self, 
        original: np.ndarray, 
        roi: np.ndarray, 
        bg: np.ndarray,
        mask: np.ndarray
    ) -> bool:
        """
        Validate that ROI + Background reconstructs original image.
        
        Args:
            original: Original image
            roi: ROI matrix
            bg: Background matrix
            mask: Binary mask used for splitting
        
        Returns:
            True if reconstruction is valid
        """
        # Ensure mask is binary
        mask = (mask > 0.5).astype(np.float32)
        
        # Handle different image formats
        if len(original.shape) == 3:
            mask_expanded = np.stack([mask] * original.shape[2], axis=2)
        else:
            mask_expanded = mask
        
        # Reconstruct: I = I_ROI + I_BG
        reconstructed = roi + bg
        
        # Check if reconstruction matches original
        mse = np.mean((original - reconstructed) ** 2)
        max_error = np.max(np.abs(original - reconstructed))
        
        is_valid = mse < 1e-6
        
        if self.verbose:
            print(f"Reconstruction validation:")
            print(f"  MSE: {mse:.2e}")
            print(f"  Max error: {max_error:.2e}")
            print(f"  Valid: {is_valid}")
        
        return is_valid
    
    def save_split_images(
        self,
        roi_image: np.ndarray,
        bg_image: np.ndarray,
        output_dir: str,
        prefix: str = "split"
    ) -> Tuple[str, str]:
        """
        Save split images to disk.
        
        Args:
            roi_image: ROI matrix
            bg_image: Background matrix
            output_dir: Directory to save images
            prefix: Prefix for output filenames
        
        Returns:
            Tuple of (roi_path, bg_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Normalize for visualization (if float32 in range [0, 1])
        roi_display = (roi_image * 255).astype(np.uint8) if roi_image.max() <= 1 else roi_image.astype(np.uint8)
        bg_display = (bg_image * 255).astype(np.uint8) if bg_image.max() <= 1 else bg_image.astype(np.uint8)
        
        roi_path = str(output_path / f"{prefix}_roi.png")
        bg_path = str(output_path / f"{prefix}_background.png")
        
        cv2.imwrite(roi_path, roi_display)
        cv2.imwrite(bg_path, bg_display)
        
        if self.verbose:
            print(f"Split images saved:")
            print(f"  ROI: {roi_path}")
            print(f"  Background: {bg_path}")
        
        return roi_path, bg_path


def load_image(image_path: str, normalize: bool = True) -> Tuple[np.ndarray, str]:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to image file
        normalize: Normalize pixel values to [0, 1]
    
    Returns:
        Image array and format info
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize if requested
    if normalize and image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    return image, image.dtype


def load_mask(mask_path: str) -> np.ndarray:
    """
    Load a segmentation mask from disk.
    
    Args:
        mask_path: Path to mask file
    
    Returns:
        Binary mask array
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    # Normalize to [0, 1]
    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32) / 255.0
    
    return mask
