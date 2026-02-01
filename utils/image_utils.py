"""
Image utilities for loading, saving, and processing images.
Independent module - no dependencies on other project code.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger("image_utils")


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image array (H, W, C) uint8
    """
    try:
        from PIL import Image
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = Image.open(path)
        
        # Convert to RGB if needed
        if img.mode in ('L', 'RGBA'):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        array = np.array(img, dtype=np.uint8)
        logger.info(f"Loaded image: {image_path}, shape: {array.shape}")
        
        return array
    
    except Exception as e:
        logger.error(f"Failed to load image: {str(e)}")
        raise


def save_image(image_array: np.ndarray, output_path: str) -> bool:
    """
    Save image array to file.
    
    Args:
        image_array: Image array (H, W, 3) uint8
        output_path: Path to save
        
    Returns:
        True if successful
    """
    try:
        from PIL import Image
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure uint8
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        # Save
        img = Image.fromarray(image_array, mode='RGB')
        img.save(path)
        
        logger.info(f"Saved image: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save image: {str(e)}")
        return False


def get_image_info(image_array: np.ndarray) -> dict:
    """
    Get image information.
    
    Args:
        image_array: Image array
        
    Returns:
        Dict with image info
    """
    return {
        'shape': image_array.shape,
        'dtype': str(image_array.dtype),
        'min': int(image_array.min()),
        'max': int(image_array.max()),
        'mean': float(image_array.mean()),
        'std': float(image_array.std()),
        'size_bytes': int(image_array.nbytes)
    }


def crop_to_blocks(image_array: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Crop image to multiple of block size.
    
    Args:
        image_array: (H, W, C) array
        block_size: Block size
        
    Returns:
        Cropped array
    """
    h, w, c = image_array.shape
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    
    return image_array[:h_crop, :w_crop, :].copy()


def extract_blocks(image_array: np.ndarray, block_size: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Extract blocks from image.
    
    Args:
        image_array: (H, W, C) array
        block_size: Block size
        
    Returns:
        (blocks, original_shape)
        blocks shape: (num_blocks, block_size, block_size, C)
    """
    # Ensure multiple of block size
    image_array = crop_to_blocks(image_array, block_size)
    
    h, w, c = image_array.shape
    num_h = h // block_size
    num_w = w // block_size
    
    # Reshape into blocks
    blocks = image_array.reshape(
        num_h, block_size,
        num_w, block_size,
        c
    ).transpose(0, 2, 1, 3, 4).reshape(
        num_h * num_w, block_size, block_size, c
    )
    
    return blocks, (h, w, c)


def reassemble_blocks(blocks: np.ndarray, original_shape: Tuple[int, int, int],
                     block_size: int = 8) -> np.ndarray:
    """
    Reassemble blocks into image.
    
    Args:
        blocks: (num_blocks, block_size, block_size, C) array
        original_shape: (H, W, C)
        block_size: Block size
        
    Returns:
        Reassembled image (H, W, C)
    """
    h, w, c = original_shape
    num_h = h // block_size
    num_w = w // block_size
    
    # Reshape blocks back into image
    image = blocks.reshape(
        num_h, num_w, block_size, block_size, c
    ).transpose(0, 2, 1, 3, 4).reshape(h, w, c)
    
    return image
