"""
Block utilities for block-level operations.
Independent module - handles block creation, indexing, and manipulation.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger("block_utils")


class BlockMetadata:
    """Metadata for a single block."""
    
    def __init__(self, block_id: int, position: Tuple[int, int], size: int):
        self.block_id = block_id
        self.position = position  # (row, col)
        self.size = size
        self.encryption_level = None
        self.encrypted = False
        self.key = None
        self.nonce = None
        self.auth_tag = None


def create_block_index(height: int, width: int, block_size: int = 8) -> Dict[int, Tuple[int, int]]:
    """
    Create mapping from block_id to position.
    
    Args:
        height: Image height
        width: Image width
        block_size: Size of each block
        
    Returns:
        Dict mapping block_id → (row, col)
    """
    block_index = {}
    block_id = 0
    
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            block_index[block_id] = (row // block_size, col // block_size)
            block_id += 1
    
    logger.info(f"Created block index with {block_id} blocks")
    return block_index


def is_edge_block(block_id: int, row: int, col: int, 
                 num_rows: int, num_cols: int) -> bool:
    """
    Check if block is on image edge.
    
    Args:
        block_id: Block ID
        row: Block row
        col: Block column
        num_rows: Total block rows
        num_cols: Total block columns
        
    Returns:
        True if block is on edge
    """
    return row == 0 or row == num_rows - 1 or col == 0 or col == num_cols - 1


def is_corner_block(row: int, col: int, num_rows: int, num_cols: int) -> bool:
    """
    Check if block is a corner block.
    
    Args:
        row: Block row
        col: Block column
        num_rows: Total block rows
        num_cols: Total block columns
        
    Returns:
        True if corner block
    """
    return (row == 0 or row == num_rows - 1) and (col == 0 or col == num_cols - 1)


def get_block_neighbors(block_id: int, row: int, col: int,
                       num_rows: int, num_cols: int) -> List[int]:
    """
    Get IDs of neighboring blocks.
    
    Args:
        block_id: Current block ID
        row: Block row
        col: Block column
        num_rows: Total block rows
        num_cols: Total block columns
        
    Returns:
        List of neighbor block IDs
    """
    neighbors = []
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            
            nr, nc = row + dr, col + dc
            if 0 <= nr < num_rows and 0 <= nc < num_cols:
                neighbor_id = nr * num_cols + nc
                neighbors.append(neighbor_id)
    
    return neighbors


def extract_block_with_padding(image: np.ndarray, row: int, col: int,
                              block_size: int = 8,
                              padding: int = 1) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract block with padding for edge detection.
    
    Args:
        image: Full image (H, W, C)
        row: Block row
        col: Block column
        block_size: Size of block
        padding: Padding around block
        
    Returns:
        (padded_block, crop_coords)
    """
    h, w, c = image.shape
    
    # Calculate coordinates
    r_start = max(0, row * block_size - padding)
    r_end = min(h, (row + 1) * block_size + padding)
    c_start = max(0, col * block_size - padding)
    c_end = min(w, (col + 1) * block_size + padding)
    
    padded_block = image[r_start:r_end, c_start:c_end, :].copy()
    
    return padded_block, (r_start, r_end, c_start, c_end)


def compute_block_statistics(block: np.ndarray) -> Dict:
    """
    Compute statistics for a block.
    
    Args:
        block: Block array (size, size, C)
        
    Returns:
        Dict with statistics
    """
    return {
        'mean': float(block.mean()),
        'std': float(block.std()),
        'min': int(block.min()),
        'max': int(block.max()),
        'entropy': float(compute_entropy(block)),
        'variance': float(block.var())
    }


def compute_entropy(data: np.ndarray) -> float:
    """
    Compute entropy of data.
    
    Args:
        data: Data array
        
    Returns:
        Entropy in bits
    """
    flat = data.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    probabilities = counts / len(flat)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy


def create_scramble_pattern(block_size: int, seed: int) -> np.ndarray:
    """
    Create deterministic scramble pattern.
    
    Args:
        block_size: Size of block
        seed: Random seed
        
    Returns:
        (block_size, block_size) permutation indices
    """
    np.random.seed(seed)
    indices = np.arange(block_size * block_size)
    np.random.shuffle(indices)
    pattern = indices.reshape(block_size, block_size)
    return pattern


def apply_permutation(block: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """
    Apply permutation to block.
    
    Args:
        block: (size, size, C) block
        pattern: (size, size) permutation indices
        
    Returns:
        Permuted block
    """
    size = block.shape[0]
    permuted = block.copy()
    
    flat_block = block.reshape(-1, block.shape[2])
    flat_pattern = pattern.flatten()
    
    permuted_flat = flat_block[flat_pattern]
    permuted = permuted_flat.reshape(size, size, block.shape[2])
    
    return permuted
