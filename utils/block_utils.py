"""
Block utilities for block-level operations.
Independent module - handles block-level statistics and analysis.
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger("block_utils")


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
