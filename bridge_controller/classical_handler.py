"""
Classical Encryption Handler
Integrates Chaos Maps (Logistic Sine Map) for fast encryption of background images.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class ClassicalEncryptionHandler:
    """
    Classical encryption using Chaos Maps for background (non-sensitive) regions.
    
    Features:
    - Hybrid Logistic Sine Map (HLSM) for chaotic key generation
    - Fast XOR-based encryption
    - Efficient for bulk data processing
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize Classical Encryption Handler.
        
        Args:
            verbose: Print processing information
        """
        self.verbose = verbose
    
    def logistic_map(
        self, 
        x0: float, 
        r: float, 
        iterations: int
    ) -> np.ndarray:
        """
        Logistic Map for chaos generation.
        
        Formula: x(n+1) = r × x(n) × (1 - x(n))
        
        Args:
            x0: Initial value (0 < x0 < 1)
            r: Parameter (typically 3.57 to 4.0 for chaotic behavior)
            iterations: Number of iterations
        
        Returns:
            Chaotic sequence
        """
        x = np.zeros(iterations)
        x[0] = x0
        
        for i in range(1, iterations):
            x[i] = r * x[i-1] * (1 - x[i-1])
        
        return x
    
    def sine_map(
        self, 
        x0: float, 
        iterations: int
    ) -> np.ndarray:
        """
        Sine Map for chaos generation.
        
        Formula: x(n+1) = β × sin(π × x(n))
        
        Args:
            x0: Initial value (0 < x0 < 1)
            iterations: Number of iterations
        
        Returns:
            Chaotic sequence
        """
        x = np.zeros(iterations)
        x[0] = x0
        beta = 4.0  # Chaotic parameter
        
        for i in range(1, iterations):
            x[i] = beta * np.sin(np.pi * x[i-1])
        
        return x
    
    def hybrid_logistic_sine_map(
        self,
        x0: float,
        y0: float,
        r: float,
        iterations: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hybrid Logistic-Sine Map (HLSM) for enhanced chaotic behavior.
        
        Combines logistic and sine maps:
        - x(n+1) = r × sin(π × y(n))
        - y(n+1) = r × x(n) × (1 - x(n))
        
        Args:
            x0: Initial x value (0 < x0 < 1)
            y0: Initial y value (0 < y0 < 1)
            r: Chaotic parameter
            iterations: Number of iterations
        
        Returns:
            Tuple of (x_sequence, y_sequence)
        """
        x = np.zeros(iterations)
        y = np.zeros(iterations)
        
        x[0] = x0
        y[0] = y0
        
        for i in range(1, iterations):
            x[i] = r * np.sin(np.pi * y[i-1])
            y[i] = r * x[i-1] * (1 - x[i-1])
        
        return x, y
    
    def generate_chaos_key(
        self,
        shape: Tuple[int, ...],
        seed_x: float = 0.3,
        seed_y: float = 0.7,
        r: float = 3.99
    ) -> np.ndarray:
        """
        Generate chaotic encryption key from HLSM.
        
        Args:
            shape: Shape of the key to generate
            seed_x: Initial x value for HLSM
            seed_y: Initial y value for HLSM
            r: Chaotic parameter
        
        Returns:
            Chaotic key as uint8 array
        """
        total_elements = np.prod(shape)
        
        # Generate HLSM sequences
        x_seq, y_seq = self.hybrid_logistic_sine_map(
            seed_x, seed_y, r, total_elements * 2
        )
        
        # Combine and normalize to [0, 255]
        combined = (x_seq[:total_elements] + y_seq[:total_elements]) / 2.0
        key = (combined * 255).astype(np.uint8)
        
        # Reshape to target shape
        key = key.reshape(shape)
        
        if self.verbose:
            print(f"Chaos key generated:")
            print(f"  Shape: {key.shape}")
            print(f"  Min value: {key.min()}")
            print(f"  Max value: {key.max()}")
            print(f"  Mean value: {key.mean():.2f}")
        
        return key
    
    def encrypt_background(
        self,
        bg_image: np.ndarray,
        seed_x: Optional[float] = None,
        seed_y: Optional[float] = None,
        r: float = 3.99
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encrypt background image using chaos map XOR.
        
        Steps:
        1. Generate chaotic key from HLSM
        2. XOR image with chaotic key
        
        Args:
            bg_image: Background matrix (non-sensitive regions)
            seed_x: HLSM seed x (random if None)
            seed_y: HLSM seed y (random if None)
            r: Chaotic parameter
        
        Returns:
            Encrypted background and chaos key
        """
        if self.verbose:
            print("=" * 60)
            print("Classical Encryption Pipeline (Background)")
            print("=" * 60)
        
        # Use random seeds if not provided
        if seed_x is None:
            seed_x = np.random.rand()
        if seed_y is None:
            seed_y = np.random.rand()
        
        if self.verbose:
            print(f"HLSM Parameters:")
            print(f"  Seed X: {seed_x:.6f}")
            print(f"  Seed Y: {seed_y:.6f}")
            print(f"  Parameter r: {r}")
        
        # Generate chaos key
        key = self.generate_chaos_key(bg_image.shape, seed_x, seed_y, r)
        
        # Ensure background image is uint8
        if bg_image.dtype != np.uint8:
            bg_uint8 = (bg_image * 255).astype(np.uint8)
        else:
            bg_uint8 = bg_image.copy()
        
        # Apply XOR encryption
        encrypted = np.bitwise_xor(bg_uint8, key)
        
        if self.verbose:
            print(f"Encryption complete!")
            print(f"  Input shape: {bg_image.shape}")
            print(f"  Encrypted shape: {encrypted.shape}")
            print(f"  Entropy: {self._calculate_entropy(encrypted):.4f}")
        
        return encrypted, key
    
    @staticmethod
    def _calculate_entropy(image: np.ndarray) -> float:
        """Calculate Shannon entropy of image."""
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        return entropy
    
    def decrypt_background(
        self,
        encrypted_image: np.ndarray,
        key: np.ndarray
    ) -> np.ndarray:
        """
        Decrypt background image (XOR is reversible).
        
        Args:
            encrypted_image: Encrypted background image
            key: Chaos key used for encryption
        
        Returns:
            Decrypted image
        """
        decrypted = np.bitwise_xor(encrypted_image, key)
        
        if self.verbose:
            print(f"Decryption complete!")
            print(f"  Decrypted shape: {decrypted.shape}")
        
        return decrypted
    
    def save_encrypted_background(
        self,
        encrypted_image: np.ndarray,
        key: np.ndarray,
        output_dir: str,
        prefix: str = "encrypted_background"
    ) -> Tuple[str, str]:
        """
        Save encrypted background image and key.
        
        Args:
            encrypted_image: Encrypted image array
            key: Chaos key
            output_dir: Directory to save
            prefix: Output filename prefix
        
        Returns:
            Tuple of (image_path, key_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save encrypted image
        image_file = output_path / f"{prefix}.npy"
        np.save(str(image_file), encrypted_image)
        
        # Save key
        key_file = output_path / f"{prefix}_key.npy"
        np.save(str(key_file), key)
        
        if self.verbose:
            print(f"Encrypted background saved: {image_file}")
            print(f"Chaos key saved: {key_file}")
        
        return str(image_file), str(key_file)
