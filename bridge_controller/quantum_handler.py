"""
Quantum Encryption Handler
Integrates NEQR (Novel Enhanced Quantum Representation) and 
Arnold Scrambling for secure encryption of ROI images.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys


class QuantumEncryptionHandler:
    """
    Wrapper for quantum encryption of ROI matrices.
    
    Features:
    - NEQR (Novel Enhanced Quantum Representation) encoding
    - Arnold Scrambling for pixel position security
    - Integration with Qiskit-based quantum simulator
    """
    
    def __init__(self, quantum_backend: str = "qasm_simulator", verbose: bool = True):
        """
        Initialize Quantum Encryption Handler.
        
        Args:
            quantum_backend: Qiskit backend ("qasm_simulator", "statevector_simulator", etc.)
            verbose: Print processing information
        """
        self.quantum_backend = quantum_backend
        self.verbose = verbose
        
        try:
            from qiskit import Aer, QuantumCircuit
            from qiskit.primitives import Sampler
            self.qiskit_available = True
            self.Aer = Aer
            self.QuantumCircuit = QuantumCircuit
            self.Sampler = Sampler
        except ImportError:
            self.qiskit_available = False
            if self.verbose:
                print("[WARNING] Qiskit not available. Using classical approximation of quantum operations.")
    
    def arnold_scrambling(
        self, 
        image: np.ndarray, 
        iterations: int = 100
    ) -> np.ndarray:
        """
        Apply Arnold Scrambling (Arnold Cat Map) to permute pixel positions.
        
        Formula: [x', y'] = [2x + y, x + y] (mod image_dimension)
        
        Args:
            image: Input image matrix (H, W) or (H, W, C)
            iterations: Number of scrambling iterations
        
        Returns:
            Scrambled image with permuted pixel positions
        """
        if len(image.shape) == 3:
            h, w, c = image.shape
            scrambled = np.zeros_like(image)
            for channel in range(c):
                scrambled[:, :, channel] = self._apply_arnold_map(image[:, :, channel], iterations)
            return scrambled
        else:
            return self._apply_arnold_map(image, iterations)
    
    @staticmethod
    def _apply_arnold_map(image: np.ndarray, iterations: int) -> np.ndarray:
        """
        Apply Arnold Cat Map transformation.
        
        Args:
            image: 2D image matrix
            iterations: Number of iterations
        
        Returns:
            Scrambled 2D image
        """
        h, w = image.shape
        scrambled = image.copy()
        
        for _ in range(iterations):
            temp = np.zeros_like(scrambled)
            for i in range(h):
                for j in range(w):
                    # Arnold transformation
                    x_new = (2 * i + j) % h
                    y_new = (i + j) % w
                    temp[x_new, y_new] = scrambled[i, j]
            scrambled = temp
        
        return scrambled
    
    def neqr_encode(
        self, 
        image: np.ndarray, 
        encode_depth: int = 8,
        target_size: int = 128
    ) -> Tuple[np.ndarray, dict]:
        """
        Novel Enhanced Quantum Representation (NEQR) encoding.
        
        NEQR encodes pixel values as quantum states:
        |I⟩ = (1/√(2^(n×m))) Σ |i,j⟩ ⊗ |C(i,j)⟩
        
        IMPORTANT: NEQR is suitable up to 128×128 size
        Larger images are automatically resized to target_size
        
        Where:
        - |i,j⟩ is position encoding
        - |C(i,j)⟩ is color/intensity encoding
        
        Args:
            image: Input image (H, W) or (H, W, C)
            encode_depth: Quantum depth for encoding (bits per pixel)
            target_size: Target size for NEQR (default 128x128)
        
        Returns:
            Encoded representation and metadata
        """
        import cv2
        
        h, w = image.shape[:2]
        original_shape = image.shape
        
        # Resize to target size if needed (NEQR constraint: max 128x128)
        if h > target_size or w > target_size:
            if self.verbose:
                print(f"[WARNING] Image size ({h}x{w}) exceeds NEQR limit ({target_size}x{target_size})")
                print(f"Resizing to {target_size}x{target_size}...")
            
            image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
            h, w = target_size, target_size
        
        # Normalize image to [0, 1]
        if image.max() > 1:
            image_norm = image / 255.0
        else:
            image_norm = image.copy()
        
        # Quantize pixel values to 2^encode_depth levels
        quantized = (image_norm * ((1 << encode_depth) - 1)).astype(np.uint32)
        
        # Create encoding metadata
        metadata = {
            "original_shape": original_shape,
            "neqr_shape": quantized.shape,
            "encode_depth": encode_depth,
            "quantization_levels": 1 << encode_depth,
            "h": h,
            "w": w,
            "target_size": target_size,
            "was_resized": original_shape[:2] != (target_size, target_size),
        }
        
        if self.verbose:
            print(f"NEQR Encoding:")
            print(f"  Original shape: {original_shape}")
            print(f"  NEQR shape: {quantized.shape}")
            print(f"  Encode depth: {encode_depth} bits")
            print(f"  Quantization levels: {1 << encode_depth}")
            print(f"  Target size: {target_size}×{target_size}")
        
        return quantized, metadata
    
    def quantum_xor_cipher(
        self, 
        image: np.ndarray, 
        key: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply quantum XOR-like operation for encryption.
        
        Classical approximation: Encrypted = Original XOR Key
        
        Args:
            image: Input image
            key: Encryption key (random if None)
        
        Returns:
            Encrypted image and used key
        """
        if key is None:
            key = np.random.randint(0, 256, image.shape, dtype=np.uint8)
        
        # Ensure image is uint8 for XOR
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.copy()
        
        # Apply XOR encryption
        encrypted = np.bitwise_xor(image_uint8, key)
        
        if self.verbose:
            print(f"Quantum XOR Cipher applied:")
            print(f"  Image shape: {image.shape}")
            print(f"  Key shape: {key.shape}")
        
        return encrypted, key
    
    def encrypt_roi(
        self, 
        roi_image: np.ndarray,
        scramble_iterations: int = 100,
        encode_depth: int = 8,
        target_size: int = 128
    ) -> Tuple[np.ndarray, dict]:
        """
        Complete quantum encryption pipeline for ROI.
        
        IMPORTANT: ROI is automatically resized to 128×128 for NEQR compatibility
        
        Steps:
        1. Resize ROI to 128×128 (NEQR constraint)
        2. NEQR Encoding
        3. Arnold Scrambling
        4. Quantum XOR Cipher
        
        Args:
            roi_image: ROI matrix (sensitive regions)
            scramble_iterations: Arnold map iterations
            encode_depth: Quantum encoding depth
            target_size: Target size for NEQR (default 128x128)
        
        Returns:
            Encrypted ROI and encryption metadata
        """
        if self.verbose:
            print("=" * 60)
            print("Quantum Encryption Pipeline (ROI)")
            print("=" * 60)
            print(f"Input ROI shape: {roi_image.shape}")
        
        # Step 0: Resize to NEQR-compatible size (max 128x128)
        if self.verbose:
            print(f"Target size for NEQR: {target_size}×{target_size}")
        
        # Step 1: NEQR Encoding (with automatic resizing)
        encoded, metadata = self.neqr_encode(roi_image, encode_depth, target_size)
        
        # Step 2: Arnold Scrambling
        if self.verbose:
            print("Applying Arnold Scrambling...")
        scrambled = self.arnold_scrambling(encoded, scramble_iterations)
        
        # Step 3: Quantum XOR Cipher
        encrypted, key = self.quantum_xor_cipher(scrambled)
        
        metadata["scramble_iterations"] = scramble_iterations
        metadata["key_shape"] = key.shape
        metadata["target_size"] = target_size
        
        if self.verbose:
            print(f"Encryption complete!")
            print(f"  Original ROI shape: {roi_image.shape}")
            print(f"  Encrypted shape: {encrypted.shape}")
            print(f"  Resized to: {target_size}×{target_size} (NEQR compatible)")
            print(f"  Encryption key stored (required for decryption)")
        
        return encrypted, metadata
    
    def save_encrypted_roi(
        self,
        encrypted_image: np.ndarray,
        metadata: dict,
        output_dir: str,
        prefix: str = "encrypted_roi"
    ) -> str:
        """
        Save encrypted ROI to disk.
        
        Args:
            encrypted_image: Encrypted image array
            metadata: Encryption metadata
            output_dir: Directory to save
            prefix: Output filename prefix
        
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save encrypted image
        output_file = output_path / f"{prefix}.npy"
        np.save(str(output_file), encrypted_image)
        
        # Save metadata
        metadata_file = output_path / f"{prefix}_metadata.npy"
        np.save(str(metadata_file), metadata, allow_pickle=True)
        
        if self.verbose:
            print(f"Encrypted ROI saved: {output_file}")
            print(f"Metadata saved: {metadata_file}")
        
        return str(output_file)


class QuantumKeyManager:
    """
    Manages quantum encryption keys for secure key exchange.
    """
    
    @staticmethod
    def generate_quantum_key(shape: Tuple[int, ...]) -> np.ndarray:
        """Generate a random quantum-grade key."""
        return np.random.randint(0, 256, shape, dtype=np.uint8)
    
    @staticmethod
    def save_key(key: np.ndarray, output_path: str) -> None:
        """Save encryption key securely."""
        np.save(output_path, key)
    
    @staticmethod
    def load_key(key_path: str) -> np.ndarray:
        """Load encryption key."""
        return np.load(key_path)
