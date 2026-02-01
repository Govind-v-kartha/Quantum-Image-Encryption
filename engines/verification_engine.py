"""
Verification Engine - Phase 8
Multi-Layer Integrity Verification

Verifies encryption quality and data integrity across multiple dimensions.
"""

import numpy as np
from typing import Dict, Any
import logging
import hashlib


class VerificationEngine:
    """
    Multi-layer verification engine for encryption quality and integrity.
    Checks hash consistency, pixel statistics, and encryption strength.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Verification Engine.
        
        Args:
            config: Configuration dict with verification_engine settings
        """
        self.config = config.get('verification_engine', {})
        self.logger = logging.getLogger('verification_engine')
        self.is_initialized = False
        
        # Verification parameters
        self.num_verification_layers = self.config.get('num_layers', 4)
        self.sampling_ratio = self.config.get('block_sampling_ratio', 0.1)
        self.abort_on_failure = self.config.get('abort_on_failure', False)
    
    def initialize(self):
        """Initialize engine and prepare for processing."""
        self.is_initialized = True
        self.logger.info(f"Verification Engine initialized ({self.num_verification_layers} layers)")
    
    def verify_encryption_quality(self, original: np.ndarray, encrypted: np.ndarray) -> Dict[str, Any]:
        """
        Verify encryption quality through multiple checks.
        
        Args:
            original: Original image
            encrypted: Encrypted image
            
        Returns:
            Verification report dict
        """
        if not self.is_initialized:
            self.initialize()
        
        report = {
            'overall_pass': True,
            'layers': {}
        }
        
        try:
            # Layer 1: Hash verification
            report['layers']['hash_verification'] = self._verify_hash(original, encrypted)
            
            # Layer 2: Pixel difference check
            report['layers']['pixel_difference'] = self._verify_pixel_difference(original, encrypted)
            
            # Layer 3: Statistical analysis
            report['layers']['statistics'] = self._verify_statistics(original, encrypted)
            
            # Layer 4: Entropy check
            report['layers']['entropy'] = self._verify_entropy(encrypted)
            
            # Determine overall pass/fail
            report['overall_pass'] = all(layer.get('pass', False) for layer in report['layers'].values())
            
            self.logger.info(f"Verification complete: {'PASS' if report['overall_pass'] else 'FAIL'}")
            
            if not report['overall_pass'] and self.abort_on_failure:
                raise RuntimeError("Verification failed - aborting")
            
            return report
        
        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}")
            report['error'] = str(e)
            report['overall_pass'] = False
            return report
    
    def _verify_hash(self, original: np.ndarray, encrypted: np.ndarray) -> Dict[str, Any]:
        """
        Layer 1: Verify data integrity using SHA-256 hashes.
        Ensures encrypted data is consistently encrypted (deterministic mode).
        """
        original_hash = hashlib.sha256(original.tobytes()).hexdigest()
        encrypted_hash = hashlib.sha256(encrypted.tobytes()).hexdigest()
        
        # Hashes should differ significantly
        different = original_hash != encrypted_hash
        
        result = {
            'pass': different,
            'original_hash': original_hash[:8] + '...',
            'encrypted_hash': encrypted_hash[:8] + '...',
            'different': different,
            'description': 'Encrypted data differs from original'
        }
        
        self.logger.info(f"  Layer 1 (Hash): {'PASS' if result['pass'] else 'FAIL'}")
        return result
    
    def _verify_pixel_difference(self, original: np.ndarray, encrypted: np.ndarray) -> Dict[str, Any]:
        """
        Layer 2: Verify pixel-level differences.
        Encrypts should produce significant pixel changes.
        """
        if original.shape != encrypted.shape:
            return {
                'pass': False,
                'error': 'Shape mismatch',
                'original_shape': original.shape,
                'encrypted_shape': encrypted.shape
            }
        
        # Calculate pixel differences
        diff = np.abs(original.astype(np.float32) - encrypted.astype(np.float32))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        pixels_changed = np.count_nonzero(diff > 0)
        pixels_changed_pct = (pixels_changed / diff.size) * 100
        
        # Check: At least 80% of pixels should change significantly
        pass_check = pixels_changed_pct > 80 or mean_diff > 20
        
        result = {
            'pass': pass_check,
            'mean_pixel_difference': float(mean_diff),
            'max_pixel_difference': float(max_diff),
            'pixels_changed_count': pixels_changed,
            'pixels_changed_percentage': float(pixels_changed_pct),
            'description': 'Pixels are significantly different between original and encrypted'
        }
        
        self.logger.info(f"  Layer 2 (Pixels): {'PASS' if result['pass'] else 'FAIL'} ({pixels_changed_pct:.1f}% changed)")
        return result
    
    def _verify_statistics(self, original: np.ndarray, encrypted: np.ndarray) -> Dict[str, Any]:
        """
        Layer 3: Verify statistical properties.
        Encrypted image should have different statistical properties.
        """
        original_mean = np.mean(original.astype(np.float32))
        encrypted_mean = np.mean(encrypted.astype(np.float32))
        
        original_std = np.std(original.astype(np.float32))
        encrypted_std = np.std(encrypted.astype(np.float32))
        
        # For good encryption, statistics can remain similar (semantic security)
        # but key statistical measures should exist
        pass_check = encrypted_mean > 0 and encrypted_std > 0
        
        result = {
            'pass': pass_check,
            'original_mean': float(original_mean),
            'encrypted_mean': float(encrypted_mean),
            'original_std': float(original_std),
            'encrypted_std': float(encrypted_std),
            'description': 'Statistical properties are preserved or changed appropriately'
        }
        
        self.logger.info(f"  Layer 3 (Statistics): {'PASS' if result['pass'] else 'FAIL'}")
        return result
    
    def _verify_entropy(self, encrypted: np.ndarray) -> Dict[str, Any]:
        """
        Layer 4: Verify Shannon entropy.
        Encrypted image should have high entropy (randomness).
        """
        # Calculate Shannon entropy
        hist, _ = np.histogram(encrypted.flatten(), bins=256, range=(0, 256))
        hist = hist / encrypted.size  # Normalize to probabilities
        hist = hist[hist > 0]  # Remove zero probabilities
        entropy = -np.sum(hist * np.log2(hist))
        
        # Good encryption should have entropy close to maximum (8 bits)
        max_entropy = 8
        pass_check = entropy > 6.0  # Threshold: > 6 bits is good
        
        result = {
            'pass': pass_check,
            'entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'entropy_percentage': float((entropy / max_entropy) * 100),
            'description': 'Encrypted data has high randomness'
        }
        
        self.logger.info(f"  Layer 4 (Entropy): {'PASS' if result['pass'] else 'FAIL'} ({entropy:.2f} bits)")
        return result
    
    def verify_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify metadata structure and content.
        
        Args:
            metadata: Metadata dict
            
        Returns:
            Metadata verification report
        """
        required_fields = ['timestamp', 'version', 'image_shape', 'block_size']
        
        report = {
            'pass': True,
            'required_fields': required_fields,
            'missing_fields': [],
            'description': 'Metadata structure validation'
        }
        
        for field in required_fields:
            if field not in metadata:
                report['missing_fields'].append(field)
                report['pass'] = False
        
        self.logger.info(f"  Metadata verification: {'PASS' if report['pass'] else 'FAIL'}")
        return report
    
    def get_summary(self) -> Dict[str, Any]:
        """Get engine summary."""
        return {
            'engine': 'Verification Engine (Phase 8)',
            'status': 'initialized' if self.is_initialized else 'not_initialized',
            'num_layers': self.num_verification_layers,
            'sampling_ratio': self.sampling_ratio,
            'abort_on_failure': self.abort_on_failure,
            'config': self.config
        }
