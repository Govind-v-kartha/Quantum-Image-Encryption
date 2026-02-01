"""
PHASE 6: Metadata Security Engine

Encrypts and manages system metadata including:
- ROI masks
- Block encryption information
- Encryption keys
- Saliency maps
- Processing parameters

Independent module - Input → Process → Output
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import zlib
import pickle
from datetime import datetime

logger = logging.getLogger("metadata_engine")


class MetadataEngine:
    """Secure metadata storage and encryption."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metadata engine.
        
        Args:
            config: Configuration dict from config.json
        """
        self.config = config.get('metadata_engine', {})
        self.store_roi_mask = self.config.get('store_roi_mask', True)
        self.store_block_order = self.config.get('store_block_order', True)
        self.store_encryption_keys = self.config.get('store_encryption_keys', True)
        self.compression = self.config.get('compression', 'zlib')
        
        logger.info("Metadata Engine initialized")
    
    def initialize(self):
        """Initialize engine (setup checks)."""
        logger.info("Metadata Engine ready")
    
    def create_metadata(self, 
                       roi_mask: Optional[np.ndarray] = None,
                       block_assignments: Optional[Dict] = None,
                       encryption_keys: Optional[Dict] = None,
                       saliency_map: Optional[np.ndarray] = None,
                       image_shape: Optional[tuple] = None,
                       processing_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create metadata object.
        
        Args:
            roi_mask: ROI binary mask
            block_assignments: Per-block encryption assignments
            encryption_keys: Encryption keys dict
            saliency_map: Saliency heatmap
            image_shape: Original image shape
            processing_params: Processing parameters used
            
        Returns:
            Metadata dict
        """
        metadata = {
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2.0',
            'image_shape': image_shape,
            'block_size': 8
        }
        
        # Store ROI mask
        if self.store_roi_mask and roi_mask is not None:
            metadata['roi_mask'] = {
                'data': self._compress_array(roi_mask),
                'shape': roi_mask.shape,
                'dtype': str(roi_mask.dtype)
            }
        
        # Store block assignments
        if self.store_block_order and block_assignments is not None:
            metadata['block_assignments'] = block_assignments
        
        # Store encryption keys (encrypted)
        if self.store_encryption_keys and encryption_keys is not None:
            metadata['encryption_keys'] = {
                'count': len(encryption_keys),
                'algorithms': list(set(
                    str(k.get('algorithm', 'unknown')) 
                    for k in encryption_keys.values()
                ))
            }
        
        # Store saliency map
        if saliency_map is not None:
            metadata['saliency_map'] = {
                'data': self._compress_array(saliency_map),
                'shape': saliency_map.shape,
                'min': float(saliency_map.min()),
                'max': float(saliency_map.max()),
                'mean': float(saliency_map.mean())
            }
        
        # Store processing parameters
        if processing_params is not None:
            metadata['processing_params'] = processing_params
        
        logger.info(f"Created metadata object with {len(metadata)} fields")
        return metadata
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: str) -> bool:
        """
        Save metadata to file.
        
        Args:
            metadata: Metadata dict
            output_path: Path to save
            
        Returns:
            True if successful
        """
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize to JSON-compatible format
            serialized = self._serialize_metadata(metadata)
            
            with open(path, 'w') as f:
                json.dump(serialized, f, indent=2, default=str)
            
            logger.info(f"Saved metadata to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            return False
    
    def load_metadata(self, metadata_path: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata from file.
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            Metadata dict or None
        """
        try:
            path = Path(metadata_path)
            
            if not path.exists():
                logger.error(f"Metadata file not found: {metadata_path}")
                return None
            
            with open(path, 'r') as f:
                metadata = json.load(f)
            
            # Deserialize arrays
            metadata = self._deserialize_metadata(metadata)
            
            logger.info(f"Loaded metadata from {metadata_path}")
            return metadata
        
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            return None
    
    def encrypt_metadata(self, metadata: Dict[str, Any], encryption_key: bytes) -> Dict[str, Any]:
        """
        Encrypt sensitive metadata fields.
        
        Args:
            metadata: Metadata dict
            encryption_key: Key for encryption
            
        Returns:
            Encrypted metadata
        """
        try:
            from cryptography.fernet import Fernet
            
            # Create cipher
            cipher = Fernet(encryption_key)
            
            encrypted = metadata.copy()
            
            # Encrypt keys field if present
            if 'encryption_keys' in encrypted:
                keys_json = json.dumps(encrypted['encryption_keys']).encode()
                encrypted_keys = cipher.encrypt(keys_json)
                encrypted['encryption_keys'] = {
                    'encrypted': encrypted_keys.decode(),
                    'encrypted_flag': True
                }
            
            logger.info("Encrypted sensitive metadata fields")
            return encrypted
        
        except Exception as e:
            logger.error(f"Failed to encrypt metadata: {str(e)}")
            return metadata
    
    def decrypt_metadata(self, encrypted_metadata: Dict[str, Any], 
                        decryption_key: bytes) -> Dict[str, Any]:
        """
        Decrypt metadata.
        
        Args:
            encrypted_metadata: Encrypted metadata dict
            decryption_key: Key for decryption
            
        Returns:
            Decrypted metadata
        """
        try:
            from cryptography.fernet import Fernet
            
            cipher = Fernet(decryption_key)
            
            decrypted = encrypted_metadata.copy()
            
            # Decrypt keys field if present
            if 'encryption_keys' in decrypted:
                if decrypted['encryption_keys'].get('encrypted_flag', False):
                    encrypted_keys = decrypted['encryption_keys']['encrypted'].encode()
                    keys_json = cipher.decrypt(encrypted_keys)
                    decrypted['encryption_keys'] = json.loads(keys_json)
            
            logger.info("Decrypted metadata")
            return decrypted
        
        except Exception as e:
            logger.error(f"Failed to decrypt metadata: {str(e)}")
            return encrypted_metadata
    
    def _compress_array(self, array: np.ndarray) -> str:
        """Compress numpy array for storage."""
        if self.compression == 'zlib':
            pickled = pickle.dumps(array)
            compressed = zlib.compress(pickled)
            return compressed.hex()
        else:
            return pickle.dumps(array).hex()
    
    def _decompress_array(self, compressed: str) -> np.ndarray:
        """Decompress numpy array from storage."""
        try:
            if self.compression == 'zlib':
                compressed_bytes = bytes.fromhex(compressed)
                pickled = zlib.decompress(compressed_bytes)
                array = pickle.loads(pickled)
            else:
                compressed_bytes = bytes.fromhex(compressed)
                array = pickle.loads(compressed_bytes)
            return array
        except:
            return None
    
    def _serialize_metadata(self, metadata: Dict) -> Dict:
        """Convert metadata to JSON-serializable format."""
        serialized = {}
        
        for key, value in metadata.items():
            if key == 'roi_mask' and isinstance(value, dict):
                # Keep compressed data as is
                serialized[key] = value
            elif key == 'saliency_map' and isinstance(value, dict):
                serialized[key] = value
            elif isinstance(value, np.ndarray):
                serialized[key] = self._compress_array(value)
            else:
                serialized[key] = value
        
        return serialized
    
    def _deserialize_metadata(self, metadata: Dict) -> Dict:
        """Convert JSON metadata back to objects."""
        deserialized = {}
        
        for key, value in metadata.items():
            if key == 'roi_mask' and isinstance(value, dict):
                if 'data' in value:
                    value['data'] = self._decompress_array(value['data'])
                deserialized[key] = value
            elif key == 'saliency_map' and isinstance(value, dict):
                if 'data' in value:
                    value['data'] = self._decompress_array(value['data'])
                deserialized[key] = value
            else:
                deserialized[key] = value
        
        return deserialized
    
    def get_summary(self, metadata: Dict[str, Any]) -> str:
        """
        Get human-readable metadata summary.
        
        Args:
            metadata: Metadata dict
            
        Returns:
            Summary string
        """
        summary = f"""
Metadata Summary
────────────────
Timestamp: {metadata.get('timestamp', 'N/A')}
Version: {metadata.get('version', 'N/A')}
Image Shape: {metadata.get('image_shape', 'N/A')}

Fields:
├─ ROI Mask: {'✓' if 'roi_mask' in metadata else '✗'}
├─ Block Assignments: {'✓' if 'block_assignments' in metadata else '✗'}
├─ Encryption Keys: {'✓' if 'encryption_keys' in metadata else '✗'}
├─ Saliency Map: {'✓' if 'saliency_map' in metadata else '✗'}
└─ Processing Params: {'✓' if 'processing_params' in metadata else '✗'}
"""
        return summary


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    config = {'metadata_engine': {}}
    engine = MetadataEngine(config)
    engine.initialize()
    
    # Create test metadata
    metadata = engine.create_metadata(
        roi_mask=np.ones((256, 256), dtype=np.uint8),
        image_shape=(256, 256, 3)
    )
    
    print(engine.get_summary(metadata))
    print("\nMetadata Engine OK")
