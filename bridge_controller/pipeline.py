"""
Bridge Controller - Main Integration Pipeline
Orchestrates data flow between FlexiMo (AI Segmentation) and 
Quantum-Image-Encryption (Hybrid Encryption).

Pipeline Flow:
Step 1: Intelligence & Segmentation (FlexiMo) → Binary Mask
Step 2: Logic Splitting (Bridge) → ROI & Background matrices
Step 3: Hybrid Encryption → Quantum (ROI) + Classical (Background)
Step 4: Data Fusion & Storage → Single encrypted image
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime

# Import bridge controller modules
from .splitter import ImageSplitter, load_image, load_mask
from .quantum_handler import QuantumEncryptionHandler, QuantumKeyManager
from .classical_handler import ClassicalEncryptionHandler


class BridgeController:
    """
    Main orchestrator for the secure image encryption pipeline.
    Coordinates segmentation, splitting, and hybrid encryption.
    """
    
    def __init__(
        self,
        project_dir: str = ".",
        quantum_backend: str = "qasm_simulator",
        verbose: bool = True
    ):
        """
        Initialize the Bridge Controller.
        
        Args:
            project_dir: Root project directory
            quantum_backend: Qiskit backend selection
            verbose: Enable detailed logging
        """
        self.project_dir = Path(project_dir)
        self.verbose = verbose
        
        # Initialize handlers
        self.splitter = ImageSplitter(verbose=verbose)
        self.quantum_handler = QuantumEncryptionHandler(
            quantum_backend=quantum_backend,
            verbose=verbose
        )
        self.classical_handler = ClassicalEncryptionHandler(verbose=verbose)
        
        # Create output directory
        self.output_dir = self.project_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline metadata
        self.pipeline_metadata = {
            "timestamp": datetime.now().isoformat(),
            "components": ["FlexiMo", "Quantum-Image-Encryption"],
            "stages": []
        }
        
        if self.verbose:
            self._print_banner()
    
    def _print_banner(self):
        """Print welcome banner."""
        print("\n" + "=" * 70)
        print("SECURE IMAGE ENCRYPTION PIPELINE")
        print("Intelligence + Quantum Computing for Satellite Security")
        print("=" * 70 + "\n")
    
    def process_image_with_segmentation(
        self,
        image_path: str,
        mask_path: str,
        output_prefix: str = "encrypted"
    ) -> Dict:
        """
        Complete end-to-end pipeline: Segment → Split → Encrypt.
        
        Args:
            image_path: Path to input satellite image
            mask_path: Path to segmentation mask (binary)
            output_prefix: Prefix for output files
        
        Returns:
            Pipeline results dictionary
        """
        results = {
            "status": "processing",
            "stages": {},
            "files": {},
            "errors": []
        }
        
        try:
            # Stage 1: Load image and mask
            if self.verbose:
                print("\n" + "─" * 70)
                print("STAGE 1: LOAD IMAGE & MASK")
                print("─" * 70)
            
            image, image_dtype = load_image(image_path, normalize=True)
            mask = load_mask(mask_path)
            
            results["stages"]["load"] = {
                "image_shape": image.shape,
                "mask_shape": mask.shape,
                "image_dtype": str(image_dtype)
            }
            
            if self.verbose:
                print(f"✓ Image loaded: {image.shape}")
                print(f"✓ Mask loaded: {mask.shape}")
            
            # Stage 2: Image Splitting
            if self.verbose:
                print("\n" + "─" * 70)
                print("STAGE 2: LOGIC SPLITTING")
                print("─" * 70)
            
            roi_image, bg_image = self.splitter.split_image(image, mask)
            
            # Validate reconstruction
            is_valid = self.splitter.validate_split(image, roi_image, bg_image, mask)
            
            results["stages"]["splitting"] = {
                "roi_shape": roi_image.shape,
                "background_shape": bg_image.shape,
                "reconstruction_valid": is_valid
            }
            
            if not is_valid:
                results["errors"].append("Image reconstruction validation failed")
            
            # Stage 3: Quantum Encryption (ROI)
            if self.verbose:
                print("\n" + "─" * 70)
                print("STAGE 3: QUANTUM ENCRYPTION (ROI)")
                print("─" * 70)
            
            # Convert to uint8 for encryption
            roi_uint8 = (roi_image * 255).astype(np.uint8)
            encrypted_roi, roi_metadata = self.quantum_handler.encrypt_roi(
                roi_uint8,
                scramble_iterations=100,
                encode_depth=8
            )
            
            results["stages"]["quantum_encryption"] = {
                "roi_encrypted_shape": encrypted_roi.shape,
                "scramble_iterations": 100,
                "encode_depth": 8
            }
            
            # Stage 4: Classical Encryption (Background)
            if self.verbose:
                print("\n" + "─" * 70)
                print("STAGE 4: CLASSICAL ENCRYPTION (Background)")
                print("─" * 70)
            
            bg_uint8 = (bg_image * 255).astype(np.uint8)
            encrypted_bg, chaos_key = self.classical_handler.encrypt_background(
                bg_uint8
            )
            
            results["stages"]["classical_encryption"] = {
                "bg_encrypted_shape": encrypted_bg.shape,
                "chaos_key_shape": chaos_key.shape
            }
            
            # Stage 5: Data Fusion
            if self.verbose:
                print("\n" + "─" * 70)
                print("STAGE 5: DATA FUSION & SUPERPOSITION")
                print("─" * 70)
            
            # Handle size difference due to NEQR resizing
            # encrypted_roi is 128x128 (NEQR compatible)
            # encrypted_bg is original size (791x1386)
            # Solution: Resize encrypted_roi back to original size for fusion
            
            import cv2
            
            if encrypted_roi.shape != encrypted_bg.shape:
                if self.verbose:
                    print(f"⚠️  Size mismatch detected:")
                    print(f"  Encrypted ROI: {encrypted_roi.shape} (NEQR resized)")
                    print(f"  Encrypted BG: {encrypted_bg.shape} (original)")
                    print(f"  Resizing ROI back to original size...")
                
                # Resize encrypted ROI back to match background
                original_h, original_w = encrypted_bg.shape[:2]
                encrypted_roi_resized = cv2.resize(
                    encrypted_roi, 
                    (original_w, original_h), 
                    interpolation=cv2.INTER_LINEAR
                )
                encrypted_roi = encrypted_roi_resized
            
            # Superimpose encrypted matrices
            final_encrypted = encrypted_roi.astype(np.uint16) + encrypted_bg.astype(np.uint16)
            final_encrypted = np.clip(final_encrypted, 0, 65535).astype(np.uint16)
            
            if self.verbose:
                print(f"✓ Encrypted matrices fused")
                print(f"  Final shape: {final_encrypted.shape}")
                print(f"  Value range: [{final_encrypted.min()}, {final_encrypted.max()}]")
            
            results["stages"]["fusion"] = {
                "final_encrypted_shape": final_encrypted.shape,
                "roi_encrypted_shape": encrypted_roi.shape,
                "bg_encrypted_shape": encrypted_bg.shape,
                "value_range": [int(final_encrypted.min()), int(final_encrypted.max())]
            }
            
            # Stage 6: Save Results
            if self.verbose:
                print("\n" + "─" * 70)
                print("STAGE 6: SAVE RESULTS")
                print("─" * 70)
            
            output_files = self._save_pipeline_results(
                final_encrypted,
                encrypted_roi,
                encrypted_bg,
                roi_metadata,
                chaos_key,
                output_prefix
            )
            
            results["files"] = output_files
            results["status"] = "success"
            
            # Save pipeline metadata
            self._save_pipeline_metadata(results, output_prefix)
            
            if self.verbose:
                print("\n" + "=" * 70)
                print("✓ PIPELINE COMPLETE - All stages successful!")
                print("=" * 70 + "\n")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(str(e))
            if self.verbose:
                print(f"\n✗ Pipeline failed: {e}")
        
        return results
    
    def _save_pipeline_results(
        self,
        final_encrypted: np.ndarray,
        encrypted_roi: np.ndarray,
        encrypted_bg: np.ndarray,
        roi_metadata: Dict,
        chaos_key: np.ndarray,
        prefix: str
    ) -> Dict:
        """Save all pipeline outputs."""
        output_dir = self.output_dir / prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Save final encrypted image
        final_path = output_dir / "final_encrypted.npy"
        np.save(str(final_path), final_encrypted)
        files["final_encrypted"] = str(final_path)
        if self.verbose:
            print(f"✓ Final encrypted image: {final_path}")
        
        # Save encrypted ROI
        roi_path = output_dir / "encrypted_roi.npy"
        np.save(str(roi_path), encrypted_roi)
        files["encrypted_roi"] = str(roi_path)
        if self.verbose:
            print(f"✓ Encrypted ROI: {roi_path}")
        
        # Save encrypted background
        bg_path = output_dir / "encrypted_background.npy"
        np.save(str(bg_path), encrypted_bg)
        files["encrypted_background"] = str(bg_path)
        if self.verbose:
            print(f"✓ Encrypted background: {bg_path}")
        
        # Save chaos key
        key_path = output_dir / "chaos_key.npy"
        np.save(str(key_path), chaos_key)
        files["chaos_key"] = str(key_path)
        if self.verbose:
            print(f"✓ Chaos key: {key_path}")
        
        # Save ROI metadata
        roi_meta_path = output_dir / "roi_metadata.json"
        with open(roi_meta_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            metadata_serializable = {
                k: (int(v) if isinstance(v, (np.integer, np.uint32)) else tuple(v) if isinstance(v, (tuple, list)) and v else v)
                for k, v in roi_metadata.items()
            }
            json.dump(metadata_serializable, f, indent=2)
        files["roi_metadata"] = str(roi_meta_path)
        if self.verbose:
            print(f"✓ ROI metadata: {roi_meta_path}")
        
        return files
    
    def _save_pipeline_metadata(self, results: Dict, prefix: str) -> str:
        """Save overall pipeline execution metadata."""
        meta_path = self.output_dir / prefix / "pipeline_metadata.json"
        
        # Make results JSON serializable
        serializable_results = self._make_serializable(results)
        
        with open(meta_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        if self.verbose:
            print(f"✓ Pipeline metadata: {meta_path}")
        
        return str(meta_path)
    
    @staticmethod
    def _make_serializable(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: BridgeController._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [BridgeController._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return obj
        return obj
    
    def decrypt_image(
        self,
        encrypted_roi_path: str,
        encrypted_bg_path: str,
        chaos_key_path: str,
        roi_metadata_path: str,
        output_path: str
    ) -> np.ndarray:
        """
        Decrypt encrypted image (decryption pipeline).
        
        Args:
            encrypted_roi_path: Path to encrypted ROI
            encrypted_bg_path: Path to encrypted background
            chaos_key_path: Path to chaos key
            roi_metadata_path: Path to ROI metadata
            output_path: Path to save decrypted image
        
        Returns:
            Decrypted image array
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("DECRYPTION PIPELINE")
            print("=" * 70)
        
        # Load encrypted components
        encrypted_roi = np.load(encrypted_roi_path)
        encrypted_bg = np.load(encrypted_bg_path)
        chaos_key = np.load(chaos_key_path)
        
        # Decrypt background using chaos key
        decrypted_bg = self.classical_handler.decrypt_background(
            encrypted_bg, chaos_key
        )
        
        # Note: Quantum decryption requires the quantum key
        # which should be stored separately during encryption
        
        if self.verbose:
            print(f"✓ Decryption complete")
            print(f"  Background decrypted: {decrypted_bg.shape}")
        
        return decrypted_bg


def main():
    """Example usage of the Bridge Controller."""
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Initialize Bridge Controller
    bridge = BridgeController(
        project_dir=project_root,
        quantum_backend="qasm_simulator",
        verbose=True
    )
    
    print("Bridge Controller initialized successfully!")
    print(f"Project root: {project_root}")
    print(f"Output directory: {bridge.output_dir}")


if __name__ == "__main__":
    main()
