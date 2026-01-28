"""
Test Module for Bridge Controller
Demonstrates the complete encryption pipeline with synthetic data.
"""

import numpy as np
import sys
from pathlib import Path

# Add bridge_controller to path
sys.path.insert(0, str(Path(__file__).parent.parent / "bridge_controller"))

from pipeline import BridgeController
from splitter import ImageSplitter
from quantum_handler import QuantumEncryptionHandler
from classical_handler import ClassicalEncryptionHandler


def generate_synthetic_satellite_image(
    height: int = 512,
    width: int = 512
) -> np.ndarray:
    """
    Generate a synthetic satellite-like image.
    Simulates multi-spectral satellite data.
    """
    # Create synthetic image with different regions
    image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Urban area (bright) - top-left
    image[50:200, 50:200] = [0.8, 0.75, 0.7]
    
    # Water body (dark) - bottom-right
    image[300:450, 300:450] = [0.2, 0.3, 0.5]
    
    # Agricultural area (green) - top-right
    image[50:200, 300:450] = [0.4, 0.6, 0.3]
    
    # Forest (dense) - bottom-left
    image[300:450, 50:200] = [0.2, 0.4, 0.2]
    
    # Add noise for realism
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    return image


def generate_synthetic_segmentation_mask(
    height: int = 512,
    width: int = 512
) -> np.ndarray:
    """
    Generate a synthetic segmentation mask.
    1 = ROI (sensitive: buildings, urban areas)
    0 = Background (non-sensitive: vegetation, water)
    """
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Mark urban areas as ROI (1)
    mask[50:200, 50:200] = 1.0  # Urban area
    mask[100:180, 320:420] = 1.0  # Building region
    
    # Rest is background (0) - water, forest, agriculture
    
    return mask


def test_image_splitting():
    """Test the image splitting functionality."""
    print("\n" + "="*70)
    print("TEST 1: IMAGE SPLITTING")
    print("="*70)
    
    # Generate synthetic data
    image = generate_synthetic_satellite_image()
    mask = generate_synthetic_segmentation_mask()
    
    # Test splitting
    splitter = ImageSplitter(verbose=True)
    roi_image, bg_image = splitter.split_image(image, mask)
    
    # Validate reconstruction
    is_valid = splitter.validate_split(image, roi_image, bg_image, mask)
    
    assert is_valid, "Reconstruction validation failed!"
    print("✓ Image splitting test PASSED\n")
    
    return image, mask, roi_image, bg_image


def test_quantum_encryption():
    """Test quantum encryption of ROI."""
    print("="*70)
    print("TEST 2: QUANTUM ENCRYPTION (ROI)")
    print("="*70)
    
    # Generate ROI
    image = generate_synthetic_satellite_image()
    mask = generate_synthetic_segmentation_mask()
    
    roi_image, _ = ImageSplitter().split_image(image, mask)
    roi_uint8 = (roi_image * 255).astype(np.uint8)
    
    # Test quantum encryption
    quantum_handler = QuantumEncryptionHandler(verbose=True)
    encrypted_roi, metadata = quantum_handler.encrypt_roi(
        roi_uint8,
        scramble_iterations=50,  # Reduced for faster testing
        encode_depth=8
    )
    
    # Verify output
    assert encrypted_roi.shape == roi_uint8.shape, "Output shape mismatch!"
    assert encrypted_roi.dtype == np.uint8, "Output type mismatch!"
    assert metadata is not None, "Metadata not generated!"
    
    print("✓ Quantum encryption test PASSED\n")
    
    return encrypted_roi, metadata


def test_classical_encryption():
    """Test classical encryption of background."""
    print("="*70)
    print("TEST 3: CLASSICAL ENCRYPTION (Background)")
    print("="*70)
    
    # Generate background
    image = generate_synthetic_satellite_image()
    mask = generate_synthetic_segmentation_mask()
    
    _, bg_image = ImageSplitter().split_image(image, mask)
    bg_uint8 = (bg_image * 255).astype(np.uint8)
    
    # Test classical encryption
    classical_handler = ClassicalEncryptionHandler(verbose=True)
    encrypted_bg, key = classical_handler.encrypt_background(bg_uint8)
    
    # Verify output
    assert encrypted_bg.shape == bg_uint8.shape, "Output shape mismatch!"
    assert key.shape == bg_uint8.shape, "Key shape mismatch!"
    
    # Test decryption
    decrypted_bg = classical_handler.decrypt_background(encrypted_bg, key)
    
    # XOR is reversible, should be identical
    assert np.array_equal(decrypted_bg, bg_uint8), "Decryption failed!"
    
    print("✓ Classical encryption test PASSED\n")
    
    return encrypted_bg, key


def test_complete_pipeline():
    """Test the complete end-to-end pipeline."""
    print("="*70)
    print("TEST 4: COMPLETE PIPELINE")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "tests" / "synthetic_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save test data
    image = generate_synthetic_satellite_image()
    mask = generate_synthetic_segmentation_mask()
    
    image_uint8 = (image * 255).astype(np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    import cv2
    image_path = test_dir / "test_image.png"
    mask_path = test_dir / "test_mask.png"
    
    cv2.imwrite(str(image_path), cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask_uint8)
    
    # Run bridge controller
    bridge = BridgeController(project_dir=project_root, verbose=True)
    results = bridge.process_image_with_segmentation(
        str(image_path),
        str(mask_path),
        output_prefix="test_run"
    )
    
    # Verify results
    assert results["status"] == "success", f"Pipeline failed: {results['errors']}"
    assert "final_encrypted" in results["files"], "Final encrypted output missing!"
    
    print("✓ Complete pipeline test PASSED\n")
    
    return results


def run_all_tests():
    """Run all unit and integration tests."""
    print("\n" + "="*70)
    print("BRIDGE CONTROLLER TEST SUITE")
    print("Satellite Image Encryption Pipeline")
    print("="*70)
    
    try:
        # Test 1: Image Splitting
        test_image_splitting()
        
        # Test 2: Quantum Encryption
        test_quantum_encryption()
        
        # Test 3: Classical Encryption
        test_classical_encryption()
        
        # Test 4: Complete Pipeline
        test_complete_pipeline()
        
        print("="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
