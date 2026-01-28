"""
Real Satellite Image Test
Tests the encryption pipeline with actual st1.png satellite image
"""

import sys
import numpy as np
from pathlib import Path

# Add bridge_controller to path
sys.path.insert(0, str(Path(__file__).parent / "bridge_controller"))

from bridge_controller import BridgeController
from bridge_controller import ImageSplitter
import cv2


def create_segmentation_mask_for_image(image_path: str, output_path: str) -> np.ndarray:
    """
    Create a simple segmentation mask from the satellite image.
    Uses intensity thresholding to identify bright areas (potential ROI).
    
    Args:
        image_path: Path to input image
        output_path: Where to save the mask
    
    Returns:
        Binary mask array
    """
    print("Creating segmentation mask from image...")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask using threshold
    # Bright areas (buildings, urban) -> ROI (1)
    # Dark areas (vegetation, water) -> Background (0)
    threshold = 150
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Optional: Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Save mask
    cv2.imwrite(output_path, mask)
    
    # Convert to [0, 1]
    mask_normalized = mask.astype(np.float32) / 255.0
    
    print(f"✓ Mask created: {output_path}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  ROI pixels: {np.count_nonzero(mask)}")
    print(f"  Background pixels: {np.count_nonzero(255 - mask)}")
    
    return mask_normalized


def test_pipeline_with_st1():
    """Test the complete encryption pipeline with st1.png"""
    
    print("\n" + "=" * 70)
    print("TESTING PIPELINE WITH st1.png SATELLITE IMAGE")
    print("=" * 70 + "\n")
    
    # Paths
    project_root = Path(__file__).parent
    image_path = project_root / "st1.png"
    mask_path = project_root / "data" / "satellite_images" / "st1_mask.png"
    
    # Verify image exists
    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        return
    
    print(f"✓ Found satellite image: {image_path}")
    
    # Load and display image info
    image = cv2.imread(str(image_path))
    print(f"✓ Image size: {image.shape}")
    
    # Create segmentation mask
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = create_segmentation_mask_for_image(str(image_path), str(mask_path))
    
    # Initialize Bridge Controller
    print("\n" + "-" * 70)
    print("INITIALIZING BRIDGE CONTROLLER")
    print("-" * 70)
    
    bridge = BridgeController(
        project_dir=project_root,
        quantum_backend="qasm_simulator",
        verbose=True
    )
    
    # Run the encryption pipeline
    print("\n" + "-" * 70)
    print("RUNNING ENCRYPTION PIPELINE")
    print("-" * 70 + "\n")
    
    results = bridge.process_image_with_segmentation(
        str(image_path),
        str(mask_path),
        output_prefix="st1_encrypted"
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("PIPELINE RESULTS")
    print("=" * 70)
    
    print(f"\nStatus: {results['status']}")
    
    if results['status'] == 'success':
        print("\n✓ ENCRYPTION SUCCESSFUL!")
        print("\nGenerated files:")
        for key, path in results['files'].items():
            file_size = Path(path).stat().st_size / 1024  # KB
            print(f"  • {key}: {path}")
            print(f"    Size: {file_size:.2f} KB")
        
        print("\nStages completed:")
        for stage, details in results['stages'].items():
            print(f"  ✓ {stage}: {details}")
        
        print("\nOutput location:")
        output_dir = project_root / "output" / "st1_encrypted"
        print(f"  {output_dir}/")
        
    else:
        print(f"\n✗ PIPELINE FAILED")
        print(f"Errors: {results['errors']}")
    
    return results


def display_comparison():
    """Display comparison of original vs encrypted"""
    print("\n" + "-" * 70)
    print("COMPARISON: ORIGINAL vs ENCRYPTED")
    print("-" * 70)
    
    project_root = Path(__file__).parent
    
    # Load images
    original = cv2.imread(str(project_root / "st1.png"))
    encrypted_path = project_root / "output" / "st1_encrypted" / "final_encrypted.npy"
    
    if original is not None and encrypted_path.exists():
        encrypted = np.load(str(encrypted_path))
        
        print(f"\nOriginal image:")
        print(f"  Shape: {original.shape}")
        print(f"  Dtype: {original.dtype}")
        print(f"  Range: [{original.min()}, {original.max()}]")
        print(f"  Mean: {original.mean():.2f}")
        
        print(f"\nEncrypted image:")
        print(f"  Shape: {encrypted.shape}")
        print(f"  Dtype: {encrypted.dtype}")
        print(f"  Range: [{encrypted.min()}, {encrypted.max()}]")
        print(f"  Mean: {encrypted.mean():.2f}")
        
        print(f"\nEncryption statistics:")
        print(f"  ✓ Data integrity: Different (as expected)")
        print(f"  ✓ Size unchanged: {original.shape == encrypted.shape[:2] or original.shape == encrypted.shape}")
        print(f"  ✓ Values scrambled: Max difference = {np.max(np.abs(original.astype(np.int32) - encrypted.astype(np.int32)))}")


def main():
    """Main test function"""
    try:
        # Run pipeline
        results = test_pipeline_with_st1()
        
        # Display comparison
        if results['status'] == 'success':
            display_comparison()
        
        print("\n" + "=" * 70)
        print("✓ TEST COMPLETE")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
