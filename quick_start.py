"""
Quick Start Example
Demonstrates complete encryption pipeline with minimal setup.
"""

import numpy as np
import sys
from pathlib import Path

# Add bridge_controller to path
sys.path.insert(0, str(Path(__file__).parent / "bridge_controller"))

from bridge_controller import BridgeController, ImageSplitter
from bridge_controller import QuantumEncryptionHandler, ClassicalEncryptionHandler

import cv2


def create_sample_satellite_image(output_path: str) -> None:
    """
    Create a realistic synthetic satellite image.
    
    Simulates Sentinel-2 satellite data with:
    - Urban areas (buildings)
    - Agricultural areas (crops)
    - Water bodies
    - Forest/vegetation
    """
    print("Creating sample satellite image...")
    
    height, width = 512, 512
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Urban area (RGB: 180, 160, 140) - top-left
    image[50:200, 50:200] = [140, 160, 180]  # BGR format
    
    # Agricultural area (RGB: 100, 150, 80) - top-right
    image[50:200, 300:450] = [80, 150, 100]
    
    # Water body (RGB: 50, 80, 150) - bottom-right
    image[300:450, 300:450] = [150, 80, 50]
    
    # Forest (RGB: 40, 100, 40) - bottom-left
    image[300:450, 50:200] = [40, 100, 40]
    
    # Mixed urban/suburban area
    image[100:180, 320:420] = [130, 140, 160]
    
    # Add Gaussian noise for realism
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Save
    cv2.imwrite(output_path, image)
    print(f"✓ Saved to: {output_path}")


def create_sample_segmentation_mask(output_path: str) -> None:
    """
    Create segmentation mask for sample image.
    
    ROI (1) = Buildings, urban infrastructure
    Background (0) = Vegetation, water, non-critical areas
    """
    print("Creating sample segmentation mask...")
    
    height, width = 512, 512
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Mark urban areas as ROI (255 = high importance)
    mask[50:200, 50:200] = 255  # Main urban area
    mask[100:180, 320:420] = 255  # Building cluster
    
    # Rest is background (0)
    
    # Save
    cv2.imwrite(output_path, mask)
    print(f"✓ Saved to: {output_path}")


def example_basic_encryption():
    """Example 1: Basic encryption pipeline."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: BASIC ENCRYPTION PIPELINE")
    print("=" * 70 + "\n")
    
    # Setup paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "satellite_images"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = data_dir / "sample_image.png"
    mask_path = data_dir / "sample_mask.png"
    
    # Create sample data
    if not image_path.exists():
        create_sample_satellite_image(str(image_path))
    if not mask_path.exists():
        create_sample_segmentation_mask(str(mask_path))
    
    # Run encryption pipeline
    bridge = BridgeController(project_dir=project_root, verbose=True)
    results = bridge.process_image_with_segmentation(
        str(image_path),
        str(mask_path),
        output_prefix="example1_basic"
    )
    
    # Print results
    if results["status"] == "success":
        print("\n" + "─" * 70)
        print("RESULTS SUMMARY")
        print("─" * 70)
        print(f"Status: {results['status']}")
        print(f"Final encrypted image: {results['files']['final_encrypted']}")
        print(f"Chaos key: {results['files']['chaos_key']}")
        print(f"ROI metadata: {results['files']['roi_metadata']}")
    else:
        print(f"Failed: {results['errors']}")


def example_custom_parameters():
    """Example 2: Encryption with custom parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: CUSTOM PARAMETERS")
    print("=" * 70 + "\n")
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "satellite_images"
    
    image_path = data_dir / "sample_image.png"
    mask_path = data_dir / "sample_mask.png"
    
    # Ensure sample data exists
    if not image_path.exists():
        create_sample_satellite_image(str(image_path))
    if not mask_path.exists():
        create_sample_segmentation_mask(str(mask_path))
    
    # Load data
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Split with custom parameters
    splitter = ImageSplitter(verbose=True)
    roi_image, bg_image = splitter.split_image(image, mask)
    
    print(f"\nROI matrix shape: {roi_image.shape}")
    print(f"Background matrix shape: {bg_image.shape}")
    
    # Encrypt with custom parameters
    print("\n" + "─" * 70)
    print("QUANTUM ENCRYPTION (ROI)")
    print("─" * 70)
    
    roi_uint8 = (roi_image * 255).astype(np.uint8)
    quantum = QuantumEncryptionHandler(verbose=True)
    encrypted_roi, roi_meta = quantum.encrypt_roi(
        roi_uint8,
        scramble_iterations=50,  # Fewer iterations for speed
        encode_depth=8
    )
    
    print(f"Encrypted ROI shape: {encrypted_roi.shape}")
    
    # Encrypt background with custom chaos parameters
    print("\n" + "─" * 70)
    print("CLASSICAL ENCRYPTION (Background)")
    print("─" * 70)
    
    bg_uint8 = (bg_image * 255).astype(np.uint8)
    classical = ClassicalEncryptionHandler(verbose=True)
    encrypted_bg, chaos_key = classical.encrypt_background(
        bg_uint8,
        seed_x=0.25,  # Custom seed
        seed_y=0.75,  # Custom seed
        r=3.99
    )
    
    print(f"Encrypted background shape: {encrypted_bg.shape}")


def example_batch_processing():
    """Example 3: Batch processing multiple images."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: BATCH PROCESSING")
    print("=" * 70 + "\n")
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "satellite_images"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple sample images
    num_images = 3
    image_pairs = []
    
    for i in range(num_images):
        image_path = data_dir / f"batch_image_{i}.png"
        mask_path = data_dir / f"batch_mask_{i}.png"
        
        if not image_path.exists():
            # Create variations of sample images
            create_sample_satellite_image(str(image_path))
        if not mask_path.exists():
            create_sample_segmentation_mask(str(mask_path))
        
        image_pairs.append((str(image_path), str(mask_path)))
    
    # Process all images
    bridge = BridgeController(project_dir=project_root, verbose=False)
    
    results_list = []
    for i, (image_path, mask_path) in enumerate(image_pairs):
        print(f"\nProcessing image {i+1}/{num_images}...")
        
        results = bridge.process_image_with_segmentation(
            image_path,
            mask_path,
            output_prefix=f"batch_{i}"
        )
        
        results_list.append(results)
        
        if results["status"] == "success":
            print(f"✓ Image {i+1} encrypted successfully")
        else:
            print(f"✗ Image {i+1} failed: {results['errors']}")
    
    # Summary
    print("\n" + "─" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("─" * 70)
    
    successful = sum(1 for r in results_list if r["status"] == "success")
    print(f"Total images: {len(results_list)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results_list) - successful}")


def main():
    """Run all examples."""
    print("\n" + "█" * 70)
    print("█ SECURE IMAGE ENCRYPTION - QUICK START EXAMPLES")
    print("█" * 70 + "\n")
    
    try:
        # Example 1: Basic pipeline
        example_basic_encryption()
        
        # Example 2: Custom parameters
        example_custom_parameters()
        
        # Example 3: Batch processing
        example_batch_processing()
        
        print("\n" + "█" * 70)
        print("█ ✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("█" * 70 + "\n")
        
        print("Next steps:")
        print("1. Check encrypted images in: output/")
        print("2. Read full documentation in: docs/")
        print("3. Run tests with: python tests/test_pipeline.py")
        print("4. Integrate your own satellite imagery")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
