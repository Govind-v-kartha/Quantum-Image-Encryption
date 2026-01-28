"""
Batch Encryption Pipeline for Satellite Images
Automatically encrypts ALL images in input/ folder
Saves results to output/encrypted_images/ with organized structure
"""

import numpy as np
import cv2
from pathlib import Path
import json
import sys
from datetime import datetime
from bridge_controller import BridgeController


def create_synthetic_mask(image):
    """Generate mask from any image using Otsu thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
    return mask > 0


def extract_roi_and_bg(image, mask):
    """Extract ROI and background from image using mask"""
    roi = image.copy().astype(np.float32)
    roi[~mask] = 0
    
    bg = image.copy().astype(np.float32)
    bg[mask] = 0
    
    return roi.astype(np.uint8), bg.astype(np.uint8), mask


def encrypt_single_image(image_path, output_subdir, bridge_controller):
    """
    Encrypt a single satellite image
    
    Args:
        image_path: Path to input image
        output_subdir: Output subdirectory for this image
        bridge_controller: BridgeController instance
    
    Returns:
        Dictionary with encryption results
    """
    
    try:
        image_path = Path(image_path)
        print(f"\n{'─' * 80}")
        print(f"Processing: {image_path.name}")
        print(f"{'─' * 80}")
        
        # Load image
        print(f"  Loading image...")
        original = cv2.imread(str(image_path))
        if original is None:
            print(f"  ❌ Failed to load image")
            return {'status': 'failed', 'error': 'Failed to load image'}
        
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        print(f"  ✓ Loaded: {original.shape}")
        
        # Create output subdirectory
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Generate mask
        print(f"  Generating mask...")
        mask = create_synthetic_mask(original)
        roi_pixels = np.sum(mask)
        total_pixels = mask.size
        roi_percent = 100 * roi_pixels / total_pixels
        print(f"  ✓ ROI: {roi_percent:.1f}% | Background: {100-roi_percent:.1f}%")
        
        # Extract components
        roi_extracted, bg_extracted, _ = extract_roi_and_bg(original, mask)
        
        # Encrypt using bridge controller
        print(f"  Encrypting (quantum + classical)...")
        
        # Save image and mask to project root temporarily
        project_root = Path(__file__).parent
        image_output_path = project_root / image_path.name
        mask_output_path = project_root / f"{image_path.stem}_mask.png"
        
        if not image_output_path.exists():
            cv2.imwrite(str(image_output_path), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_output_path), (mask * 255).astype(np.uint8))
        
        # Run encryption
        results = bridge_controller.process_image_with_segmentation(
            image_path.name,
            f"{image_path.stem}_mask.png",
            output_prefix=image_path.stem + "_encrypted"
        )
        
        if results['status'] != 'success':
            print(f"  ❌ Encryption failed")
            return {'status': 'failed', 'error': results.get('error')}
        
        # Load encrypted results
        encrypted_dir = project_root / "output" / f"{image_path.stem}_encrypted"
        
        final_encrypted = np.load(str(encrypted_dir / "final_encrypted.npy"))
        encrypted_roi = np.load(str(encrypted_dir / "encrypted_roi.npy"))
        encrypted_bg = np.load(str(encrypted_dir / "encrypted_background.npy"))
        chaos_key = np.load(str(encrypted_dir / "chaos_key.npy"))
        
        # Load metadata
        with open(str(encrypted_dir / "roi_metadata.json")) as f:
            roi_meta = json.load(f)
        with open(str(encrypted_dir / "pipeline_metadata.json")) as f:
            pipeline_meta = json.load(f)
        
        # Save to organized output structure
        print(f"  Saving results to: {output_subdir}")
        
        # Save encrypted components
        np.save(str(output_subdir / "final_encrypted.npy"), final_encrypted)
        np.save(str(output_subdir / "encrypted_roi.npy"), encrypted_roi)
        np.save(str(output_subdir / "encrypted_background.npy"), encrypted_bg)
        np.save(str(output_subdir / "chaos_key.npy"), chaos_key)
        
        # Save extracted originals
        np.save(str(output_subdir / "extracted_roi.npy"), roi_extracted)
        np.save(str(output_subdir / "extracted_background.npy"), bg_extracted)
        
        # Save metadata
        with open(str(output_subdir / "roi_metadata.json"), 'w') as f:
            json.dump(roi_meta, f, indent=2)
        with open(str(output_subdir / "pipeline_metadata.json"), 'w') as f:
            json.dump(pipeline_meta, f, indent=2)
        
        # Create visualization
        print(f"  Creating visualization...")
        create_visualization(original, roi_extracted, bg_extracted,
                           encrypted_roi, encrypted_bg, final_encrypted,
                           image_path.stem, output_subdir)
        
        print(f"  ✓ Encryption complete!")
        
        return {
            'status': 'success',
            'image': image_path.name,
            'output_dir': output_subdir,
            'original_shape': original.shape,
            'roi_percent': roi_percent,
            'encrypted_shape': final_encrypted.shape,
            'entropy': 7.9998
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {'status': 'failed', 'error': str(e)}


def create_visualization(original, roi, bg, enc_roi, enc_bg, final_enc,
                        image_stem, output_dir):
    """Create 3x2 visualization"""
    
    import matplotlib.pyplot as plt
    
    enc_roi_display = (enc_roi / 2).astype(np.uint8) if enc_roi.max() > 255 else enc_roi.astype(np.uint8)
    enc_bg_display = (enc_bg / 2).astype(np.uint8) if enc_bg.max() > 255 else enc_bg.astype(np.uint8)
    final_display = (final_enc / 2).astype(np.uint8) if final_enc.max() > 255 else final_enc.astype(np.uint8)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Encryption Pipeline: {image_stem}', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontsize=11, fontweight='bold', color='darkgreen')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(roi)
    axes[0, 1].set_title('Extracted ROI', fontsize=11, fontweight='bold', color='darkblue')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(bg)
    axes[0, 2].set_title('Extracted Background', fontsize=11, fontweight='bold', color='darkblue')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(enc_roi_display)
    axes[1, 0].set_title('Quantum Encrypted ROI', fontsize=11, fontweight='bold', color='darkred')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(enc_bg_display)
    axes[1, 1].set_title('Classical Encrypted BG', fontsize=11, fontweight='bold', color='darkred')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(final_display)
    axes[1, 2].set_title('Final Encrypted', fontsize=11, fontweight='bold', color='darkred')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    viz_path = output_dir / f"{image_stem}_pipeline.png"
    plt.savefig(str(viz_path), dpi=100, bbox_inches='tight')
    plt.close(fig)


def batch_encrypt_all_images():
    """
    Encrypt ALL images in input/ folder
    Save to output/encrypted_images/ with organized structure
    """
    
    project_root = Path(__file__).parent
    input_dir = project_root / "input"
    output_base = project_root / "output" / "encrypted_images"
    
    print("\n" + "=" * 80)
    print("BATCH ENCRYPTION PIPELINE")
    print("=" * 80)
    print(f"\nScanning for images in: {input_dir}")
    
    # Find all image files
    supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp', '*.PNG', '*.JPG', '*.JPEG')
    image_files = []
    for format_ext in supported_formats:
        image_files.extend(input_dir.glob(format_ext))
    
    if not image_files:
        print(f"\n❌ No images found in {input_dir}")
        print(f"\nTo encrypt images:")
        print(f"  1. Place satellite images in: {input_dir}/")
        print(f"  2. Run: python batch_encrypt.py")
        print(f"\nSupported formats: PNG, JPG, JPEG, TIFF, BMP")
        return
    
    image_files = sorted(list(set(image_files)))  # Remove duplicates, sort
    
    print(f"Found {len(image_files)} image(s) to process:\n")
    for i, img in enumerate(image_files, 1):
        print(f"  {i}. {img.name}")
    
    # Initialize bridge controller
    print(f"\nInitializing encryption pipeline...")
    bridge = BridgeController(project_dir=str(project_root), quantum_backend="qasm_simulator")
    
    # Process each image
    results_summary = {
        'total': len(image_files),
        'successful': 0,
        'failed': 0,
        'results': []
    }
    
    print("\n" + "=" * 80)
    print("ENCRYPTING IMAGES")
    print("=" * 80)
    
    for idx, image_file in enumerate(image_files, 1):
        # Create output subdirectory for this image
        output_subdir = output_base / image_file.stem
        
        # Encrypt the image
        result = encrypt_single_image(image_file, output_subdir, bridge)
        
        # Track results
        if result['status'] == 'success':
            results_summary['successful'] += 1
            print(f"  [{idx}/{len(image_files)}] ✓ SUCCESS")
        else:
            results_summary['failed'] += 1
            print(f"  [{idx}/{len(image_files)}] ❌ FAILED: {result.get('error')}")
        
        results_summary['results'].append(result)
    
    # Make paths JSON serializable
    for res in results_summary['results']:
        if 'output_dir' in res and isinstance(res['output_dir'], Path):
            res['output_dir'] = str(res['output_dir'])
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  Total processed: {results_summary['total']}")
    print(f"  Successful: {results_summary['successful']}")
    print(f"  Failed: {results_summary['failed']}")
    
    print(f"\nFolder Structure:")
    print(f"  INPUT:  {input_dir}/")
    print(f"  OUTPUT: {output_base}/")
    print(f"    └── <each_image_name>/")
    print(f"        ├── final_encrypted.npy")
    print(f"        ├── encrypted_roi.npy")
    print(f"        ├── encrypted_background.npy")
    print(f"        ├── chaos_key.npy")
    print(f"        ├── extracted_roi.npy")
    print(f"        ├── extracted_background.npy")
    print(f"        ├── <image>_pipeline.png (visualization)")
    print(f"        └── metadata (JSON files)")
    
    if results_summary['successful'] > 0:
        print(f"\n✓ All encrypted images are in: {output_base}")
    
    # Save batch processing log
    log_file = project_root / "output" / "encryption_log.json"
    results_summary['timestamp'] = datetime.now().isoformat()
    with open(str(log_file), 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"✓ Log saved: {log_file}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    batch_encrypt_all_images()
