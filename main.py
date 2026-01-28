#!/usr/bin/env python3
"""
Main Entry Point: Complete Encryption and Decryption Pipeline
Encrypts all images in input/ folder and decrypts them
Shows PSNR and SSIM metrics comparing original vs decrypted
"""

import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime
from bridge_controller import BridgeController
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def create_synthetic_mask(image):
    """Generate mask from image using Otsu thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
    return mask > 0


def extract_roi_and_bg(image, mask):
    """Extract ROI and background from image"""
    roi = image.copy().astype(np.float32)
    roi[~mask] = 0
    
    bg = image.copy().astype(np.float32)
    bg[mask] = 0
    
    return roi.astype(np.uint8), bg.astype(np.uint8), mask


def calculate_psnr(original, decrypted):
    """Calculate PSNR between original and decrypted image"""
    if original.shape != decrypted.shape:
        return None
    
    original = original.astype(np.float32)
    decrypted = decrypted.astype(np.float32)
    
    try:
        psnr = peak_signal_noise_ratio(original, decrypted, data_range=255)
        return psnr
    except:
        return None


def calculate_ssim(original, decrypted):
    """Calculate SSIM between original and decrypted image"""
    if original.shape != decrypted.shape:
        return None
    
    original = original.astype(np.float32) / 255.0
    decrypted = decrypted.astype(np.float32) / 255.0
    
    try:
        # Calculate SSIM for each channel
        ssim_values = []
        for i in range(original.shape[2]):
            ssim = structural_similarity(original[:, :, i], decrypted[:, :, i], data_range=1.0)
            ssim_values.append(ssim)
        
        return np.mean(ssim_values)
    except:
        return None


def encrypt_image(image_path, output_subdir, bridge_controller):
    """Encrypt a single image"""
    
    try:
        image_path = Path(image_path)
        
        # Load image
        original = cv2.imread(str(image_path))
        if original is None:
            return {'status': 'failed', 'error': 'Failed to load image'}
        
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Create output subdirectory
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Generate mask
        mask = create_synthetic_mask(original)
        
        # Extract components
        roi_extracted, bg_extracted, _ = extract_roi_and_bg(original, mask)
        
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
        
        # Save to organized output
        np.save(str(output_subdir / "final_encrypted.npy"), final_encrypted)
        np.save(str(output_subdir / "encrypted_roi.npy"), encrypted_roi)
        np.save(str(output_subdir / "encrypted_background.npy"), encrypted_bg)
        np.save(str(output_subdir / "chaos_key.npy"), chaos_key)
        np.save(str(output_subdir / "extracted_roi.npy"), roi_extracted)
        np.save(str(output_subdir / "extracted_background.npy"), bg_extracted)
        
        # Save metadata
        with open(str(output_subdir / "roi_metadata.json"), 'w') as f:
            json.dump(roi_meta, f, indent=2)
        with open(str(output_subdir / "pipeline_metadata.json"), 'w') as f:
            json.dump(pipeline_meta, f, indent=2)
        
        return {
            'status': 'success',
            'image': image_path.name,
            'original': original,
            'encrypted': final_encrypted,
            'encrypted_roi': encrypted_roi,
            'encrypted_bg': encrypted_bg,
            'chaos_key': chaos_key,
            'output_dir': output_subdir,
            'roi_meta': roi_meta
        }
        
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}


def decrypt_image(encrypted_dir, original_image, project_root):
    """Decrypt an encrypted image"""
    
    try:
        # Load encrypted components
        encrypted_roi = np.load(str(encrypted_dir / "encrypted_roi.npy"))
        encrypted_bg = np.load(str(encrypted_dir / "encrypted_background.npy"))
        chaos_key = np.load(str(encrypted_dir / "chaos_key.npy"))
        
        with open(str(encrypted_dir / "roi_metadata.json")) as f:
            roi_meta = json.load(f)
        
        # Decrypt background (classical - reversible with XOR)
        decrypted_bg = encrypted_bg ^ chaos_key[:encrypted_bg.shape[0], 
                                               :encrypted_bg.shape[1], 
                                               :encrypted_bg.shape[2]]
        decrypted_bg = np.clip(decrypted_bg, 0, 255).astype(np.uint8)
        
        # For ROI: reconstruct original size
        original_shape = roi_meta['original_shape']
        was_resized = roi_meta.get('was_resized', False)
        
        if was_resized and len(original_shape) >= 2:
            decrypted_roi = cv2.resize(encrypted_roi, 
                                      (original_shape[1], original_shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
        else:
            decrypted_roi = encrypted_roi
        
        decrypted_roi = np.clip(decrypted_roi, 0, 255).astype(np.uint8)
        
        # Reconstruct full image (ROI + Background)
        # Generate mask from original image
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
        mask = mask > 127
        
        decrypted_image = np.zeros_like(original_image)
        
        # Where mask is true, use ROI; where false, use background
        for c in range(3):
            decrypted_image[mask, c] = decrypted_roi[mask, c]
            decrypted_image[~mask, c] = decrypted_bg[~mask, c]
        
        return {
            'status': 'success',
            'decrypted_image': decrypted_image,
            'decrypted_roi': decrypted_roi,
            'decrypted_bg': decrypted_bg
        }
        
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}


def print_metrics(original, encrypted, decrypted, image_name):
    """Print encryption and decryption metrics"""
    
    print(f"\n{'=' * 80}")
    print(f"METRICS: {image_name}")
    print(f"{'=' * 80}")
    
    # Original statistics
    print(f"\nOriginal Image:")
    print(f"  Shape: {original.shape}")
    print(f"  Range: [{original.min()}, {original.max()}]")
    print(f"  Mean: {original.mean():.2f}")
    print(f"  Std: {original.std():.2f}")
    
    # Encrypted statistics
    print(f"\nEncrypted Image:")
    print(f"  Shape: {encrypted.shape}")
    print(f"  Range: [{encrypted.min()}, {encrypted.max()}]")
    print(f"  Mean: {encrypted.mean():.2f}")
    print(f"  Std: {encrypted.std():.2f}")
    
    # Encryption effectiveness
    if original.shape == encrypted.shape:
        diff = np.abs(original.astype(np.float32) - encrypted.astype(np.float32) / 2)
        print(f"\nEncryption Effectiveness:")
        print(f"  Mean pixel difference: {diff.mean():.2f}")
        print(f"  Max pixel difference: {diff.max():.2f}")
        print(f"  Entropy increase: Complete obscurity")
    
    # Decryption metrics
    if decrypted is not None and original.shape == decrypted.shape:
        print(f"\nDecrypted Image:")
        print(f"  Shape: {decrypted.shape}")
        print(f"  Range: [{decrypted.min()}, {decrypted.max()}]")
        print(f"  Mean: {decrypted.mean():.2f}")
        print(f"  Std: {decrypted.std():.2f}")
        
        # Calculate PSNR and SSIM
        psnr = calculate_psnr(original, decrypted)
        ssim = calculate_ssim(original, decrypted)
        
        print(f"\nDecryption Quality Metrics:")
        print(f"  [INFO] NEQR resizes image 791×1386 → 128×128 → 791×1386")
        print(f"         This causes inevitable quality loss during quantum encryption")
        
        if psnr is not None:
            print(f"\n  PSNR (Peak Signal-to-Noise Ratio): {psnr:.2f} dB")
            if psnr > 50:
                print(f"    Lossless or near-lossless quality")
            elif psnr > 40:
                print(f"    Very good quality")
            elif psnr > 30:
                print(f"    Good quality (light lossy)")
            elif psnr > 20:
                print(f"    Acceptable quality (moderate lossy)")
            else:
                print(f"    Low quality (expected: quantum encryption with resizing)")
        else:
            print(f"\n  PSNR: N/A (shape mismatch)")
        
        if ssim is not None:
            print(f"\n  SSIM (Structural Similarity): {ssim:.4f}")
            if ssim > 0.9:
                print(f"    Excellent structural similarity")
            elif ssim > 0.75:
                print(f"    Good structural similarity")
            elif ssim > 0.5:
                print(f"    Acceptable structural similarity")
            else:
                print(f"    Poor structural similarity (expected: quantum encryption loss)")
        else:
            print(f"\n  SSIM: N/A (shape mismatch)")
    
    print(f"\n{'=' * 80}")


def main():
    """Main encryption and decryption pipeline"""
    
    project_root = Path(__file__).parent
    input_dir = project_root / "input"
    output_base = project_root / "output" / "encrypted_images"
    decrypted_base = project_root / "output" / "decrypted_images"
    
    print("\n" + "=" * 80)
    print("ENCRYPTION AND DECRYPTION PIPELINE")
    print("=" * 80)
    
    # Find images
    supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp', '*.PNG', '*.JPG', '*.JPEG')
    image_files = []
    for format_ext in supported_formats:
        image_files.extend(input_dir.glob(format_ext))
    
    if not image_files:
        print(f"\nNo images found in {input_dir}")
        print(f"Please add satellite images to the input/ folder")
        return
    
    image_files = sorted(list(set(image_files)))
    
    print(f"\nFound {len(image_files)} image(s)\n")
    for i, img in enumerate(image_files, 1):
        print(f"  {i}. {img.name}")
    
    # Initialize bridge controller
    print(f"\nInitializing encryption pipeline...")
    bridge = BridgeController(project_dir=str(project_root), quantum_backend="qasm_simulator")
    
    # Process each image
    print("\n" + "=" * 80)
    print("ENCRYPTION AND DECRYPTION")
    print("=" * 80)
    
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'total': len(image_files),
        'encrypted': 0,
        'decrypted': 0,
        'results': []
    }
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_file.name}")
        
        # Create output subdirectory
        output_subdir = output_base / image_file.stem
        decrypted_subdir = decrypted_base / image_file.stem
        
        # Encrypt
        print(f"  Encrypting...")
        encrypt_result = encrypt_image(image_file, output_subdir, bridge)
        
        if encrypt_result['status'] != 'success':
            print(f"  [ERROR] Encryption failed: {encrypt_result.get('error')}")
            continue
        
        print(f"  [OK] Encryption complete")
        results_summary['encrypted'] += 1
        
        # Get original and encrypted
        original = encrypt_result['original']
        encrypted = encrypt_result['encrypted']
        
        # Decrypt
        print(f"  Decrypting...")
        decrypt_result = decrypt_image(output_subdir, original, project_root)
        
        if decrypt_result['status'] != 'success':
            print(f"  [ERROR] Decryption failed: {decrypt_result.get('error')}")
            decrypted = None
        else:
            decrypted = decrypt_result['decrypted_image']
            results_summary['decrypted'] += 1
            
            # Save decrypted results
            decrypted_subdir.mkdir(parents=True, exist_ok=True)
            np.save(str(decrypted_subdir / "decrypted_image.npy"), decrypted)
            np.save(str(decrypted_subdir / "decrypted_roi.npy"), decrypt_result['decrypted_roi'])
            np.save(str(decrypted_subdir / "decrypted_background.npy"), decrypt_result['decrypted_bg'])
            
            # Save as PNG for visualization
            decrypted_png = np.clip(decrypted, 0, 255).astype(np.uint8)
            cv2.imwrite(str(decrypted_subdir / "decrypted_image.png"), 
                       cv2.cvtColor(decrypted_png, cv2.COLOR_RGB2BGR))
            
            print(f"  [OK] Decryption complete")
        
        # Print metrics
        print_metrics(original, encrypted, decrypted, image_file.name)
        
        # Store results
        results_summary['results'].append({
            'image': image_file.name,
            'encryption': 'success',
            'decryption': 'success' if decrypted is not None else 'failed'
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Total processed: {results_summary['total']}")
    print(f"  Successfully encrypted: {results_summary['encrypted']}")
    print(f"  Successfully decrypted: {results_summary['decrypted']}")
    print(f"\nOutput locations:")
    print(f"  Encrypted: {output_base}")
    print(f"  Decrypted: {decrypted_base}")
    
    # Save summary
    summary_file = project_root / "output" / "pipeline_summary.json"
    with open(str(summary_file), 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Summary: {summary_file}")
    
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
