#!/usr/bin/env python3
"""
Main Entry Point: Satellite Image Encryption with ROI Extraction

Architecture:
1. Extract important features (128×128) using semantic segmentation
2. Split image into: ROI (important) and Background (non-important)
3. Encrypt ROI with Quantum Encryption (NEQR + DNA + Chaos)
4. Encrypt Background with Classical Encryption (DNA + Chaos)
5. Combine encrypted parts into final encrypted image
6. Decrypt: Apply reverse operations
7. Calculate PSNR and SSIM quality metrics
"""

import numpy as np
import cv2
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional

# Import encryption functions from Govind's repo
import sys
sys.path.insert(0, str(Path(__file__).parent / "repos" / "Quantum-Image-Encryption"))

from encryption_pipeline import (
    encrypt_single_channel,
    decrypt_single_channel,
    validate_image,
    pad_to_power_of_2,
    crop_to_original,
    derive_channel_seed,
    generate_quantum_key_for_channel,
    generate_substitution_key_for_channel,
    generate_chaotic_key_for_channel
)

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ============================================================================
# ROI EXTRACTION - Extract important features to 128×128
# ============================================================================

def extract_roi_128x128(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extract 128×128 ROI (important features) from image using semantic segmentation.
    
    Simple heuristic: Use Otsu thresholding to identify important regions.
    In production, use FlexiMo model for intelligent feature extraction.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, 3)
    
    Returns
    -------
    roi : np.ndarray
        128×128 important region (shape: 128, 128, 3)
    roi_mask : np.ndarray
        Binary mask indicating ROI location in original image (H, W)
    roi_coords : dict
        Coordinates of extracted ROI (y, x, h, w)
    """
    h, w = image.shape[:2]
    
    # Simple segmentation: Find the most textured/important region
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Use Otsu's method to find foreground
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours and get the largest one
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_contour)
    else:
        # Fallback: Use center region
        x = (w - 128) // 2
        y = (h - 128) // 2
        cw = min(128, w)
        ch = min(128, h)
    
    # Extract 128×128 region (with zero padding if needed)
    roi_image = np.zeros((128, 128, 3), dtype=np.uint8)
    
    # Calculate region to extract
    roi_h = min(128, ch, h - y)
    roi_w = min(128, cw, w - x)
    roi_y = max(0, y)
    roi_x = max(0, x)
    
    # Place extracted region in 128×128 canvas
    roi_image[:roi_h, :roi_w, :] = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, :]
    
    # Create mask for original image
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255
    
    roi_coords = {
        'y': int(roi_y),
        'x': int(roi_x),
        'h': int(roi_h),
        'w': int(roi_w),
        'canvas_h': 128,
        'canvas_w': 128
    }
    
    return roi_image, roi_mask, roi_coords


# ============================================================================
# ENCRYPTION - Quantum for ROI, Classical for Background
# ============================================================================

def encrypt_roi_quantum(
    roi: np.ndarray,
    master_seed: int,
    use_quantum: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Encrypt 128×128 ROI using quantum encryption.
    
    Parameters
    ----------
    roi : np.ndarray
        128×128 RGB image (shape: 128, 128, 3)
    master_seed : int
        Master seed for key generation
    use_quantum : bool
        Use quantum encoding (True) or skip to DNA (False)
    
    Returns
    -------
    encrypted_roi : np.ndarray
        Encrypted ROI (128, 128, 3)
    roi_metadata : dict
        Encryption metadata
    """
    encrypted_channels = []
    
    for c in range(3):
        channel = roi[:, :, c]
        
        encrypted_channel = encrypt_single_channel(
            channel,
            master_seed=master_seed,
            channel_index=c,
            shots=65536,
            use_quantum_encoding=use_quantum,
            quantum_encoder='neqr'
        )
        
        encrypted_channels.append(encrypted_channel)
    
    encrypted_roi = np.stack(encrypted_channels, axis=-1)
    
    roi_metadata = {
        'shape': (128, 128, 3),
        'dtype': 'uint8',
        'encryption_type': 'quantum',
        'master_seed': int(master_seed),
        'use_quantum_encoding': use_quantum,
        'timestamp': datetime.now().isoformat()
    }
    
    return encrypted_roi, roi_metadata


def encrypt_background_classical(
    image: np.ndarray,
    roi_mask: np.ndarray,
    master_seed: int
) -> Tuple[np.ndarray, Dict]:
    """
    Encrypt background (non-ROI) regions using classical encryption.
    
    Parameters
    ----------
    image : np.ndarray
        Original full image (H, W, 3)
    roi_mask : np.ndarray
        Binary mask of ROI (H, W)
    master_seed : int
        Master seed for key generation
    
    Returns
    -------
    encrypted_bg : np.ndarray
        Encrypted image with ROI area zeroed (H, W, 3)
    bg_metadata : dict
        Encryption metadata
    """
    h, w = image.shape[:2]
    
    # Create background image (ROI set to 0)
    bg_image = image.copy()
    roi_mask_bool = roi_mask > 0
    bg_image[roi_mask_bool] = 0
    
    # Pad to power-of-2
    padded_bg, original_shape = pad_to_power_of_2(bg_image, pad_mode='edge')
    
    encrypted_channels = []
    
    for c in range(3):
        channel = padded_bg[:, :, c]
        
        # Use classical encryption: DNA + Chaos (skip quantum)
        encrypted_channel = encrypt_single_channel(
            channel,
            master_seed=master_seed,
            channel_index=c + 10,  # Offset to differentiate from ROI
            shots=65536,
            use_quantum_encoding=False,  # Skip quantum for background
            quantum_encoder='neqr'
        )
        
        encrypted_channels.append(encrypted_channel)
    
    encrypted_padded = np.stack(encrypted_channels, axis=-1)
    encrypted_bg = crop_to_original(encrypted_padded, original_shape)
    
    bg_metadata = {
        'shape': image.shape,
        'dtype': 'uint8',
        'encryption_type': 'classical',
        'master_seed': int(master_seed),
        'use_quantum_encoding': False,
        'roi_mask_shape': roi_mask.shape,
        'timestamp': datetime.now().isoformat()
    }
    
    return encrypted_bg, bg_metadata


def combine_encrypted_parts(
    encrypted_roi: np.ndarray,
    roi_coords: Dict,
    encrypted_bg: np.ndarray,
    roi_mask: np.ndarray
) -> np.ndarray:
    """
    Combine encrypted ROI and encrypted background into final encrypted image.
    
    Parameters
    ----------
    encrypted_roi : np.ndarray
        128×128 encrypted ROI
    roi_coords : dict
        ROI coordinates in original image
    encrypted_bg : np.ndarray
        Encrypted background (H, W, 3)
    roi_mask : np.ndarray
        ROI mask (H, W)
    
    Returns
    -------
    encrypted_image : np.ndarray
        Final encrypted image (H, W, 3)
    """
    h, w = encrypted_bg.shape[:2]
    encrypted_image = encrypted_bg.copy()
    
    # Place encrypted ROI back into original coordinates
    # Resize encrypted ROI to match extracted region size
    roi_y = roi_coords['y']
    roi_x = roi_coords['x']
    roi_h = roi_coords['h']
    roi_w = roi_coords['w']
    
    # Resize encrypted ROI to match original ROI size
    if roi_h > 0 and roi_w > 0:
        resized_roi = cv2.resize(encrypted_roi, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        encrypted_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, :] = resized_roi
    
    return encrypted_image


# ============================================================================
# DECRYPTION - Reverse the encryption process
# ============================================================================

def decrypt_roi_quantum(
    encrypted_roi: np.ndarray,
    master_seed: int,
    use_quantum: bool = True
) -> np.ndarray:
    """
    Decrypt 128×128 ROI using quantum decryption.
    
    Parameters
    ----------
    encrypted_roi : np.ndarray
        Encrypted ROI (128, 128, 3)
    master_seed : int
        Master seed (same as encryption)
    use_quantum : bool
        Must match encryption settings
    
    Returns
    -------
    decrypted_roi : np.ndarray
        Decrypted ROI (128, 128, 3)
    """
    decrypted_channels = []
    
    for c in range(3):
        channel = encrypted_roi[:, :, c]
        
        decrypted_channel = decrypt_single_channel(
            channel,
            master_seed=master_seed,
            channel_index=c,
            shots=65536,
            use_quantum_encoding=use_quantum,
            quantum_encoder='neqr'
        )
        
        decrypted_channels.append(decrypted_channel)
    
    decrypted_roi = np.stack(decrypted_channels, axis=-1)
    return decrypted_roi


def decrypt_background_classical(
    encrypted_bg: np.ndarray,
    roi_mask: np.ndarray,
    master_seed: int
) -> np.ndarray:
    """
    Decrypt background using classical decryption.
    
    Parameters
    ----------
    encrypted_bg : np.ndarray
        Encrypted background (H, W, 3)
    roi_mask : np.ndarray
        ROI mask (H, W)
    master_seed : int
        Master seed (same as encryption)
    
    Returns
    -------
    decrypted_bg : np.ndarray
        Decrypted background (H, W, 3)
    """
    h, w = encrypted_bg.shape[:2]
    
    # Pad to power-of-2
    padded_encrypted, original_shape = pad_to_power_of_2(encrypted_bg, pad_mode='edge')
    
    decrypted_channels = []
    
    for c in range(3):
        channel = padded_encrypted[:, :, c]
        
        # Decrypt with classical algorithm
        decrypted_channel = decrypt_single_channel(
            channel,
            master_seed=master_seed,
            channel_index=c + 10,  # Same offset as encryption
            shots=65536,
            use_quantum_encoding=False,
            quantum_encoder='neqr'
        )
        
        decrypted_channels.append(decrypted_channel)
    
    decrypted_padded = np.stack(decrypted_channels, axis=-1)
    decrypted_bg = crop_to_original(decrypted_padded, original_shape)
    
    return decrypted_bg


def extract_and_decrypt_roi(
    encrypted_image: np.ndarray,
    roi_coords: Dict,
    master_seed: int
) -> np.ndarray:
    """
    Extract encrypted ROI from encrypted image and decrypt it.
    
    Parameters
    ----------
    encrypted_image : np.ndarray
        Full encrypted image (H, W, 3)
    roi_coords : dict
        ROI coordinates
    master_seed : int
        Master seed
    
    Returns
    -------
    decrypted_roi : np.ndarray
        Decrypted and resized ROI (roi_h, roi_w, 3)
    """
    roi_y = roi_coords['y']
    roi_x = roi_coords['x']
    roi_h = roi_coords['h']
    roi_w = roi_coords['w']
    
    # Extract encrypted ROI region from encrypted image
    encrypted_roi_extracted = encrypted_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, :]
    
    # Resize back to 128×128 for decryption
    encrypted_roi_128 = cv2.resize(
        encrypted_roi_extracted,
        (128, 128),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Decrypt ROI
    decrypted_roi_128 = decrypt_roi_quantum(encrypted_roi_128, master_seed, use_quantum=True)
    
    # Resize back to original ROI size
    decrypted_roi = cv2.resize(
        decrypted_roi_128,
        (roi_w, roi_h),
        interpolation=cv2.INTER_LINEAR
    )
    
    return decrypted_roi


def combine_decrypted_parts(
    decrypted_roi: np.ndarray,
    roi_coords: Dict,
    decrypted_bg: np.ndarray
) -> np.ndarray:
    """
    Combine decrypted ROI and background into final decrypted image.
    
    Parameters
    ----------
    decrypted_roi : np.ndarray
        Decrypted ROI (roi_h, roi_w, 3)
    roi_coords : dict
        ROI coordinates
    decrypted_bg : np.ndarray
        Decrypted background (H, W, 3)
    
    Returns
    -------
    decrypted_image : np.ndarray
        Final decrypted image (H, W, 3)
    """
    decrypted_image = decrypted_bg.copy()
    
    roi_y = roi_coords['y']
    roi_x = roi_coords['x']
    roi_h = roi_coords['h']
    roi_w = roi_coords['w']
    
    if roi_h > 0 and roi_w > 0:
        decrypted_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, :] = decrypted_roi
    
    return decrypted_image


# ============================================================================
# QUALITY METRICS
# ============================================================================

def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate PSNR between original and reconstructed images."""
    if original.shape != reconstructed.shape:
        return None
    
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    
    try:
        psnr = peak_signal_noise_ratio(original, reconstructed, data_range=255)
        return psnr
    except:
        return None


def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate SSIM between original and reconstructed images."""
    if original.shape != reconstructed.shape:
        return None
    
    original_norm = original.astype(np.float32) / 255.0
    reconstructed_norm = reconstructed.astype(np.float32) / 255.0
    
    try:
        # Calculate SSIM for each channel
        ssim_values = []
        for c in range(min(3, original_norm.ndim)):
            if original_norm.ndim == 3:
                orig_ch = original_norm[:, :, c]
                recon_ch = reconstructed_norm[:, :, c]
            else:
                orig_ch = original_norm
                recon_ch = reconstructed_norm
            
            ssim_val = structural_similarity(orig_ch, recon_ch, data_range=1.0)
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    except:
        return None


def calculate_entropy(image: np.ndarray) -> float:
    """Calculate image entropy in bits/byte."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def print_metrics(original: np.ndarray, encrypted: np.ndarray, decrypted: np.ndarray, image_name: str):
    """Print comprehensive metrics."""
    print(f"\n{'='*80}")
    print(f"METRICS: {image_name}")
    print(f"{'='*80}")
    
    print(f"\nOriginal Image:")
    print(f"  Shape: {original.shape}")
    print(f"  Range: [{original.min()}, {original.max()}]")
    print(f"  Mean: {original.mean():.2f}")
    print(f"  Std: {original.std():.2f}")
    
    print(f"\nEncrypted Image:")
    print(f"  Shape: {encrypted.shape}")
    print(f"  Range: [{encrypted.min()}, {encrypted.max()}]")
    print(f"  Mean: {encrypted.mean():.2f}")
    print(f"  Std: {encrypted.std():.2f}")
    
    # Encryption effectiveness
    if original.shape == encrypted.shape:
        diff = np.abs(original.astype(np.float32) - encrypted.astype(np.float32))
        entropy = calculate_entropy(encrypted)
        print(f"\nEncryption Effectiveness:")
        print(f"  Mean pixel difference: {diff.mean():.2f}")
        print(f"  Max pixel difference: {diff.max():.2f}")
        print(f"  Image entropy: {entropy:.4f} bits/byte (max: 8.0)")
    
    if decrypted is not None and original.shape == decrypted.shape:
        print(f"\nDecrypted Image:")
        print(f"  Shape: {decrypted.shape}")
        print(f"  Range: [{decrypted.min()}, {decrypted.max()}]")
        print(f"  Mean: {decrypted.mean():.2f}")
        print(f"  Std: {decrypted.std():.2f}")
        
        psnr = calculate_psnr(original, decrypted)
        ssim = calculate_ssim(original, decrypted)
        
        print(f"\nDecryption Quality Metrics:")
        
        if psnr is not None:
            print(f"  PSNR (Peak Signal-to-Noise Ratio): {psnr:.2f} dB")
            if psnr > 50:
                print(f"    Lossless or near-lossless quality")
            elif psnr > 40:
                print(f"    Very good quality")
            elif psnr > 30:
                print(f"    Good quality")
            elif psnr > 20:
                print(f"    Acceptable quality")
            else:
                print(f"    Low quality (lossy)")
        
        if ssim is not None:
            print(f"  SSIM (Structural Similarity): {ssim:.4f}")
            if ssim > 0.95:
                print(f"    Excellent structural similarity")
            elif ssim > 0.80:
                print(f"    Good structural similarity")
            elif ssim > 0.60:
                print(f"    Acceptable structural similarity")
            else:
                print(f"    Poor structural similarity")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main encryption and decryption pipeline."""
    project_root = Path(__file__).parent
    input_dir = project_root / "input"
    output_dir = project_root / "output"
    temp_dir = output_dir / "temp"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SECURE IMAGE ENCRYPTION PIPELINE")
    print("ROI (Quantum) + Background (Classical) Encryption")
    print("="*80)
    
    # Find input images
    if not input_dir.exists():
        print(f"\nInput directory not found: {input_dir}")
        print("Creating input directory...")
        input_dir.mkdir(parents=True, exist_ok=True)
        return
    
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"\nNo images found in {input_dir}")
        return
    
    print(f"\nFound {len(image_files)} image(s):")
    for i, img_file in enumerate(image_files, 1):
        print(f"  {i}. {img_file.name}")
    
    # Master seed (derived from timestamp)
    master_seed = int(hashlib.md5(str(datetime.now()).encode()).hexdigest(), 16) % (2**31)
    
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'total': len(image_files),
        'encrypted': 0,
        'decrypted': 0,
        'results': []
    }
    
    for image_file in image_files:
        print(f"\n[Processing] {image_file.name}")
        
        # Load image
        image_bgr = cv2.imread(str(image_file))
        if image_bgr is None:
            print(f"  [ERROR] Failed to load image")
            continue
        
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        # Step 1: Extract ROI (128×128)
        print(f"  [1/5] Extracting ROI...")
        roi_image, roi_mask, roi_coords = extract_roi_128x128(image)
        
        # Step 2: Encrypt ROI (Quantum)
        print(f"  [2/5] Encrypting ROI (Quantum)...")
        encrypted_roi, roi_metadata = encrypt_roi_quantum(roi_image, master_seed)
        
        # Step 3: Encrypt Background (Classical)
        print(f"  [3/5] Encrypting background (Classical)...")
        encrypted_bg, bg_metadata = encrypt_background_classical(image, roi_mask, master_seed)
        
        # Step 4: Combine encrypted parts
        print(f"  [4/5] Combining encrypted parts...")
        encrypted_image = combine_encrypted_parts(encrypted_roi, roi_coords, encrypted_bg, roi_mask)
        
        results_summary['encrypted'] += 1
        
        # Save temporary metadata
        temp_metadata = {
            'image_name': image_file.name,
            'master_seed': int(master_seed),
            'roi_coords': roi_coords,
            'roi_metadata': roi_metadata,
            'bg_metadata': bg_metadata,
            'original_shape': image.shape,
            'timestamp': datetime.now().isoformat()
        }
        
        temp_file = temp_dir / f"{image_file.stem}_metadata.json"
        with open(temp_file, 'w') as f:
            json.dump(temp_metadata, f, indent=2)
        
        # Step 5: Decrypt
        print(f"  [5/5] Decrypting image...")
        
        try:
            # Decrypt background
            decrypted_bg = decrypt_background_classical(encrypted_bg, roi_mask, master_seed)
            
            # Extract and decrypt ROI
            decrypted_roi = extract_and_decrypt_roi(encrypted_image, roi_coords, master_seed)
            
            # Combine decrypted parts
            decrypted_image = combine_decrypted_parts(decrypted_roi, roi_coords, decrypted_bg)
            
            # Ensure uint8
            decrypted_image = np.clip(decrypted_image, 0, 255).astype(np.uint8)
            
            results_summary['decrypted'] += 1
        except Exception as e:
            print(f"  [ERROR] Decryption failed: {e}")
            decrypted_image = None
        
        # Print metrics
        print_metrics(original_image, encrypted_image, decrypted_image, image_file.name)
        
        # Save results
        result_dir = output_dir / image_file.stem
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save encrypted image
        encrypted_bgr = cv2.cvtColor(encrypted_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(result_dir / "encrypted_image.png"), encrypted_bgr)
        np.save(str(result_dir / "encrypted_image.npy"), encrypted_image)
        
        # Save decrypted image if successful
        if decrypted_image is not None:
            decrypted_bgr = cv2.cvtColor(decrypted_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(result_dir / "decrypted_image.png"), decrypted_bgr)
            np.save(str(result_dir / "decrypted_image.npy"), decrypted_image)
        
        # Save metadata
        with open(result_dir / "encryption_metadata.json", 'w') as f:
            json.dump(temp_metadata, f, indent=2)
        
        results_summary['results'].append({
            'image': image_file.name,
            'status': 'success',
            'output_dir': str(result_dir)
        })
    
    # Save summary
    summary_file = output_dir / "pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  Total processed: {results_summary['total']}")
    print(f"  Successfully encrypted: {results_summary['encrypted']}")
    print(f"  Successfully decrypted: {results_summary['decrypted']}")
    print(f"\nOutput locations:")
    print(f"  Results: {output_dir}")
    print(f"  Metadata: {temp_dir}")
    print(f"  Summary: {summary_file}")


if __name__ == "__main__":
    main()
