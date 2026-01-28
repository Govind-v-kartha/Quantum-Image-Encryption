#!/usr/bin/env python3
"""
Dual-Engine Secure Image Encryption Pipeline
Engine A (Intelligence): FlexiMo for ROI detection
Engine B (Security): Quantum encryption (ROI) + Classical encryption (Background)
"""

import numpy as np
import cv2
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
import time
import sys

# Add repos to path
sys.path.insert(0, str(Path(__file__).parent / "repos" / "Quantum-Image-Encryption"))

try:
    from encryption_pipeline import (
        encrypt_single_channel,
        decrypt_single_channel,
        pad_to_power_of_2,
        crop_to_original
    )
except ImportError as e:
    print(f"[ERROR] Failed to import encryption pipeline: {e}")
    sys.exit(1)

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError:
    peak_signal_noise_ratio = None
    structural_similarity = None


# ============================================================================
# STAGE 2: INTELLIGENCE LAYER - ROI Detection
# ============================================================================

def stage2_ai_segmentation(image: np.ndarray) -> np.ndarray:
    """
    Engine A: Intelligence Layer (FlexiMo Bridge)
    Detects sensitive ROI (buildings, military bases, etc.)
    Returns binary segmentation mask: 1=ROI, 0=Background
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Use Canny edge detection as Phase 1 bridge
    # Phase 2 will replace with FlexiMo OFAViT model
    edges = cv2.Canny(gray, 75, 200)
    
    # Morphological closing to connect regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    roi_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return roi_mask  # Binary mask: 255=ROI, 0=Background


# ============================================================================
# STAGE 3: LOGIC CORE - Smart Bridge with Zero-Loss Tiling
# ============================================================================

def stage3_smart_bridge(image: np.ndarray, roi_mask: np.ndarray) -> Tuple[List[np.ndarray], Dict, np.ndarray]:
    """
    Engine A: Smart Bridge
    - Splits image into ROI and Background
    - Implements tiling mode if ROI > 128x128
    - Preserves every pixel (zero loss)
    
    Returns:
    - roi_tiles: List of 128x128 tiles from ROI
    - tile_metadata: Positions and reconstruction info
    - background_image: Full background with ROI masked out
    """
    h, w = image.shape[:2]
    roi_mask_bool = roi_mask > 127
    
    # Find ROI bounding box
    y_indices, x_indices = np.where(roi_mask_bool)
    
    if len(y_indices) == 0:
        # No ROI detected - return single black tile
        return [np.zeros((128, 128, 3), dtype=np.uint8)], {'total_tiles': 1}, image.copy()
    
    roi_y_min = y_indices.min()
    roi_y_max = y_indices.max() + 1
    roi_x_min = x_indices.min()
    roi_x_max = x_indices.max() + 1
    
    roi_height = roi_y_max - roi_y_min
    roi_width = roi_x_max - roi_x_min
    
    # Create background (ROI masked out)
    background_image = image.copy()
    background_image[roi_mask_bool] = 0
    
    # TILING MODE: If ROI > 128x128, slice into 128x128 tiles
    tile_size = 128
    roi_tiles = []
    tile_positions = []
    
    # Slice ROI into tiles
    for tile_y in range(0, roi_height, tile_size):
        for tile_x in range(0, roi_width, tile_size):
            # Calculate tile dimensions (may be smaller at boundaries)
            tile_h = min(tile_size, roi_height - tile_y)
            tile_w = min(tile_size, roi_width - tile_x)
            
            # Original image coordinates
            img_y = roi_y_min + tile_y
            img_x = roi_x_min + tile_x
            
            # Extract tile
            tile = image[img_y:img_y+tile_h, img_x:img_x+tile_w, :].copy()
            
            # Pad to 128x128 if needed (for last row/column tiles)
            if tile.shape != (128, 128, 3):
                tile_padded = np.zeros((128, 128, 3), dtype=np.uint8)
                tile_padded[:tile_h, :tile_w, :] = tile
                tile = tile_padded
            
            roi_tiles.append(tile)
            tile_positions.append({
                'original_y': int(img_y),
                'original_x': int(img_x),
                'tile_height': int(tile_h),
                'tile_width': int(tile_w)
            })
    
    tile_metadata = {
        'total_tiles': len(roi_tiles),
        'tile_size': 128,
        'roi_bbox': (int(roi_y_min), int(roi_x_min), int(roi_height), int(roi_width)),
        'tile_positions': tile_positions,
        'original_shape': image.shape
    }
    
    return roi_tiles, tile_metadata, background_image


# ============================================================================
# STAGE 4: HYBRID ENCRYPTION LAYER
# ============================================================================

def stage4_encrypt_roi_quantum(roi_tiles: List[np.ndarray], master_seed: int) -> Tuple[List[np.ndarray], Dict]:
    """
    Path A: Quantum Encryption (ROI)
    - NEQR: Novel Enhanced Quantum Representation
    - Arnold Scrambling: Cat map permutation
    - DNA Encoding: Nucleic acid encoding
    - XOR: Quantum logic gates
    """
    encrypted_tiles = []
    tile_info = []
    
    for tile_idx, tile in enumerate(roi_tiles):
        encrypted_channels = []
        
        for c in range(3):
            channel = tile[:, :, c]
            
            # Unique seed per tile
            tile_seed = (master_seed + tile_idx) % (2**31)
            
            try:
                encrypted_channel = encrypt_single_channel(
                    channel,
                    master_seed=tile_seed,
                    channel_index=c,
                    shots=16384,  # Optimized quantum shots
                    use_quantum_encoding=True,  # QUANTUM PATH
                    quantum_encoder='neqr'
                )
                encrypted_channels.append(encrypted_channel)
            except Exception as e:
                print(f"[WARNING] Quantum encryption failed for tile {tile_idx}: {e}")
                # Fallback: use channel as-is
                encrypted_channels.append(channel)
        
        encrypted_tile = np.stack(encrypted_channels, axis=-1)
        encrypted_tiles.append(encrypted_tile)
        tile_info.append({'tile_index': tile_idx, 'seed': int(tile_seed)})
    
    metadata = {
        'num_tiles': len(roi_tiles),
        'encryption_type': 'quantum',
        'tile_info': tile_info,
        'master_seed': int(master_seed)
    }
    
    return encrypted_tiles, metadata


def stage4_encrypt_background_classical(background_image: np.ndarray, master_seed: int) -> Tuple[np.ndarray, Dict]:
    """
    Path B: Classical Encryption (Background)
    - HLSM: Hybrid Logistic-Sine Map for chaos
    - DNA Encoding
    - XOR diffusion
    """
    encrypted_channels = []
    
    # Pad to power of 2
    padded_bg, original_shape = pad_to_power_of_2(background_image, pad_mode='edge')
    
    for c in range(3):
        channel = padded_bg[:, :, c]
        
        try:
            encrypted_channel = encrypt_single_channel(
                channel,
                master_seed=master_seed,
                channel_index=c + 10,  # Offset from ROI channels
                shots=8192,  # Optimized for classical
                use_quantum_encoding=False,  # CLASSICAL PATH (no quantum)
                quantum_encoder='neqr'
            )
            encrypted_channels.append(encrypted_channel)
        except Exception as e:
            print(f"[WARNING] Classical encryption failed: {e}")
            encrypted_channels.append(channel)
    
    encrypted_padded = np.stack(encrypted_channels, axis=-1)
    encrypted_bg = crop_to_original(encrypted_padded, original_shape)
    
    metadata = {
        'shape': background_image.shape,
        'encryption_type': 'classical',
        'master_seed': int(master_seed)
    }
    
    return encrypted_bg, metadata


# ============================================================================
# STAGE 5: RECONSTRUCTION & FUSION
# ============================================================================

def stage5_stitch_tiles(encrypted_tiles: List[np.ndarray], tile_metadata: Dict, encrypted_bg: np.ndarray) -> np.ndarray:
    """
    Stitch encrypted ROI tiles back and merge with encrypted background
    """
    h, w = encrypted_bg.shape[:2]
    encrypted_image = encrypted_bg.copy()
    
    # Place each encrypted tile at its original coordinates
    for tile_idx, tile_pos in enumerate(tile_metadata['tile_positions']):
        if tile_idx >= len(encrypted_tiles):
            break
        
        encrypted_tile = encrypted_tiles[tile_idx]
        
        orig_y = tile_pos['original_y']
        orig_x = tile_pos['original_x']
        tile_h = tile_pos['tile_height']
        tile_w = tile_pos['tile_width']
        
        # Place without resizing (zero loss)
        encrypted_image[orig_y:orig_y+tile_h, orig_x:orig_x+tile_w, :] = encrypted_tile[:tile_h, :tile_w, :]
    
    return encrypted_image


# ============================================================================
# DECRYPTION (Reverse process)
# ============================================================================

def decrypt_roi_tiles_quantum(encrypted_tiles: List[np.ndarray], master_seed: int) -> List[np.ndarray]:
    """Decrypt ROI tiles"""
    decrypted_tiles = []
    
    for tile_idx, encrypted_tile in enumerate(encrypted_tiles):
        decrypted_channels = []
        
        for c in range(3):
            channel = encrypted_tile[:, :, c]
            tile_seed = (master_seed + tile_idx) % (2**31)
            
            try:
                decrypted_channel = decrypt_single_channel(
                    channel,
                    master_seed=tile_seed,
                    channel_index=c,
                    shots=16384,
                    use_quantum_encoding=True,
                    quantum_encoder='neqr'
                )
                decrypted_channels.append(decrypted_channel)
            except:
                decrypted_channels.append(channel)
        
        decrypted_tile = np.stack(decrypted_channels, axis=-1)
        decrypted_tiles.append(decrypted_tile)
    
    return decrypted_tiles


def decrypt_background_classical(encrypted_bg: np.ndarray, master_seed: int) -> np.ndarray:
    """Decrypt background"""
    padded_encrypted, original_shape = pad_to_power_of_2(encrypted_bg, pad_mode='edge')
    decrypted_channels = []
    
    for c in range(3):
        channel = padded_encrypted[:, :, c]
        
        try:
            decrypted_channel = decrypt_single_channel(
                channel,
                master_seed=master_seed,
                channel_index=c + 10,
                shots=8192,
                use_quantum_encoding=False,
                quantum_encoder='neqr'
            )
            decrypted_channels.append(decrypted_channel)
        except:
            decrypted_channels.append(channel)
    
    decrypted_padded = np.stack(decrypted_channels, axis=-1)
    return crop_to_original(decrypted_padded, original_shape)


def reconstruct_roi(decrypted_tiles: List[np.ndarray], tile_metadata: Dict, original_shape: Tuple) -> np.ndarray:
    """Reconstruct ROI from decrypted tiles"""
    roi_bbox = tile_metadata['roi_bbox']
    roi_y_min, roi_x_min, roi_height, roi_width = roi_bbox
    
    roi = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)
    
    for tile_idx, tile_pos in enumerate(tile_metadata['tile_positions']):
        if tile_idx >= len(decrypted_tiles):
            break
        
        decrypted_tile = decrypted_tiles[tile_idx]
        tile_h = tile_pos['tile_height']
        tile_w = tile_pos['tile_width']
        
        rel_y = tile_pos['original_y'] - roi_y_min
        rel_x = tile_pos['original_x'] - roi_x_min
        
        roi[rel_y:rel_y+tile_h, rel_x:rel_x+tile_w, :] = decrypted_tile[:tile_h, :tile_w, :]
    
    return roi


# ============================================================================
# QUALITY METRICS
# ============================================================================

def calculate_metrics(original: np.ndarray, decrypted: np.ndarray) -> Dict:
    """Calculate PSNR, SSIM, and entropy"""
    metrics = {}
    
    if original.shape != decrypted.shape:
        return {'error': 'Shape mismatch'}
    
    original = original.astype(np.float32)
    decrypted = decrypted.astype(np.float32)
    
    # PSNR
    if peak_signal_noise_ratio:
        try:
            metrics['psnr'] = peak_signal_noise_ratio(original, decrypted, data_range=255)
        except:
            metrics['psnr'] = None
    
    # SSIM
    if structural_similarity:
        try:
            metrics['ssim'] = structural_similarity(original/255.0, decrypted/255.0, data_range=1.0, channel_axis=2)
        except:
            metrics['ssim'] = None
    
    # Entropy
    encrypted = np.clip(decrypted, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(encrypted.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    metrics['entropy'] = -np.sum(hist * np.log2(hist))
    
    return metrics


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main encryption/decryption pipeline"""
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("DUAL-ENGINE SECURE IMAGE ENCRYPTION PIPELINE")
    print("Engine A (Intelligence) + Engine B (Security)")
    print("="*80)
    
    # Find input image
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    if not image_files:
        print(f"[ERROR] No images in {input_dir}")
        return
    
    image_file = image_files[0]
    print(f"\n[PROCESSING] {image_file.name}")
    
    # Load image
    image_bgr = cv2.imread(str(image_file))
    if image_bgr is None:
        print(f"[ERROR] Failed to load {image_file}")
        return
    
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    master_seed = int(hashlib.md5(str(datetime.now()).encode()).hexdigest(), 16) % (2**31)
    
    # STAGE 2: AI Segmentation
    print("\n[STAGE 2] Engine A - Intelligence Layer (ROI Detection)...")
    t_start = time.time()
    roi_mask = stage2_ai_segmentation(image)
    t_stage2 = time.time() - t_start
    print(f"  Time: {t_stage2:.2f}s")
    
    # STAGE 3: Smart Bridge
    print("\n[STAGE 3] Engine A - Logic Core (Tiling)...")
    t_start = time.time()
    roi_tiles, tile_metadata, background_image = stage3_smart_bridge(image, roi_mask)
    t_stage3 = time.time() - t_start
    print(f"  Total tiles: {len(roi_tiles)}")
    print(f"  Time: {t_stage3:.2f}s")
    
    # STAGE 4A: Quantum Encryption (ROI)
    print("\n[STAGE 4A] Engine B - Quantum Encryption (ROI)...")
    t_start = time.time()
    encrypted_roi_tiles, roi_metadata = stage4_encrypt_roi_quantum(roi_tiles, master_seed)
    t_stage4a = time.time() - t_start
    print(f"  Time: {t_stage4a:.2f}s")
    
    # STAGE 4B: Classical Encryption (Background)
    print("\n[STAGE 4B] Engine B - Classical Encryption (Background)...")
    t_start = time.time()
    encrypted_bg, bg_metadata = stage4_encrypt_background_classical(background_image, master_seed)
    t_stage4b = time.time() - t_start
    print(f"  Time: {t_stage4b:.2f}s")
    
    # STAGE 5: Reconstruct & Fuse
    print("\n[STAGE 5] Engine B - Reconstruction & Fusion...")
    t_start = time.time()
    encrypted_image = stage5_stitch_tiles(encrypted_roi_tiles, tile_metadata, encrypted_bg)
    t_stage5 = time.time() - t_start
    print(f"  Time: {t_stage5:.2f}s")
    
    # Decryption
    print("\n[DECRYPTION] Reversing encryption...")
    t_start = time.time()
    
    decrypted_roi_tiles = decrypt_roi_tiles_quantum(encrypted_roi_tiles, master_seed)
    decrypted_bg = decrypt_background_classical(encrypted_bg, master_seed)
    
    # Reconstruct full image
    decrypted_roi = reconstruct_roi(decrypted_roi_tiles, tile_metadata, original_image.shape)
    
    decrypted_image = decrypted_bg.copy()
    roi_bbox = tile_metadata['roi_bbox']
    roi_y_min, roi_x_min, roi_height, roi_width = roi_bbox
    decrypted_image[roi_y_min:roi_y_min+roi_height, roi_x_min:roi_x_min+roi_width, :] = decrypted_roi
    
    decrypted_image = np.clip(decrypted_image, 0, 255).astype(np.uint8)
    t_decrypt = time.time() - t_start
    print(f"  Time: {t_decrypt:.2f}s")
    
    # Calculate metrics
    print("\n[METRICS] Quality Analysis...")
    metrics = calculate_metrics(original_image, decrypted_image)
    print(f"  PSNR: {metrics.get('psnr', 'N/A'):.2f} dB" if metrics.get('psnr') else f"  PSNR: N/A")
    print(f"  SSIM: {metrics.get('ssim', 'N/A'):.4f}" if metrics.get('ssim') else f"  SSIM: N/A")
    print(f"  Entropy: {metrics.get('entropy', 'N/A'):.4f} bits/byte")
    
    # Save results
    result_dir = output_dir / image_file.stem
    result_dir.mkdir(exist_ok=True)
    
    encrypted_bgr = cv2.cvtColor(encrypted_image, cv2.COLOR_RGB2BGR)
    decrypted_bgr = cv2.cvtColor(decrypted_image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(result_dir / "encrypted_image.png"), encrypted_bgr)
    cv2.imwrite(str(result_dir / "decrypted_image.png"), decrypted_bgr)
    np.save(str(result_dir / "encrypted_image.npy"), encrypted_image)
    np.save(str(result_dir / "decrypted_image.npy"), decrypted_image)
    
    # Save metadata
    metadata = {
        'image': image_file.name,
        'master_seed': int(master_seed),
        'tile_metadata': tile_metadata,
        'roi_metadata': roi_metadata,
        'bg_metadata': bg_metadata,
        'metrics': {k: float(v) if isinstance(v, (int, np.number)) else v for k, v in metrics.items()},
        'execution_times': {
            'stage2_ai_segmentation': t_stage2,
            'stage3_logic_core': t_stage3,
            'stage4a_quantum_roi': t_stage4a,
            'stage4b_classical_bg': t_stage4b,
            'stage5_reconstruction': t_stage5,
            'decryption': t_decrypt,
            'total': t_stage2 + t_stage3 + t_stage4a + t_stage4b + t_stage5 + t_decrypt
        }
    }
    
    with open(result_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    total_time = metadata['execution_times']['total']
    print("\n" + "="*80)
    print(f"[COMPLETE] Total time: {total_time:.2f}s")
    print(f"Output: {result_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
