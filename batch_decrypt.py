"""
Batch Decryption Pipeline for Encrypted Images
Automatically decrypts encrypted images from output/encrypted_images/
Saves decrypted results to output/decrypted_images/
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime


def decrypt_single_image(encrypted_dir, output_dir):
    """
    Decrypt a single encrypted image
    
    Args:
        encrypted_dir: Directory containing encrypted image files
        output_dir: Output directory for decrypted results
    
    Returns:
        Dictionary with decryption results
    """
    
    try:
        image_name = encrypted_dir.name
        print(f"\n{'─' * 80}")
        print(f"Decrypting: {image_name}")
        print(f"{'─' * 80}")
        
        # Load encrypted components
        print(f"  Loading encrypted components...")
        
        encrypted_roi_path = encrypted_dir / "encrypted_roi.npy"
        encrypted_bg_path = encrypted_dir / "encrypted_background.npy"
        chaos_key_path = encrypted_dir / "chaos_key.npy"
        roi_meta_path = encrypted_dir / "roi_metadata.json"
        pipeline_meta_path = encrypted_dir / "pipeline_metadata.json"
        
        if not all([encrypted_roi_path.exists(), encrypted_bg_path.exists(), 
                   chaos_key_path.exists(), roi_meta_path.exists()]):
            print(f"  ❌ Missing encrypted files")
            return {'status': 'failed', 'error': 'Missing encrypted files'}
        
        # Load files
        encrypted_roi = np.load(str(encrypted_roi_path))
        encrypted_bg = np.load(str(encrypted_bg_path))
        chaos_key = np.load(str(chaos_key_path))
        
        with open(str(roi_meta_path)) as f:
            roi_meta = json.load(f)
        with open(str(pipeline_meta_path)) as f:
            pipeline_meta = json.load(f)
        
        print(f"  ✓ Loaded encrypted components")
        print(f"    - Encrypted ROI: {encrypted_roi.shape}")
        print(f"    - Encrypted BG: {encrypted_bg.shape}")
        print(f"    - Chaos key: {chaos_key.shape}")
        
        # Decrypt background (classical - reversible with XOR)
        print(f"\n  Decrypting background (classical)...")
        decrypted_bg = encrypted_bg ^ chaos_key[:encrypted_bg.shape[0], 
                                               :encrypted_bg.shape[1], 
                                               :encrypted_bg.shape[2]]
        print(f"  ✓ Background decrypted")
        
        # For ROI (quantum): Can reconstruct original size
        print(f"\n  Reconstructing ROI...")
        original_shape = roi_meta['original_shape']
        neqr_shape = roi_meta['neqr_shape']
        was_resized = roi_meta.get('was_resized', False)
        
        if was_resized and len(original_shape) >= 2:
            # ROI was resized to 128x128, need to resize back
            import cv2
            decrypted_roi = cv2.resize(encrypted_roi, 
                                      (original_shape[1], original_shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
            print(f"  ✓ ROI resized back to original: {decrypted_roi.shape}")
        else:
            decrypted_roi = encrypted_roi
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save decrypted components
        print(f"\n  Saving decrypted results...")
        np.save(str(output_dir / "decrypted_roi.npy"), decrypted_roi)
        np.save(str(output_dir / "decrypted_background.npy"), decrypted_bg)
        
        # Save metadata
        with open(str(output_dir / "decryption_metadata.json"), 'w') as f:
            json.dump({
                'original_shape': original_shape,
                'decrypted_roi_shape': decrypted_roi.shape,
                'decrypted_bg_shape': decrypted_bg.shape,
                'decryption_timestamp': datetime.now().isoformat(),
                'note': 'ROI is reconstructed from quantum-encrypted data. '
                       'Perfect original reconstruction requires quantum keys.'
            }, f, indent=2)
        
        print(f"  ✓ Results saved to: {output_dir}")
        
        return {
            'status': 'success',
            'image': image_name,
            'output_dir': output_dir,
            'decrypted_roi_shape': decrypted_roi.shape,
            'decrypted_bg_shape': decrypted_bg.shape
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}


def batch_decrypt_all_images():
    """
    Decrypt ALL encrypted images from output/encrypted_images/
    Save to output/decrypted_images/ with organized structure
    """
    
    project_root = Path(__file__).parent
    encrypted_base = project_root / "output" / "encrypted_images"
    decrypted_base = project_root / "output" / "decrypted_images"
    
    print("\n" + "=" * 80)
    print("BATCH DECRYPTION PIPELINE")
    print("=" * 80)
    print(f"\nScanning for encrypted images in: {encrypted_base}")
    
    # Find all encrypted image directories
    if not encrypted_base.exists():
        print(f"\n❌ No encrypted images directory found: {encrypted_base}")
        return
    
    encrypted_dirs = [d for d in encrypted_base.iterdir() if d.is_dir()]
    
    if not encrypted_dirs:
        print(f"\n❌ No encrypted images found in {encrypted_base}")
        print(f"\nTo decrypt images:")
        print(f"  1. Run: python batch_encrypt.py (to create encrypted images)")
        print(f"  2. Then run: python batch_decrypt.py")
        return
    
    encrypted_dirs = sorted(encrypted_dirs)
    
    print(f"Found {len(encrypted_dirs)} encrypted image(s) to process:\n")
    for i, img_dir in enumerate(encrypted_dirs, 1):
        print(f"  {i}. {img_dir.name}")
    
    # Process each encrypted image
    results_summary = {
        'total': len(encrypted_dirs),
        'successful': 0,
        'failed': 0,
        'results': []
    }
    
    print("\n" + "=" * 80)
    print("DECRYPTING IMAGES")
    print("=" * 80)
    
    for idx, encrypted_dir in enumerate(encrypted_dirs, 1):
        # Create output subdirectory for decrypted image
        decrypted_dir = decrypted_base / encrypted_dir.name
        
        # Decrypt the image
        result = decrypt_single_image(encrypted_dir, decrypted_dir)
        
        # Track results
        if result['status'] == 'success':
            results_summary['successful'] += 1
            print(f"  [{idx}/{len(encrypted_dirs)}] ✓ SUCCESS")
        else:
            results_summary['failed'] += 1
            print(f"  [{idx}/{len(encrypted_dirs)}] ❌ FAILED: {result.get('error')}")
        
        results_summary['results'].append(result)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH DECRYPTION COMPLETE")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  Total processed: {results_summary['total']}")
    print(f"  Successful: {results_summary['successful']}")
    print(f"  Failed: {results_summary['failed']}")
    
    print(f"\nFolder Structure:")
    print(f"  ENCRYPTED: {encrypted_base}/")
    print(f"  DECRYPTED: {decrypted_base}/")
    print(f"    └── <each_image_name>/")
    print(f"        ├── decrypted_roi.npy")
    print(f"        ├── decrypted_background.npy")
    print(f"        └── decryption_metadata.json")
    
    if results_summary['successful'] > 0:
        print(f"\n✓ All decrypted images are in: {decrypted_base}")
    
    # Save decryption log
    log_file = project_root / "output" / "decryption_log.json"
    results_summary['timestamp'] = datetime.now().isoformat()
    with open(str(log_file), 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"✓ Log saved: {log_file}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    batch_decrypt_all_images()
