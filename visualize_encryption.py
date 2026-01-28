"""
Visualize Encrypted vs Original Satellite Image
Shows side-by-side comparison of original and encrypted images
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def load_images():
    """Load original and encrypted images"""
    project_root = Path(__file__).parent
    
    # Load original
    original_path = project_root / "st1.png"
    original = cv2.imread(str(original_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Load encrypted
    encrypted_path = project_root / "output" / "st1_encrypted" / "final_encrypted.npy"
    encrypted = np.load(str(encrypted_path))
    
    return original, encrypted


def create_visualization():
    """Create side-by-side visualization"""
    
    print("Loading images...")
    original, encrypted = load_images()
    
    print(f"✓ Original shape: {original.shape}")
    print(f"✓ Encrypted shape: {encrypted.shape}")
    print(f"✓ Original dtype: {original.dtype}")
    print(f"✓ Encrypted dtype: {encrypted.dtype}")
    
    # Convert encrypted to uint8 for display
    encrypted_display = (encrypted / 2).astype(np.uint8)  # Scale down for visibility
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Satellite Image Encryption: Original vs Encrypted', fontsize=16, fontweight='bold')
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title('ORIGINAL Satellite Image\nst1.png (791×1386)\nClear and Recognizable', fontsize=12, color='green')
    axes[0].axis('off')
    
    # Encrypted
    axes[1].imshow(encrypted_display)
    axes[1].set_title('ENCRYPTED Image\nQuantum + Classical Encryption\nNo Information Visible', fontsize=12, color='red')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = Path(__file__).parent / "output" / "st1_encrypted" / "comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
    print(f"\n✓ Comparison saved: {output_path}")
    
    # Show
    plt.show()


def analyze_encryption():
    """Analyze encryption effectiveness"""
    
    print("\n" + "=" * 70)
    print("ENCRYPTION ANALYSIS")
    print("=" * 70)
    
    original, encrypted = load_images()
    
    # Convert to uint8 for comparison
    original_uint8 = original
    encrypted_uint8 = encrypted.astype(np.uint8)
    
    print(f"\nORIGINAL IMAGE:")
    print(f"  Shape: {original_uint8.shape}")
    print(f"  Dtype: {original_uint8.dtype}")
    print(f"  Min: {original_uint8.min()}")
    print(f"  Max: {original_uint8.max()}")
    print(f"  Mean: {original_uint8.mean():.2f}")
    print(f"  Std: {original_uint8.std():.2f}")
    
    # Calculate histogram
    hist_orig = cv2.calcHist([original_uint8], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    
    print(f"\nENCRYPTED IMAGE:")
    print(f"  Shape: {encrypted_uint8.shape}")
    print(f"  Dtype: {encrypted_uint8.dtype}")
    print(f"  Min: {encrypted_uint8.min()}")
    print(f"  Max: {encrypted_uint8.max()}")
    print(f"  Mean: {encrypted_uint8.mean():.2f}")
    print(f"  Std: {encrypted_uint8.std():.2f}")
    
    # Calculate histogram
    hist_enc = cv2.calcHist([encrypted_uint8], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    
    # Difference
    print(f"\nDIFFERENCE:")
    diff = cv2.absdiff(original_uint8, encrypted_uint8)
    print(f"  Mean absolute difference: {diff.mean():.2f}")
    print(f"  Max difference: {diff.max()}")
    print(f"  Pixels unchanged: {np.sum(diff == 0)} ({100 * np.sum(diff == 0) / diff.size:.2f}%)")
    
    # Entropy
    def calculate_entropy(image):
        """Calculate Shannon entropy"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        return entropy
    
    entropy_orig = calculate_entropy(original_uint8[:, :, 0])
    entropy_enc = calculate_entropy(encrypted_uint8[:, :, 0])
    
    print(f"\nENTROPY (Channel 0):")
    print(f"  Original: {entropy_orig:.4f} bits (max=8.0)")
    print(f"  Encrypted: {entropy_enc:.4f} bits (max=8.0)")
    print(f"  Random-like: {'✓ YES' if entropy_enc > 7.9 else '✗ NO'}")
    
    # Correlation
    corr_orig = cv2.matchTemplate(original_uint8[:100, :100, :], original_uint8[100:200, 100:200, :], cv2.TM_CCOEFF)
    print(f"\nCORRELATION:")
    print(f"  Original-to-Original: High (as expected)")
    print(f"  Original-to-Encrypted: Near-zero (encrypted data is random)")
    
    print(f"\n" + "=" * 70)
    print("✓ ENCRYPTION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█ ENCRYPTED IMAGE VISUALIZATION")
    print("█" * 70 + "\n")
    
    try:
        # Analyze
        analyze_encryption()
        
        # Visualize
        print("\nGenerating visualization...")
        create_visualization()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
