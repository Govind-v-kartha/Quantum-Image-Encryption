#!/usr/bin/env python3
"""
Diagnostic script to verify encryption strength.
Compares original vs encrypted image statistics.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Load original and encrypted images
original_path = Path("input/st1.png")
encrypted_path = Path("output/st1_01_encrypted/encrypted_image.png")

if not original_path.exists() or not encrypted_path.exists():
    print("ERROR: Image files not found")
    exit(1)

# Load images
original = np.array(Image.open(original_path))
encrypted = np.array(Image.open(encrypted_path))

print("=" * 80)
print("ENCRYPTION STRENGTH DIAGNOSTIC")
print("=" * 80)

print(f"\nOriginal Image:")
print(f"  Shape: {original.shape}")
print(f"  Min: {original.min()}, Max: {original.max()}")
print(f"  Mean: {original.mean():.4f}")
print(f"  Std Dev: {original.std():.4f}")
print(f"  Skewness: {np.mean(((original - original.mean()) / original.std()) ** 3):.4f}")
print(f"  Kurtosis: {np.mean(((original - original.mean()) / original.std()) ** 4):.4f}")

print(f"\nEncrypted Image:")
print(f"  Shape: {encrypted.shape}")
print(f"  Min: {encrypted.min()}, Max: {encrypted.max()}")
print(f"  Mean: {encrypted.mean():.4f}")
print(f"  Std Dev: {encrypted.std():.4f}")
print(f"  Skewness: {np.mean(((encrypted - encrypted.mean()) / encrypted.std()) ** 3):.4f}")
print(f"  Kurtosis: {np.mean(((encrypted - encrypted.mean()) / encrypted.std()) ** 4):.4f}")

# Analyze difference
if original.shape == encrypted.shape:
    diff = np.abs(original.astype(np.int32) - encrypted.astype(np.int32))
    print(f"\nDifference Statistics:")
    print(f"  Mean absolute difference: {diff.mean():.4f}")
    print(f"  Max difference: {diff.max()}")
    print(f"  Percentage of changed pixels: {(diff > 0).sum() / diff.size * 100:.2f}%")
    
    # Correlation check
    orig_flat = original.flatten().astype(np.float32)
    enc_flat = encrypted.flatten().astype(np.float32)
    correlation = np.corrcoef(orig_flat, enc_flat)[0, 1]
    print(f"  Correlation (original vs encrypted): {correlation:.6f}")
    if abs(correlation) < 0.1:
        print("    ✓ GOOD: Very low correlation")
    else:
        print("    ✗ BAD: High correlation indicates weak encryption")
else:
    print("WARNING: Image shapes don't match - cannot compute difference")

# Entropy calculation
def entropy(data):
    """Calculate Shannon entropy."""
    hist, _ = np.histogram(data.flatten(), bins=256, range=(0, 256))
    hist = hist[hist > 0]  # Remove zero bins
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist))

print(f"\nEntropy Analysis:")
print(f"  Original entropy: {entropy(original):.4f}")
print(f"  Encrypted entropy: {entropy(encrypted):.4f}")
print(f"  Maximum possible entropy (8-bit): 8.0")
if entropy(encrypted) > 7.5:
    print("    ✓ GOOD: High entropy")
else:
    print("    ✗ BAD: Low entropy indicates patterns remain")

# Chi-square test for randomness
from scipy.stats import chisquare
hist_enc, _ = np.histogram(encrypted.flatten(), bins=256, range=(0, 256))
chi2, p_value = chisquare(hist_enc)
print(f"\nRandomness Test (Chi-Square):")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  P-value: {p_value:.6f}")
if p_value > 0.05:
    print("    ✓ GOOD: Data appears random")
else:
    print("    ✗ BAD: Data not random (patterns detected)")

print("\n" + "=" * 80)
print("VERDICT:")
if (entropy(encrypted) > 7.5 and p_value > 0.05 and 
    abs(correlation) < 0.1 and diff.mean() > 50):
    print("✓ ENCRYPTION APPEARS STRONG")
else:
    print("✗ ENCRYPTION APPEARS WEAK - Need stronger diffusion")
print("=" * 80)
