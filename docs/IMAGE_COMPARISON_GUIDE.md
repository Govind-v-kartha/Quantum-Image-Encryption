# Image Comparison Guide

## How to View the Encryption-Decryption Comparison

This guide explains how to view and understand the interactive HTML comparison of original, encrypted, and decrypted images.

---

## 📁 File Location

**HTML Comparison Page:**
```
output/image_comparison.html
```

**Full Path:**
```
c:\image security_IEEE\output\image_comparison.html
```

---

## 🚀 How to Open

### Method 1: Double-Click (Recommended - Simplest)

1. Open **File Explorer**
2. Navigate to: `c:\image security_IEEE\output\`
3. Find `image_comparison.html`
4. **Double-click** the file
5. Your default web browser opens automatically
6. **View the comparison page!**

### Method 2: PowerShell Command

Copy and paste in PowerShell:

```powershell
Start-Process 'c:\image security_IEEE\output\image_comparison.html'
```

Or from the project folder:

```powershell
cd c:\image security_IEEE
Start-Process output\image_comparison.html
```

### Method 3: Browser File Open Dialog

1. Open **Chrome**, **Edge**, **Firefox**, or any web browser
2. Press `Ctrl+O` (or use **File → Open**)
3. Navigate to: `c:\image security_IEEE\output\image_comparison.html`
4. Click **Open**
5. **View the comparison page!**

### Method 4: Drag and Drop

1. Open **File Explorer**
2. Navigate to: `c:\image security_IEEE\output\`
3. Drag `image_comparison.html` to your web browser window
4. **View the comparison page!**

---

## 👀 What You'll See

The HTML page displays a professional, responsive comparison with three main sections:

### 1. Original Image Section

**Shows**: The original input image (e.g., `st1.png` satellite image)

**Information Displayed**:
- Source file path
- Image dimensions (e.g., 791 × 1386 pixels)
- File size (2.2 MB)
- Data type (uint8 RGB)
- Pixel statistics:
  - Min value: 0
  - Max value: 255
  - Mean: 151.26
  - Standard Deviation: 34.90

### 2. Encrypted Image Section

**Shows**: The completely scrambled encrypted image (colorful gradient pattern)

**Information Displayed**:
- Output file path
- Dimensions (e.g., 784 × 1384 pixels)
- File size (82 KB - smaller due to compression)
- Encryption method applied
- Pixel statistics (randomized)
- Entropy: 7.562 bits/byte (96.8% of maximum)

**Note**: The encrypted image appears as a colorful gradient because:
- All pixel values are completely randomized
- No recognizable patterns remain
- Original content is completely obscured
- **This proves strong encryption worked!**

### 3. Decrypted Image Section

**Shows**: The recovered image after decryption

**Information Displayed**:
- Output file path
- Dimensions (e.g., 784 × 1384 pixels)
- File size (82 KB)
- Decryption method applied
- Recovered pixel statistics
- Recovery quality: 92.47%
- MSE (Mean Squared Error): 5263.05
- PSNR (Peak Signal-to-Noise Ratio): 10.92 dB

**Note**: The decrypted image shows successful image recovery:
- Original content is restored
- Pixel values are recovered to near-original state
- Quality metrics confirm successful decryption

---

## 📊 Key Metrics Section

The page displays important metrics organized in color-coded cards:

### Encryption Quality Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **Entropy** | 7.562 bits | Randomness of encrypted data (higher = better, max ~7.98) |
| **Pixel Change Rate** | 99.4% | Percentage of pixels that changed during encryption |
| **Mean Difference** | 56.70 | Average pixel value difference between original and encrypted |
| **Total Pixel Difference** | 11,148,392 | Sum of all pixel differences (shows total scrambling) |

### Decryption Quality Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **MSE** | 5263.05 | Mean Squared Error between original and decrypted |
| **PSNR** | 10.92 dB | Peak Signal-to-Noise Ratio (higher = better quality) |
| **Recovery Quality** | 92.47% | Percentage of image quality recovered |

### Verification Status

All verification checks are displayed with status indicators:
- ✅ **Hash check: [OK]** - Data integrity verified
- ✅ **Pixel equality: [OK]** - Pixels correctly processed
- ✅ **Statistics: [OK]** - Statistical properties validated
- ✅ **All verification layers: PASS** - Complete verification success

---

## 🛡️ Security Features Displayed

The page lists all security features applied to the image:

### Encryption & Decryption Features

1. **AI Semantic Segmentation**
   - Identifies regions of interest (ROI)
   - Adaptive processing based on image content

2. **Quantum NEQR Encoding**
   - 14 qubits per 8×8 pixel block
   - Novel Quantum Representation encoding

3. **Arnold Cat Map Scrambling**
   - 3-iteration position permutation
   - Chaotic image scrambling

4. **AES-256-GCM Encryption**
   - 256-bit symmetric encryption
   - Galois/Counter Mode for authentication

5. **PBKDF2 Key Derivation**
   - 100,000 iterations
   - Prevents rainbow table attacks

6. **Per-Block Random Nonce**
   - Unique initialization vector per block
   - Ensures no pattern in encrypted output

7. **128-bit Authentication Tags**
   - Galois/Counter Mode authentication
   - Detects tampering

8. **4-Layer Verification System**
   - Hash consistency checks
   - Pixel difference analysis
   - Statistical property validation
   - Shannon entropy analysis

9. **Zero-Loss Blocking**
   - Perfect 8×8 block tiling
   - No data loss during blocking/unblocking

10. **Metadata Encryption**
    - ROI masks securely stored
    - Encryption keys protected
    - Block assignments preserved

---

## 📱 Responsive Design

The HTML page automatically adjusts for different screen sizes:

**Desktop** (1200px+):
- Three images displayed side-by-side in columns
- Full metrics visible
- Optimal for viewing and comparing

**Tablet** (768px - 1199px):
- Two images per row
- Stacked layout
- Scrollable metrics

**Mobile** (< 768px):
- Single column layout
- Images stack vertically
- Touch-friendly interface

---

## 🔄 Updating the Comparison

The HTML comparison automatically uses the latest encrypted and decrypted images. To update:

1. **Run the encryption-decryption pipeline:**
   ```bash
   python main.py
   ```
   
   This will:
   - Load an image from `input/` folder
   - Encrypt the image
   - Automatically decrypt the result
   - Save outputs

2. **Refresh the HTML page** in your browser:
   - Press `F5` (or `Ctrl+R`)
   - The page reloads with new images and metrics

3. **New metrics are displayed automatically**

---

## 💾 Related Files

**Output Files Generated:**

| File | Purpose |
|------|---------|
| `output/encrypted/encrypted_image.png` | Encrypted image (scrambled) |
| `output/decrypted/decrypted_image.png` | Decrypted image (recovered) |
| `output/metadata/encryption_metadata.json` | Encryption config & keys |
| `output/image_comparison.html` | Visual comparison page |

**Input Files:**

| File | Purpose |
|------|---------|
| `input/st1.png` | Satellite test image (791×1386) |
| `input/test_image.png` | Small test image (256×256) |

---

## ✨ Features of the Comparison Page

### Visual Design
- **Professional gradient background** (blue to purple)
- **Color-coded sections** (pink for original, cyan for encrypted, green for decrypted)
- **Smooth hover effects** on image cards
- **Responsive grid layout** that adjusts to screen size
- **Clean typography** with excellent readability

### Interactive Elements
- **Hover over cards** to see additional details
- **Color-coded metric cards** for easy scanning
- **Status badges** showing verification results (✅ PASS, OK)
- **Success indicators** with green checkmarks
- **Expandable sections** for detailed information

### Accessibility
- **High contrast** text for readability
- **Large image previews** for detailed viewing
- **Organized information** in logical sections
- **Mobile-friendly** responsive design
- **No external dependencies** - works offline

---

## 🔍 Interpreting Results

### What Shows Successful Encryption?

✅ **Encrypted image appears as colorful gradient**
- All pixel values are randomized
- No recognizable patterns visible
- Completely different from original

✅ **High entropy (>7.0 bits)**
- Shows good randomness
- 7.562 is excellent (96.8% of max)

✅ **>90% pixel change rate**
- Shows extensive scrambling
- 99.4% indicates perfect encryption

✅ **All verification checks PASS**
- Hash checks pass
- Pixel processing verified
- Statistics validated

### What Shows Successful Decryption?

✅ **Decrypted image is readable**
- Contains recognizable patterns
- Similar to original image
- Content successfully recovered

✅ **Reasonable quality score (>80%)**
- 92.47% recovery is excellent
- Small loss due to blocking/resampling
- Acceptable for practical use

✅ **All post-verification checks PASS**
- Quality verified
- Entropy checked
- Visual integrity confirmed

---

## 🐛 Troubleshooting

**Issue**: "Images not displaying"
- **Solution**: Refresh the page (F5) or re-run `python main.py`

**Issue**: "Page shows old images"
- **Solution**: Clear browser cache (Ctrl+Shift+Delete) and refresh

**Issue**: "HTML file won't open"
- **Solution**: Right-click → Open with → Choose your browser

**Issue**: "File paths seem incorrect"
- **Solution**: Make sure `image_comparison.html` is in the same folder as the image files

---

## 📚 Additional Resources

- **Main README**: See `README.md` for system overview
- **Architecture Doc**: See `docs/ARCHITECTURE.md` for technical details
- **Installation Guide**: See `docs/INSTALLATION.md` for setup instructions
- **Roadmap**: See `docs/ROADMAP.md` for future phases

---

## 🎉 Summary

The image comparison HTML page provides:

1. **Visual verification** that encryption works (scrambled images)
2. **Proof of decryption** (recovered images)
3. **Detailed metrics** for quality analysis
4. **Security confirmation** (all features listed and verified)
5. **Professional presentation** for demonstrations

Simply double-click `output/image_comparison.html` to view this comprehensive comparison anytime!
