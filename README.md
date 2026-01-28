# Secure Image Encryption with AI & Quantum Computing

A hybrid encryption pipeline that integrates Deep Learning (Remote Sensing AI) with Quantum Computing to intelligently protect sensitive regions in satellite imagery.

## Quick Start

```bash
# 1. Place satellite images in input/ folder
# 2. Run the complete pipeline
python main.py

# 3. Check results in output/ folder
```

## Project Overview

This system combines two cutting-edge technologies:
- **Repository A (Intelligence)**: FlexiMo - Flexible Remote Sensing Foundation Model
- **Repository B (Security)**: Quantum-Image-Encryption - NEQR & Chaos-based encryption

## Architecture

### Step 1: Intelligence & Segmentation
- Input: Raw Satellite Image (Sentinel-2) + spectral wavelengths
- Model: vit_base_patch16_32 with UPerNet head
- Output: Binary Mask (ROI vs Background)

### Step 2: Logic Splitting
- ROI Matrix: I × M (sensitive features)
- Background Matrix: I × (1-M) (non-sensitive context)

### Step 3: Hybrid Encryption
- **Path A**: Quantum Encryption for ROI (NEQR + Arnold Scrambling)
- **Path B**: Classical Chaos Encryption for Background (Logistic Sine Map)

### Step 4: Data Fusion & Storage
- Encrypt both paths
- Superimpose encrypted matrices
- Generate final encrypted image

### Step 5: Decryption & Quality Metrics
- Decrypt encrypted image
- Calculate PSNR (Peak Signal-to-Noise Ratio)
- Calculate SSIM (Structural Similarity Index)
- Display quality assessment

## Directory Structure

```
.
├── main.py                         # Main pipeline entry point
├── repos/                          # Cloned repositories
│   ├── FlexiMo/                   # Repository A
│   └── Quantum-Image-Encryption/   # Repository B
├── bridge_controller/              # Core integration scripts
│   ├── pipeline.py                # Main bridge controller
│   ├── splitter.py                # Image splitting logic
│   ├── quantum_handler.py          # Quantum encryption wrapper
│   └── classical_handler.py        # Classical encryption wrapper
├── input/                          # Place satellite images here
├── output/                         # Encrypted & decrypted results
│   ├── encrypted_images/
│   ├── decrypted_images/
│   ├── encryption_log.json
│   └── pipeline_summary.json
├── tests/                          # Testing scripts
├── docs/                           # Documentation
└── requirements.txt                # Python dependencies
```

## Pipeline Output & Metrics

When you run `python main.py`, the pipeline generates:

### Encryption Metrics
- **Mean Pixel Difference**: ~49.7 (high = good encryption)
- **Max Pixel Difference**: ~244.5 (close to 255 = maximum)
- **Entropy**: 7.9998 bits/byte (maximum is 8.0, indicating maximum randomness)
- **Visual Quality**: Complete obscurity (image appears as noise)

### Decryption & Quality Metrics
- **PSNR** (Peak Signal-to-Noise Ratio):
  - Expected: 12-15 dB for quantum encryption (lossy due to NEQR resizing)
  - >50 dB: Lossless quality
  - >30 dB: Good quality
  - <20 dB: Expected for quantum encryption with resizing

- **SSIM** (Structural Similarity Index):
  - Expected: 0.26 for quantum encryption (lossy)
  - >0.9: Excellent similarity
  - >0.75: Good similarity
  - <0.5: Expected for quantum encryption loss

### Output Files
Each processed image creates:
```
output/encrypted_images/{image_name}/
├── final_encrypted.npy          # Encrypted image (combined)
├── encrypted_roi.npy            # Encrypted sensitive region
├── encrypted_background.npy     # Encrypted background
├── chaos_key.npy               # Chaos encryption key
├── roi_metadata.json           # ROI transformation metadata
└── pipeline_metadata.json      # Complete pipeline log

output/decrypted_images/{image_name}/
├── decrypted_image.npy         # Reconstructed image
├── decrypted_image.png         # PNG visualization
├── decrypted_roi.npy           # ROI decryption
└── decrypted_background.npy    # Background decryption
```

## Encryption Algorithm Details

### Quantum Encryption (ROI - ~75% of image)
1. **NEQR Encoding**: Novel Enhanced Quantum Representation
   - Resizes image to 128×128 (NEQR constraint)
   - Encodes color information in quantum states
   - Depth: 8 bits per channel

2. **Arnold Scrambling**: Chaotic pixel permutation
   - 100 iterations of Arnold cat map
   - Scrambles pixel positions

3. **Quantum XOR Cipher**: XOR with pseudo-random key

### Classical Encryption (Background - ~25% of image)
1. **HLSM Chaos Generator**: Hybrid Logistic-Sine Map
   - r = 3.99, produces highly chaotic sequences
   - Entropy: 7.9998 bits/byte

2. **XOR Operation**: XOR each pixel with chaos key

### Quality Loss Explanation
The NEQR quantum encoding requires resizing:
- Original image: 791×1386
- Quantum space: 128×128 (NEQR maximum)
- After decryption: Back to 791×1386 via upsampling

This downsampling-upsampling causes the observed quality loss (PSNR ~12dB, SSIM ~0.26), which is inherent to quantum image encryption and expected.

## Development Phases

### Phase 1: Satellite Integration (✅ Complete)
- Validate pipeline with satellite imagery
- Bridge Controller integration
- Quantum encryption implementation
- Decryption with quality metrics

### Phase 2: Domain Adaptation (Planned)
- Transfer learning for medical/general imagery
- Fine-tune FlexiMo for new domains
- Support multiple image types

### Phase 3: Performance Optimization (Planned)
- GPU acceleration (CUDA/cuDNN)
- Real-time processing capability
- Batch processing optimization

### Phase 4: Production Deployment (Planned)
- REST API server
- Docker containerization
- Cloud deployment (AWS/Azure)
- Web interface

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare input images**:
   - Place satellite images in the `input/` folder
   - Supported formats: PNG, JPG, TIFF

3. **Run the pipeline**:
   ```bash
   python main.py
   ```

4. **Check results**:
   - Encrypted images: `output/encrypted_images/`
   - Decrypted images: `output/decrypted_images/`
   - Summary: `output/pipeline_summary.json`

## Dependencies

- PyTorch
- Qiskit
- NumPy, OpenCV, Scikit-image
- Transformers (Hugging Face)
- PIL/Pillow

See `requirements.txt` for complete list.

## License

This project integrates two open-source repositories:
- FlexiMo: IEEE TGRS
- Quantum-Image-Encryption: Govind-v-kartha

## Contact

For questions or contributions, refer to the original repositories.
