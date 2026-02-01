# Installation & Quick Start Guide

## System Requirements

### Hardware
- **CPU**: Modern multi-core processor (Intel i5+ or AMD Ryzen 5+)
- **RAM**: Minimum 4GB, recommended 8GB or more
- **Disk**: 1GB free space (for dependencies + test images)
- **GPU**: Optional (Phase 10 will support CUDA acceleration)

### Operating System
- ✅ Windows 10/11 (tested)
- ✅ Linux (Ubuntu 18.04+)
- ✅ macOS (10.14+)

### Python Version
- **Required**: Python 3.8 or higher
- **Recommended**: Python 3.10 or 3.11
- **Tested with**: Python 3.10

---

## Quick Start (5 minutes)

### 1. Navigate to Project

```bash
cd c:\image\ security_IEEE
```

### 2. Activate Virtual Environment

**Windows**:
```bash
.venv\Scripts\activate
```

**Linux/macOS**:
```bash
source .venv/bin/activate
```

### 3. Verify Installation

```bash
python -c "import numpy, PIL, cryptography; print('✅ All dependencies ready')"
```

### 4. Run Encryption

```bash
python main.py
```

### 5. View Results

```bash
# Encrypted image:
dir output\encrypted\encrypted_image.png

# Decrypted image:
dir output\decrypted\decrypted_image.png
```

---

## Full Installation Guide

---

## Installation Steps

### 1. Clone the Repository

**Windows**:
```bash
git clone https://github.com/Govind-v-kartha/Quantum-Image-Encryption.git
cd Quantum-Image-Encryption
```

**Linux/macOS**:
```bash
git clone https://github.com/Govind-v-kartha/Quantum-Image-Encryption.git
cd Quantum-Image-Encryption
```

### 2. Create Virtual Environment

**Windows**:
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**What's installed**:
```
numpy              # Array operations
opencv-python     # Image processing
qiskit             # Quantum circuit simulation
qiskit-aer         # Quantum simulator backend
```

### 5. Verify Installation

```bash
python -c "import cv2, numpy, qiskit; print('All dependencies installed successfully!')"
```

---

## Detailed Dependency Installation

### If using conda (Anaconda/Miniconda)

```bash
# Create conda environment
conda create -n quantum-encryption python=3.10

# Activate environment
conda activate quantum-encryption

# Install dependencies
conda install -c conda-forge numpy opencv qiskit qiskit-aer
```

### If specific versions needed

```bash
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install qiskit==0.43.0
pip install qiskit-aer==0.13.0
```

### Troubleshooting Installation Issues

**Issue**: ImportError for cv2

**Solution**:
```bash
pip uninstall opencv-python
pip install opencv-python --force-reinstall
```

**Issue**: qiskit installation fails

**Solution** (try pre-built wheel):
```bash
pip install --only-binary :all: qiskit
```

**Issue**: Permission denied errors

**Solution**:
```bash
# Use --user flag
pip install --user -r requirements.txt
```

---

## Project Setup

### 1. Create Input/Output Directories

The script automatically creates these, but you can pre-create them:

```bash
# Windows
mkdir input
mkdir output
mkdir docs

# Linux/macOS
mkdir -p input output docs
```

### 2. Verify Project Structure

```
Quantum-Image-Encryption/
├── main.py                      # Main pipeline
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
├── input/                       # Place images here
├── output/                      # Results saved here
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── INSTALLATION.md
│   └── ROADMAP.md
└── repos/                       # External repositories
    ├── FlexiMo/
    └── Quantum-Image-Encryption/
```

### 3. Add Test Image

```bash
# Copy a test satellite image to input folder
cp your_image.png input/

# Or use the provided example
# (if included in repository)
```

---

## First Run

### Quick Test

```bash
# Activate virtual environment (if not already)
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# Run the pipeline
python main.py
```

### Expected Output

```
================================================================================
SECURE SATELLITE IMAGE ENCRYPTION PIPELINE
Engine A (Intelligence) + Engine B (Security)
NEQR Quantum Encryption with 8x8 Zero-Loss Tiling
================================================================================

Found 1 image(s)

[Processing] st1.png
  Image shape: 791x1386

  [Stage 1] AI Segmentation (Canny Edge Detection)
           Time: 0.01s

  [Stage 2] ROI Extraction & 8x8 Blocking
           Total 8x8 blocks: 14985
           ROI pixels: 857933 (42%)
           Background pixels: 2431045 (58%)
           Time: 0.08s

Saved: extracted_roi.png, extracted_background.png

  [Stage 3] NEQR + Quantum Scrambling Encryption
           Processing 14985 blocks...
           Encrypted blocks: 14985
           Time: 0.40s

  [Stage 4] Chaos Cipher Encryption (Background)
           Background encrypted
           Time: 0.01s

  [Stage 5] Reconstruct Encrypted Image
           Full encrypted image shape: (791, 1386, 3)
           Time: 0.01s

Saved: encrypted_image.png

  [Stage 6] Decryption
           Time: 0.39s

  [Metrics]
    PSNR: inf dB (Perfect)
    SSIM: 1.0000

Saved: decrypted_image.png

  [Verification]
    Mean pixel difference: 0.00
    Max pixel difference: 0.00

Perfect reconstruction: YES

  [COMPLETE] Total time: 1.18s
  [OUTPUT] C:\image security_IEEE\output\st1_encrypted/
```

### Check Results

```bash
# View generated files
# Windows
dir output\st1_encrypted
dir output\st1_decrypted

# Linux/macOS
ls output/st1_encrypted/
ls output/st1_decrypted/
```

---

## Configuration

### Master Seed Customization

Edit `main.py` to change the encryption key:

```python
def main():
    master_seed = 12345  # Change this value for different encryption
    ...
```

Different seeds = different encryption keys

---

## Deactivating Virtual Environment

When finished working:

```bash
# Deactivate virtual environment
deactivate
```

To reactivate later:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

---

## Advanced Setup

### Using Anaconda Distribution

**Advantage**: Pre-configured for scientific computing

```bash
# Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html

# Create environment from file (if provided)
conda env create -f environment.yml

# Or manually
conda create -n quantum-img python=3.10 numpy opencv qiskit
conda activate quantum-img
```

### GPU Acceleration (Future)

```bash
# For CUDA support (requires NVIDIA GPU)
conda install -c conda-forge cudatoolkit=11.8
pip install qiskit-aer-gpu
```

### Docker Container (Future)

```bash
# Build Docker image
docker build -t quantum-encryption .

# Run in container
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output quantum-encryption
```

---

## Troubleshooting

### Issue: "Python not found"

**Windows**:
```bash
# Use full path
C:\Python311\python.exe --version

# Or add Python to PATH and restart terminal
```

### Issue: "Module not found" after installation

```bash
# Verify virtual environment is activated
# Windows: prompt should show (.venv)
# Linux/Mac: prompt should show (.venv)

# Reinstall package
pip install --force-reinstall package_name
```

### Issue: Permission denied (Linux/macOS)

```bash
# Use sudo (not recommended)
sudo pip install -r requirements.txt

# Better: Use --user flag
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: OpenCV ImportError on macOS

```bash
# Try this solution
pip uninstall opencv-python
brew install opencv
pip install opencv-python
```

### Issue: Qiskit version conflicts

```bash
# Check installed version
pip show qiskit

# Uninstall and reinstall latest
pip uninstall qiskit qiskit-aer
pip install qiskit qiskit-aer --upgrade
```

---

## System Verification

### Full System Check Script

```bash
# Create check_system.py
python -c "
import sys
import numpy
import cv2
try:
    from qiskit import __version__ as qiskit_version
except:
    qiskit_version = 'Not installed'

print(f'Python version: {sys.version}')
print(f'NumPy version: {numpy.__version__}')
print(f'OpenCV version: {cv2.__version__}')
print(f'Qiskit version: {qiskit_version}')
print('System ready for quantum encryption pipeline!')
"
```

---

## Performance Optimization

### For Faster Processing

1. **Use SSD instead of HDD**
   - Significantly faster image I/O

2. **Allocate more RAM**
   - Process larger images simultaneously

3. **Use GPU** (when implemented)
   - Will provide 10-50x speedup

### Typical Performance

| Image Size | Processing Time | Blocks |
|------------|-----------------|--------|
| 512×512    | 0.3s           | 4,096  |
| 1024×1024  | 1.5s           | 16,384 |
| 1386×791   | 1.2s           | 14,985 |
| 2048×2048  | 6.0s           | 65,536 |

---

## Next Steps

1. **Place your satellite image in `input/` folder**
2. **Run `python main.py`**
3. **Check results in `output/` folder**
4. **Read ARCHITECTURE.md for technical details**
5. **See ROADMAP.md for future improvements**

---

## Support

### Getting Help

1. Check this installation guide
2. Review ARCHITECTURE.md for technical details
3. Open an issue on GitHub
4. Contact: govind.v.kartha@example.com

### Common Resources

- **Qiskit Documentation**: https://qiskit.org/documentation/
- **OpenCV Documentation**: https://docs.opencv.org/
- **NumPy Documentation**: https://numpy.org/doc/

---

## Uninstalling

### Complete Removal

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
# Windows
rmdir /s .venv

# Linux/macOS
rm -rf .venv

# Remove repository
# Windows
rmdir /s Quantum-Image-Encryption

# Linux/macOS
rm -rf Quantum-Image-Encryption
```

---

**Installation Guide Version**: 1.0  
**Last Updated**: January 30, 2026
