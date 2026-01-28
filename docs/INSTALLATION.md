# Installation & Setup Guide

## Prerequisites

- **Python 3.9+** (tested on 3.10, 3.11)
- **CUDA 11.8+** (for GPU acceleration, optional)
- **Git** (for cloning repositories)
- **4GB+ RAM** (minimum), 8GB+ recommended
- **2GB+ Storage** (for dependencies and models)

## Quick Start (5 Minutes)

### 1. Clone the Main Project
```bash
git clone https://github.com/your-org/image-security-ieee.git
cd image-security-ieee
```

### 2. Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Tests
```bash
python tests/test_pipeline.py
```

### 5. Process Your First Image
```bash
python -c "
from bridge_controller import BridgeController
bridge = BridgeController(project_dir='.')
results = bridge.process_image_with_segmentation(
    'data/satellite_images/your_image.png',
    'data/satellite_images/your_mask.png'
)
print(f'Encrypted image saved to: {results[\"files\"][\"final_encrypted\"]}')
"
```

---

## Detailed Installation

### Step 1: Environment Setup

#### Windows
```powershell
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Verify
python --version  # Should be 3.9+
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Verify
python --version  # Should be 3.9+
```

### Step 2: Install Core Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install NumPy, SciPy (foundation)
pip install numpy scipy scikit-learn scikit-image

# Install PyTorch (CPU or GPU)
# For CPU:
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html

# For GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install AI/ML Dependencies

```bash
# Vision Transformers and timm
pip install timm einops transformers kornia

# Image processing
pip install opencv-python pillow
pip install rasterio gdal  # For geospatial data
```

### Step 4: Install Quantum Computing Dependencies

```bash
# Qiskit ecosystem
pip install qiskit qiskit-aer qiskit-ibm-runtime

# Visualization (optional but recommended)
pip install matplotlib jupyter
```

### Step 5: Install Development Tools

```bash
# Testing
pip install pytest pytest-cov

# Code quality
pip install black flake8

# Documentation
pip install sphinx sphinx-rtd-theme
```

### Step 6: Verify Installation

```python
# Run this script to verify all dependencies
python -c "
import sys
print('Python:', sys.version)

# Core
import numpy as np
print('✓ NumPy:', np.__version__)

import scipy
print('✓ SciPy:', scipy.__version__)

# ML
import torch
print('✓ PyTorch:', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())

# Vision
import cv2
import kornia
print('✓ OpenCV & Kornia')

# Quantum
import qiskit
print('✓ Qiskit:', qiskit.__version__)

print('\n✓ All core dependencies installed!')
"
```

---

## Repository Integration

### Cloning Submodules

The repositories have already been cloned into `repos/`:

```bash
cd repos/

# Verify FlexiMo
cd FlexiMo
pip install -e .
cd ..

# Verify Quantum-Image-Encryption
cd Quantum-Image-Encryption
pip install -e .
cd ..
```

### Using FlexiMo

```python
from fleximo import FlexiMo
import torch

# Load pretrained model
model = FlexiMo.from_pretrained(
    model_name="vit_base_patch16_32",
    head_type="upernet",
    pretrained=True
)
model.eval()

# For GPU
model = model.cuda()

# Process image
image = torch.randn(1, 3, 512, 512)  # Batch of 1, RGB, 512x512
with torch.no_grad():
    segmentation_mask = model(image)

print(f"Mask shape: {segmentation_mask.shape}")  # [1, num_classes, 512, 512]
```

### Using Quantum-Image-Encryption Modules

```python
import sys
sys.path.insert(0, 'repos/Quantum-Image-Encryption')

# Import quantum encryption modules
from quantum import quantum_encryption_pipeline
from chaos import chaos_map_encryption

# These are wrapped by our BridgeController
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Python path
PYTHONPATH=.:./repos/FlexiMo:./repos/Quantum-Image-Encryption

# Model cache
TORCH_HOME=./models
HF_HOME=./huggingface_cache

# Quantum backend
QISKIT_SETTINGS_URL=https://quantum.ibm.com

# Logging
LOG_LEVEL=INFO
VERBOSE=1
```

### Configuration File

Create `config/settings.yaml`:

```yaml
# Quantum Encryption Settings
quantum:
  backend: "qasm_simulator"  # or "statevector_simulator"
  enable_noise: false
  shots: 1024
  arnold_iterations: 100
  encode_depth: 8

# Classical Encryption Settings
classical:
  chaos_param_r: 3.99
  hlsm_iterations: 1000000
  seed_precision: 64

# Model Settings
model:
  architecture: "vit_base_patch16_32"
  head_type: "upernet"
  pretrained: true
  device: "cuda"  # or "cpu"

# Processing Settings
processing:
  image_format: "RGB"
  max_image_size: 2048
  tile_size: 512
  num_workers: 4

# Output Settings
output:
  save_intermediate: true
  compression: "lossless"
  metadata_format: "json"
```

### Load Configuration

```python
import yaml
from bridge_controller import BridgeController

with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

bridge = BridgeController(
    project_dir=".",
    quantum_backend=config["quantum"]["backend"],
    verbose=True
)
```

---

## GPU Acceleration (Optional)

### NVIDIA CUDA Setup

1. **Install CUDA Toolkit 11.8**
   - https://developer.nvidia.com/cuda-11-8-0-download-archive

2. **Install cuDNN 8.6+**
   - https://developer.nvidia.com/cudnn

3. **Set Environment Variables**
   ```bash
   set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
   set PATH=%CUDA_HOME%\bin;%PATH%
   ```

4. **Verify CUDA**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Your GPU name
   ```

### PyTorch GPU Acceleration

```python
import torch
from bridge_controller import BridgeController
from bridge_controller.splitter import ImageSplitter

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Use GPU in your code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models automatically move to GPU with .cuda()
model = FlexiMo.from_pretrained("vit_base_patch16_32").cuda()
```

---

## Docker Setup (Advanced)

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/repos/FlexiMo:/app/repos/Quantum-Image-Encryption

ENTRYPOINT ["python"]
```

### Build and Run

```bash
# Build
docker build -t image-security:latest .

# Run
docker run --gpus all -v /path/to/data:/app/data image-security:latest \
  -c "
    from bridge_controller import BridgeController
    bridge = BridgeController()
    results = bridge.process_image_with_segmentation(
        'data/image.png',
        'data/mask.png'
    )
  "
```

---

## Troubleshooting

### Issue: ImportError for bridge_controller

**Problem**: `ModuleNotFoundError: No module named 'bridge_controller'`

**Solution**:
```bash
# Add to Python path
set PYTHONPATH=%cd%;%PYTHONPATH%  # Windows
export PYTHONPATH=$PWD:$PYTHONPATH  # macOS/Linux

# Or install in development mode
pip install -e .
```

### Issue: Qiskit Backend Not Available

**Problem**: `BackendV2Error: The device does not support the qasm_simulator`

**Solution**:
```bash
# Install Qiskit Aer
pip install qiskit-aer

# Or use statevector simulator
bridge = BridgeController(quantum_backend="statevector_simulator")
```

### Issue: CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Reduce image size
image = cv2.resize(image, (512, 512))

# Reduce batch size
# Process one image at a time

# Clear GPU cache
import torch
torch.cuda.empty_cache()
```

### Issue: FlexiMo Model Download Fails

**Problem**: Connection timeout when downloading pretrained model

**Solution**:
```bash
# Manually download model
cd models/
wget https://github.com/danfenghong/IEEE_TGRS_Fleximo/releases/download/...

# Set cache path
export TORCH_HOME=./models
```

---

## Next Steps

1. **Read Architecture Guide**: `docs/ARCHITECTURE.md`
2. **Run Demo**: `python tests/test_pipeline.py`
3. **Process Your Data**: Place satellite images in `data/satellite_images/`
4. **Integrate FlexiMo**: Generate masks using FlexiMo model
5. **Encrypt Images**: Run bridge controller on your data
6. **Monitor Results**: Check `output/` for encrypted images

---

## Support

- **Issues**: Check GitHub issues or create a new one
- **Documentation**: See `docs/` directory
- **Examples**: Check `tests/test_pipeline.py` for usage examples
- **Community**: Discussion forum (if available)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-27 | Initial release - Phase 1 |
| 0.9.0 | 2026-01-20 | Beta testing |
| 0.5.0 | 2026-01-10 | Alpha version |
