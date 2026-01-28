# Secure Image Encryption with AI & Quantum Computing

A hybrid encryption pipeline that integrates Deep Learning (Remote Sensing AI) with Quantum Computing to intelligently protect sensitive regions in satellite imagery.

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

## Directory Structure

```
.
├── repos/                          # Cloned repositories
│   ├── FlexiMo/                   # Repository A
│   └── Quantum-Image-Encryption/   # Repository B
├── bridge_controller/              # Core integration scripts
│   ├── pipeline.py                # Main bridge controller
│   ├── splitter.py                # Image splitting logic
│   ├── quantum_handler.py          # Quantum encryption wrapper
│   └── classical_handler.py        # Classical encryption wrapper
├── data/
│   ├── satellite_images/           # Input satellite data
│   └── output/                     # Encrypted results
├── tests/                          # Testing scripts
├── docs/                           # Documentation
└── requirements.txt                # Python dependencies
```

## Development Phases

### Phase 1: Satellite Integration (Current)
- Validate pipeline with satellite imagery
- Test Bridge Controller
- Verify quantum encryption output
- Status: In Progress

### Phase 2: Domain Adaptation (Future)
- Transfer learning for medical/general imagery
- Fine-tune FlexiMo for new domains
- Status: Planned

## Getting Started

1. Clone the project and repositories
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your environment
4. Run the bridge controller pipeline
5. Process satellite imagery

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
