# Phase 3: Dual-Engine Integration Guide

## Overview
This document outlines how to integrate the FlexiMo AI engine and Quantum encryption engine into the main pipeline.

## Step 1: Obtain Pre-trained Weights

### For FlexiMo (Intelligence Engine)
```bash
# Option 1: Download from HuggingFace
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('earthflow/DOFA', 'DOFA_ViT_base_e100.pth', local_dir='./models')"

# Option 2: Manual download
# Visit: https://huggingface.co/earthflow/DOFA/tree/main
# Download DOFA_ViT_base_e100.pth to ./models/
```

### For Quantum (Security Engine)
- Pre-trained weights not needed
- Uses Qiskit-aer simulator (already installed)

## Step 2: Update main.py Configuration

### Configuration Block
```python
# ========== DUAL-ENGINE CONFIGURATION ==========
INTELLIGENCE_ENGINE_CONFIG = {
    'enabled': True,
    'model_path': 'models/DOFA_ViT_base_e100.pth',  # Path to pre-trained weights
    'device': 'cpu',  # 'cpu' or 'cuda'
    'model_type': 'vit_base_patch16_224',
    'input_size': 224,
    'num_classes': 13,
}

SECURITY_ENGINE_CONFIG = {
    'enabled': True,
    'backend': 'qiskit-aer',
    'method': 'statevector',
    'use_quantum_gates': True,  # X, Z, SWAP gates for scrambling
    'use_neqr': True,  # NEQR encoding for quantum states
}

# =========== ENCRYPTION MODE ==========
# HYBRID: Use both AI and Quantum (production)
# QUANTUM_ONLY: Skip AI segmentation, use quantum on entire image
# DISABLED: Use fallback classical methods (debug only)
ENCRYPTION_MODE = 'HYBRID'
```

## Step 3: Integration Points in Pipeline

### Stage 1: AI Segmentation
```python
# OLD (Classical Fallback)
roi_mask = get_roi_mask_canny(image)

# NEW (Real AI)
from fleximo_integration import FlexiMoSegmentor
fleximo = FlexiMoSegmentor(
    model_path=INTELLIGENCE_ENGINE_CONFIG['model_path'],
    device=INTELLIGENCE_ENGINE_CONFIG['device']
)
roi_mask = fleximo.segment(image)
```

### Stage 3: ROI Encryption
```python
# OLD (Classical XOR)
encrypted_blocks, block_keys = encrypt_roi_blocks(roi_blocks, master_seed)

# NEW (Quantum NEQR + Gates)
from quantum_encryption import encrypt_roi_blocks_quantum
encrypted_blocks, block_keys = encrypt_roi_blocks_quantum(roi_blocks, master_seed)
```

## Step 4: Validation Checklist

### FlexiMo Engine
- [ ] Pre-trained weights file exists at configured path
- [ ] Model initializes without errors
- [ ] Produces valid segmentation masks (0-255 range)
- [ ] Handles variable input sizes via dynamic patching
- [ ] Processing time: ~1-5 seconds per image

### Quantum Engine
- [ ] Qiskit-aer simulator initializes correctly
- [ ] NEQR encoding produces valid quantum circuits
- [ ] Quantum gates execute without errors
- [ ] Statevector output converts to image correctly
- [ ] Processing time: ~10-30 seconds per block set (slower due to simulation)

### Integration
- [ ] Both engines work together without conflicts
- [ ] Master seed properly links AI and Quantum operations
- [ ] Encryption/decryption produces valid output
- [ ] Metrics calculated correctly with real encryption

## Step 5: Performance Expectations

### Processing Timeline
```
Stage 1 (AI Segmentation):     ~3 seconds (was ~0.01s)
Stage 2 (ROI Extraction):      ~0.1 seconds
Stage 3 (Quantum Encryption):  ~15 seconds (was ~0.4s)
Stage 4 (Background Cipher):   ~0.1 seconds
Stage 5 (Reconstruction):      ~0.1 seconds
Stage 6 (Decryption):          ~15 seconds
────────────────────────────────────────────────
Total:                          ~33 seconds (vs ~1.2s before)
```

### Quality Metrics
```
PSNR:    13-26 dB (real quantum encryption may not achieve perfect reconstruction)
SSIM:    0.6-0.9 (depends on quantum simulation fidelity)
```

## Step 6: Error Handling

### If FlexiMo weights missing:
```
RuntimeError: "Pre-trained weights not found at {path}"
→ Download from HuggingFace or disable INTELLIGENCE_ENGINE_CONFIG['enabled']
```

### If Qiskit not installed:
```
ImportError: "Qiskit not available"
→ pip install qiskit qiskit-aer
```

### If both engines fail:
```
# Pipeline should FAIL HARD (no fallback to classical)
raise RuntimeError("Dual-engine configuration failed. Cannot proceed without AI and Quantum engines.")
```

## Next Steps

1. Download FlexiMo pre-trained weights
2. Update main.py with configuration block
3. Update Stage 1 and Stage 3 implementations
4. Run validation tests
5. Document performance metrics
6. Commit Phase 3 completion
