"""
Decryption Workflow - Reverse 6-Layer Pipeline

Reads metadata, decrypts ROI blocks (quantum NEQR) and BG (AES-256-GCM),
then reconstructs the original image.
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

from engines.preprocessing_engine import PreprocessingEngine
from engines.decision_engine import DecisionEngine
from engines.quantum_engine import QuantumEngine
from engines.classical_engine import ClassicalEngine
from engines.metadata_engine import MetadataEngine
from engines.verification_engine import VerificationEngine


# ====================================================================
# Helpers
# ====================================================================

def load_config(path: str = "config.json") -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    lc = config.get('logging', {})
    level = getattr(logging, lc.get('level', 'INFO'))
    fmt   = lc.get('format',
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger('decryption_orchestrator')
    if logger.handlers:
        return logger
    logger.setLevel(level)

    if lc.get('console_output', True):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)

    log_file = 'logs/decryption.log'
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    return logger


# ====================================================================
# Main orchestration
# ====================================================================

def orchestrate_decryption(
    encrypted_image_path: str,
    metadata_path: str,
    config_path: str = "config.json",
    output_dir: str = "output/decrypted",
) -> Dict[str, Any]:
    """
    Run the full decryption pipeline.

    Steps:
      1. Load metadata  (seeds, block_map, AES bundle, encrypted ROI blocks)
      2. Decrypt ROI blocks via quantum NEQR inverse
      3. Decrypt BG via AES-256-GCM
      4. Reconstruct image using DecisionEngine
      5. Verify quality
      6. Save output

    Returns
    -------
    result : dict
    """
    t0 = time.time()
    phase_times: Dict[str, float] = {}

    config = load_config(config_path)
    logger = setup_logging(config)

    logger.info("=" * 80)
    logger.info("6-LAYER HYBRID QUANTUM-CLASSICAL IMAGE DECRYPTION")
    logger.info("=" * 80)

    result: Dict[str, Any] = {
        'success': False,
        'error': None,
        'decrypted_image': None,
        'metrics': {},
        'processing_time': 0,
        'phase_times': {},
    }

    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # ==============================================================
        # STEP 1 - Load metadata
        # ==============================================================
        ps = time.time()
        logger.info("\n[STEP 1] Loading encryption metadata")
        me = MetadataEngine(config)
        me.initialize()
        meta = me.load(metadata_path)

        master_seed  = meta['master_seed']
        shots        = meta['shots']
        image_shape  = meta['image_shape']
        block_map    = meta['block_map']
        roi_mask     = meta['roi_mask']
        bg_bundle    = meta['bg_encrypted_bundle']
        enc_roi      = meta['encrypted_roi_blocks']

        logger.info(f"  master_seed={master_seed}, shots={shots}")
        logger.info(f"  image_shape={image_shape}, ROI blocks={len(enc_roi)}")
        phase_times['1_load_metadata'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['1_load_metadata']:.3f}s")

        # ==============================================================
        # STEP 2 - Decrypt ROI blocks (Quantum NEQR inverse)
        # ==============================================================
        ps = time.time()
        logger.info(f"\n[STEP 2] Quantum NEQR Decryption ({len(enc_roi)} ROI blocks)")
        qe = QuantumEngine(config)
        # Override shots from metadata
        qe.shots = shots
        qe.initialize()

        def _progress(cur, tot):
            logger.info(f"    Quantum progress: {cur}/{tot} blocks ({cur/tot*100:.0f}%)")

        decrypted_roi = qe.decrypt_blocks(enc_roi, master_seed,
                                           progress_callback=_progress)
        phase_times['2_quantum_dec'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['2_quantum_dec']:.3f}s")

        # ==============================================================
        # STEP 3 - Decrypt BG (AES-256-GCM)
        # ==============================================================
        ps = time.time()
        logger.info("\n[STEP 3] Classical AES-256-GCM Decryption (background)")
        ce = ClassicalEngine(config)
        ce.initialize()
        decrypted_bg = ce.decrypt(bg_bundle, master_seed)
        phase_times['3_classical_dec'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['3_classical_dec']:.3f}s")

        # ==============================================================
        # STEP 4 - Reconstruct image
        # ==============================================================
        ps = time.time()
        logger.info("\n[STEP 4] Reconstructing image (ROI blocks + BG)")
        dec_eng = DecisionEngine(config)
        dec_eng.initialize()
        decrypted_image = dec_eng.reconstruct_image(
            decrypted_roi, decrypted_bg, block_map, image_shape)
        phase_times['4_reconstruct'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['4_reconstruct']:.3f}s")

        # ==============================================================
        # STEP 5 - Save decrypted image
        # ==============================================================
        ps = time.time()
        logger.info("\n[STEP 5] Saving decrypted image")
        prep = PreprocessingEngine(config)
        prep.initialize()
        dec_img_path = str(out_path / 'decrypted_image.png')
        prep.save_image(decrypted_image, dec_img_path)
        phase_times['5_save'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['5_save']:.3f}s")

        # ==============================================================
        # STEP 6 - Verification (if original available)
        # ==============================================================
        ps = time.time()
        logger.info("\n[STEP 6] Verification")
        ve = VerificationEngine(config)
        ve.initialize()
        # We can only compare with original if we have it
        # For now log basic stats
        logger.info(f"  Decrypted shape: {decrypted_image.shape}")
        logger.info(f"  Pixel range: [{decrypted_image.min()}, {decrypted_image.max()}]")
        result['metrics']['decrypted_shape'] = list(decrypted_image.shape)
        result['metrics']['pixel_range'] = [int(decrypted_image.min()), int(decrypted_image.max())]
        phase_times['6_verify'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['6_verify']:.3f}s")

        # ==============================================================
        # Done
        # ==============================================================
        elapsed = time.time() - t0
        result['success'] = True
        result['decrypted_image'] = decrypted_image
        result['processing_time'] = elapsed
        result['phase_times'] = phase_times
        result['decrypted_image_path'] = dec_img_path

        logger.info("\n" + "=" * 80)
        logger.info(f"[SUCCESS] DECRYPTION COMPLETE in {elapsed:.2f}s")
        logger.info("-" * 80)
        for k, v in sorted(phase_times.items()):
            pct = v / elapsed * 100 if elapsed else 0
            logger.info(f"  {k:.<35} {v:>8.3f}s ({pct:>5.1f}%)")
        logger.info("-" * 80)
        logger.info(f"  {'Total':.<35} {elapsed:>8.2f}s (100.0%)")
        logger.info("=" * 80)

        return result

    except Exception as e:
        elapsed = time.time() - t0
        result['error'] = str(e)
        result['processing_time'] = elapsed
        logger.error(f"\n[FAILED] {e}")
        import traceback
        logger.error(traceback.format_exc())
        return result


# ====================================================================
# CLI
# ====================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m workflows.decrypt <encrypted_image> <metadata_json> [config] [output_dir]")
        sys.exit(1)

    enc_path  = sys.argv[1]
    meta_path = sys.argv[2]
    cfg_path  = sys.argv[3] if len(sys.argv) > 3 else "config.json"
    out_dir   = sys.argv[4] if len(sys.argv) > 4 else "output/decrypted"

    result = orchestrate_decryption(enc_path, meta_path, cfg_path, out_dir)
    if result['success']:
        print("\nDecryption successful!")
    else:
        print(f"\nDecryption failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
