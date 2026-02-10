"""
Encryption Workflow - 6-Layer Hybrid Quantum-Classical Pipeline

Architecture:
  Layer 1  PreprocessingEngine   - load & validate image
  Layer 2  AIEngine              - FlexiMo ROI / BG segmentation
  Layer 3  DecisionEngine        - split ROI 8x8 blocks + BG
  Layer 4  QuantumEngine         - NEQR encryption on ROI blocks  (Repo B)
  Layer 5  ClassicalEngine       - AES-256-GCM on background     (cryptography)
  Layer 6  FusionEngine          - merge encrypted ROI + BG
         + MetadataEngine        - persist everything for decryption
         + VerificationEngine    - quality checks
"""

import json
import logging
import secrets
import time
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

from engines.preprocessing_engine import PreprocessingEngine
from engines.ai_engine import AIEngine
from engines.decision_engine import DecisionEngine
from engines.quantum_engine import QuantumEngine
from engines.classical_engine import ClassicalEngine
from engines.fusion_engine import FusionEngine
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

    logger = logging.getLogger('orchestrator')
    if logger.handlers:
        return logger
    logger.setLevel(level)

    if lc.get('console_output', True):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)

    log_file = lc.get('file_output')
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)

    return logger


# ====================================================================
# Main orchestration
# ====================================================================

def orchestrate_encryption(
    image_path: str,
    config_path: str = "config.json",
) -> Dict[str, Any]:
    """
    Run the full 6-layer encryption pipeline.

    Returns
    -------
    result : dict  with keys success, error, metrics, processing_time, ...
    """
    t0 = time.time()
    phase_times: Dict[str, float] = {}

    config = load_config(config_path)
    logger = setup_logging(config)

    logger.info("=" * 80)
    logger.info("6-LAYER HYBRID QUANTUM-CLASSICAL IMAGE ENCRYPTION")
    logger.info("=" * 80)

    result: Dict[str, Any] = {
        'success': False,
        'error': None,
        'metrics': {},
        'processing_time': 0,
        'phase_times': {},
    }

    try:
        # ---- output paths ----
        stem = Path(image_path).stem
        out_root    = Path('output')
        enc_dir     = out_root / f"{stem}_01_encrypted"
        dec_dir     = out_root / f"{stem}_02_decrypted"
        meta_dir    = out_root / 'metadata'
        for d in (enc_dir, dec_dir, meta_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ==============================================================
        # LAYER 1 - Preprocessing
        # ==============================================================
        ps = time.time()
        logger.info("\n[LAYER 1] Preprocessing")
        prep = PreprocessingEngine(config)
        prep.initialize()
        image = prep.load_image(image_path)
        assert prep.validate(image), "Image validation failed"
        phase_times['L1_preprocess'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['L1_preprocess']:.3f}s")

        # ==============================================================
        # LAYER 2 - AI Segmentation (FlexiMo / fallback)
        # ==============================================================
        ps = time.time()
        logger.info("\n[LAYER 2] AI Segmentation (FlexiMo / contrast fallback)")
        ai = AIEngine(config)
        ai.initialize()
        roi_mask = ai.generate_roi_mask(image)
        phase_times['L2_ai_segment'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['L2_ai_segment']:.3f}s")

        # ==============================================================
        # LAYER 3 - Decision: ROI / BG split + 8x8 blocking
        # ==============================================================
        ps = time.time()
        logger.info("\n[LAYER 3] Decision Engine - ROI/BG split (8x8 blocks)")
        dec_eng = DecisionEngine(config)
        dec_eng.initialize()
        split = dec_eng.separate_roi_bg(image, roi_mask)
        roi_blocks = split['roi_blocks']
        block_map  = split['block_map']
        bg_image   = split['bg_image']
        logger.info(f"  ROI blocks: {len(roi_blocks)}, BG shape: {bg_image.shape}")
        phase_times['L3_decision'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['L3_decision']:.3f}s")

        # ==============================================================
        # Generate master seed
        # ==============================================================
        master_seed = secrets.randbelow(2**31 - 1)
        logger.info(f"\n  Master seed (secure random): {master_seed}")

        # ==============================================================
        # LAYER 4 - Quantum Encryption (NEQR - Repo B)
        # ==============================================================
        ps = time.time()
        logger.info(f"\n[LAYER 4] Quantum NEQR Encryption ({len(roi_blocks)} ROI blocks)")
        qe = QuantumEngine(config)
        qe.initialize()

        def _progress(cur, tot):
            logger.info(f"    Quantum progress: {cur}/{tot} blocks ({cur/tot*100:.0f}%)")

        encrypted_roi = qe.encrypt_blocks(roi_blocks, master_seed,
                                           progress_callback=_progress)
        phase_times['L4_quantum_enc'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['L4_quantum_enc']:.3f}s")

        # ==============================================================
        # LAYER 5 - Classical AES-256-GCM (background)
        # ==============================================================
        ps = time.time()
        logger.info("\n[LAYER 5] Classical AES-256-GCM (background)")
        ce = ClassicalEngine(config)
        ce.initialize()
        bg_bundle = ce.encrypt(bg_image, master_seed)
        phase_times['L5_classical_enc'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['L5_classical_enc']:.3f}s")

        # ==============================================================
        # LAYER 6 - Fusion
        # ==============================================================
        ps = time.time()
        logger.info("\n[LAYER 6] Fusion - merge encrypted ROI + BG")
        fe = FusionEngine(config)
        fe.initialize()
        bg_visual = fe.create_bg_visual(bg_image, master_seed)
        encrypted_image = fe.fuse_encrypted(
            encrypted_roi, bg_visual, block_map, image.shape)
        phase_times['L6_fusion'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['L6_fusion']:.3f}s")

        # ==============================================================
        # Metadata
        # ==============================================================
        ps = time.time()
        logger.info("\n[META] Saving encryption metadata")
        me = MetadataEngine(config)
        me.initialize()
        metadata_path = str(meta_dir / 'encryption_metadata.json')
        me.save(
            metadata_path,
            master_seed=master_seed,
            shots=qe.shots,
            image_shape=image.shape,
            block_map=block_map,
            roi_mask=roi_mask,
            bg_encrypted_bundle=bg_bundle,
            encrypted_roi_blocks=encrypted_roi,
        )
        phase_times['metadata'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['metadata']:.3f}s")

        # ==============================================================
        # Save encrypted image
        # ==============================================================
        ps = time.time()
        logger.info("\n[SAVE] Saving encrypted image")
        enc_path = str(enc_dir / 'encrypted_image.png')
        prep.save_image(encrypted_image, enc_path)
        phase_times['save_enc'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['save_enc']:.3f}s")

        # ==============================================================
        # Verification
        # ==============================================================
        ps = time.time()
        logger.info("\n[VERIFY] Encryption quality check")
        ve = VerificationEngine(config)
        ve.initialize()
        enc_metrics = ve.verify_encryption(image, encrypted_image)
        result['metrics'].update(enc_metrics)
        phase_times['verify'] = time.time() - ps
        logger.info(f"  [TIME] {phase_times['verify']:.3f}s")

        # ==============================================================
        # Done
        # ==============================================================
        elapsed = time.time() - t0
        result['success'] = True
        result['processing_time'] = elapsed
        result['phase_times'] = phase_times
        result['encrypted_image_path'] = enc_path
        result['metadata_path'] = metadata_path
        result['encrypted_dir'] = str(enc_dir)
        result['decrypted_dir'] = str(dec_dir)
        result['input_filename_stem'] = stem

        logger.info("\n" + "=" * 80)
        logger.info(f"[SUCCESS] ENCRYPTION COMPLETE in {elapsed:.2f}s")
        logger.info("-" * 80)
        for k, v in sorted(phase_times.items()):
            pct = v / elapsed * 100 if elapsed else 0
            logger.info(f"  {k:.<35} {v:>8.3f}s ({pct:>5.1f}%)")
        logger.info("-" * 80)
        logger.info(f"  {'Total':.<35} {elapsed:>8.2f}s (100.0%)")
        logger.info("=" * 80)

        # ---- automatic decryption ----
        logger.info("\n\nAUTOMATICALLY STARTING DECRYPTION...\n")
        try:
            from workflows.decrypt import orchestrate_decryption

            dec_result = orchestrate_decryption(
                enc_path, metadata_path, config_path, str(dec_dir))

            if dec_result.get('success'):
                result['decryption'] = dec_result
                result['full_cycle'] = True
                total = elapsed + dec_result.get('processing_time', 0)
                logger.info(f"\n[FULL CYCLE] Encrypt + Decrypt = {total:.2f}s")
        except Exception as ex:
            logger.warning(f"Auto-decryption skipped: {ex}")

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
    image_path = None

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        input_dir = Path("input")
        for name in ("st1.png", "test_image.png"):
            p = input_dir / name
            if p.exists():
                image_path = str(p)
                break
        if image_path is None:
            pngs = list(input_dir.glob("*.png"))
            jpgs = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
            imgs = pngs + jpgs
            if imgs:
                image_path = str(imgs[0])
                print(f"Auto-detected: {image_path}")
            else:
                print("ERROR: No images found in input/")
                sys.exit(1)

    if not Path(image_path).exists():
        print(f"ERROR: {image_path} not found")
        sys.exit(1)

    result = orchestrate_encryption(image_path)
    if result['success']:
        print("\nPipeline completed successfully!")
    else:
        print(f"\nPipeline failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
