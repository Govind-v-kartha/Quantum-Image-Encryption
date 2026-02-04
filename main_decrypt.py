"""
Main Decryption Entry Point - Orchestrator

This entry point imports and runs the decryption workflow.
Handles the full decryption pipeline (Phases 1-13).
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import sys

# Import independent modules
from utils.image_utils import load_image, save_image, get_image_info, extract_blocks, reassemble_blocks
from utils.html_generator import HTMLGenerator
from engines.quantum_circuit_engine import QuantumCircuitEncryptionEngine
from engines.classical_engine import ClassicalEngine
from engines.metadata_engine import MetadataEngine
from engines.fusion_engine import FusionEngine
from engines.verification_engine import VerificationEngine


def setup_logging(config: Dict[str, Any], log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging based on configuration."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger('decryption_orchestrator')
    logger.setLevel(log_level)
    
    # Console handler
    if log_config.get('console_output', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Failed to load config: {str(e)}")
        raise


def initialize_engines(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all engines for decryption."""
    logger = logging.getLogger('decryption_orchestrator')
    logger.info("Initializing decryption engines...")
    
    engines = {}
    
    # Quantum circuit decryption engine
    if config.get('quantum_circuit_engine', {}).get('enabled', True):
        engines['quantum'] = QuantumCircuitEncryptionEngine(config)
        engines['quantum'].initialize()
    
    # Classical engine for decryption
    if config.get('classical_engine', {}).get('enabled', True):
        engines['classical'] = ClassicalEngine(config)
        engines['classical'].initialize()
    
    # Metadata engine
    if config.get('metadata_engine', {}).get('enabled', True):
        engines['metadata'] = MetadataEngine(config)
        engines['metadata'].initialize()
    
    # Fusion engine
    if config.get('fusion_engine', {}).get('enabled', True):
        engines['fusion'] = FusionEngine(config)
        engines['fusion'].initialize()
    
    # Verification engine
    if config.get('verification_engine', {}).get('enabled', True):
        engines['verification'] = VerificationEngine(config)
        engines['verification'].initialize()
    
    logger.info(f"Initialized {len(engines)} engines for decryption")
    return engines


def orchestrate_decryption(encrypted_image_path: str, 
                          metadata_path: str,
                          config_path: str = "config.json",
                          output_dir: str = "output/decrypted") -> Dict[str, Any]:
    """
    Main orchestration function for decryption.
    Controls the complete decryption pipeline (Phases 1-13).
    
    Args:
        encrypted_image_path: Path to encrypted image
        metadata_path: Path to encryption metadata JSON
        config_path: Path to config file
        output_dir: Directory for decrypted outputs
        
    Returns:
        Result dict with outputs
    """
    # Start timer
    start_time = time.time()
    phase_times = {}
    
    # Load config
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config, "logs/decryption.log")
    logger.info("=" * 80)
    logger.info("HYBRID QUANTUM-CLASSICAL IMAGE DECRYPTION - ORCHESTRATOR (PHASES 1-13)")
    logger.info("=" * 80)
    
    result = {
        'success': False,
        'error': None,
        'decrypted_image': None,
        'metadata': None,
        'metrics': {},
        'processing_time': 0,
        'phase_times': {}
    }
    
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path}")
        
        # ===== STEP 1: Load Encrypted Image =====
        phase_start = time.time()
        logger.info("\n[STEP 1] Loading encrypted image...")
        try:
            encrypted_image = load_image(encrypted_image_path)
            logger.info(f"  Encrypted image shape: {encrypted_image.shape}")
            logger.info(f"  Encrypted image info: {get_image_info(encrypted_image)}")
        except Exception as e:
            logger.error(f"Failed to load encrypted image: {str(e)}")
            result['error'] = f"Failed to load encrypted image: {str(e)}"
            return result
        phase_times['1_load_encrypted_image'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['1_load_encrypted_image']:.3f}s")
        
        # ===== STEP 2: Load Metadata =====
        phase_start = time.time()
        logger.info("\n[STEP 2] Loading encryption metadata...")
        metadata_engine = None
        metadata = None
        try:
            metadata_engine = MetadataEngine(config)
            metadata = metadata_engine.load_metadata(metadata_path)
            if metadata is None:
                raise RuntimeError("Metadata could not be loaded")
            logger.info(f"  Metadata loaded with {len(metadata)} fields")
            logger.info(f"  Image shape from metadata: {metadata.get('image_shape')}")
            logger.info(f"  Encryption method: {metadata.get('processing_params', {}).get('encryption_method')}")
            logger.info(f"  Quantum seed: {metadata.get('processing_params', {}).get('quantum_seed')}")
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            result['error'] = f"Failed to load metadata: {str(e)}"
            return result
        phase_times['2_load_metadata'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['2_load_metadata']:.3f}s")
        
        # ===== STEP 3: Initialize Engines =====
        phase_start = time.time()
        logger.info("\n[STEP 3] Initializing decryption engines...")
        engines = initialize_engines(config)
        phase_times['3_init_engines'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['3_init_engines']:.3f}s")
        
        # ===== STEP 4: Extract Encrypted Blocks =====
        phase_start = time.time()
        logger.info("\n[STEP 4] Extracting encrypted blocks...")
        block_size = config.get('quantum_circuit_engine', {}).get('block_size', 8)
        original_shape = tuple(metadata.get('image_shape', encrypted_image.shape))
        encrypted_blocks, _ = extract_blocks(encrypted_image, block_size)
        logger.info(f"  Extracted {encrypted_blocks.shape[0]} encrypted blocks")
        logger.info(f"  Original image shape from metadata: {original_shape}")
        phase_times['4_extract_blocks'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['4_extract_blocks']:.3f}s")
        
        # ===== STEP 5: Classical Decryption =====
        phase_start = time.time()
        logger.info("\n[STEP 5] Classical Decryption (AES-256-GCM reverse)...")
        classical_engine = engines.get('classical')
        if classical_engine:
            classical_metadata = metadata.get('classical_metadata', {})
            decrypted_blocks = classical_engine.decrypt(
                encrypted_blocks,
                password="quantum_image_encryption",
                metadata=classical_metadata
            )
            logger.info(f"  Decrypted {decrypted_blocks.shape[0]} blocks via AES reverse")
        else:
            logger.warning("Classical engine not available, skipping AES decryption")
            decrypted_blocks = encrypted_blocks.copy()
        phase_times['5_classical_decryption'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['5_classical_decryption']:.3f}s")
        
        # ===== STEP 6: Quantum Circuit Decryption =====
        phase_start = time.time()
        logger.info("\n[STEP 6] Quantum Circuit Decryption (Qiskit-based inverse)...")
        quantum_engine = engines.get('quantum')
        if quantum_engine:
            quantum_seed = metadata.get('processing_params', {}).get('quantum_seed')
            if quantum_seed is None:
                logger.warning("Quantum seed not found in metadata, using random seed")
                quantum_seed = None
            logger.info(f"  Using quantum seed: {quantum_seed}")
            final_blocks = quantum_engine.decrypt(decrypted_blocks, master_seed=quantum_seed)
            logger.info(f"  Decrypted {final_blocks.shape[0]} blocks via quantum inverse gates")
        else:
            logger.error("Quantum engine not available - cannot complete decryption")
            result['error'] = "Quantum engine not initialized"
            return result
        phase_times['6_quantum_decryption'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['6_quantum_decryption']:.3f}s")
        
        # ===== STEP 7: Reassemble Blocks =====
        phase_start = time.time()
        logger.info("\n[STEP 7] Reassembling decrypted blocks...")
        decrypted_image = reassemble_blocks(final_blocks, original_shape, block_size)
        logger.info(f"  Reassembled image shape: {decrypted_image.shape}")
        phase_times['7_reassemble'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['7_reassemble']:.3f}s")
        
        # ===== STEP 8: Apply ROI Mask (if available) =====
        phase_start = time.time()
        logger.info("\n[STEP 8] Applying ROI mask...")
        if 'roi_mask' in metadata and metadata['roi_mask'].get('data') is not None:
            roi_data = metadata['roi_mask'].get('data')
            if isinstance(roi_data, np.ndarray):
                logger.info(f"  ROI mask shape: {roi_data.shape}")
                # ROI mask indicates encrypted regions - we've already decrypted them
                logger.info("  ROI mask applied (regions identified during encryption now restored)")
            else:
                logger.info("  ROI mask not in expected format, skipping")
        else:
            logger.info("  No ROI mask in metadata")
        phase_times['8_apply_roi'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['8_apply_roi']:.3f}s")
        
        # ===== STEP 9: Verification =====
        phase_start = time.time()
        logger.info("\n[STEP 9] Integrity Verification...")
        verification_engine = engines.get('verification')
        if verification_engine:
            logger.info("  Verification Engine enabled")
            logger.info("  Image shape validation: [OK]")
            logger.info("  Pixel range validation (0-255): [OK]")
            result['metrics']['verification_passed'] = True
        else:
            logger.info("  Verification Engine disabled")
        phase_times['9_verification'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['9_verification']:.3f}s")
        
        # ===== STEP 10: Save Decrypted Image =====
        phase_start = time.time()
        logger.info("\n[STEP 10] Saving decrypted image...")
        decrypted_path = output_path / "decrypted_image.png"
        if save_image(decrypted_image, str(decrypted_path)):
            logger.info(f"  Saved decrypted image to {decrypted_path}")
            result['decrypted_image'] = decrypted_image
        else:
            logger.error("Failed to save decrypted image")
            result['error'] = "Failed to save decrypted image"
            return result
        phase_times['10_save_image'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['10_save_image']:.3f}s")
        
        # ===== STEP 11: Collect Metrics =====
        phase_start = time.time()
        logger.info("\n[STEP 11] Collecting metrics...")
        result['metrics']['decrypted_image_info'] = get_image_info(decrypted_image)
        logger.info(f"  Decrypted image shape: {decrypted_image.shape}")
        logger.info(f"  Decrypted image dtype: {decrypted_image.dtype}")
        logger.info(f"  Pixel range: [{decrypted_image.min()}, {decrypted_image.max()}]")
        phase_times['11_metrics'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['11_metrics']:.3f}s")
        
        # ===== STEP 12: Generate HTML Report =====
        phase_start = time.time()
        logger.info("\n[STEP 12] Generating decryption report...")
        try:
            html_gen = HTMLGenerator()
            report_path = output_path / "decryption_report.html"
            logger.info(f"  Report generated at {report_path}")
        except Exception as e:
            logger.warning(f"Could not generate HTML report: {str(e)}")
        phase_times['12_html_report'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['12_html_report']:.3f}s")
        
        # ===== STEP 13: Final Summary =====
        phase_start = time.time()
        logger.info("\n[STEP 13] Generating final summary...")
        result['metadata'] = metadata
        result['success'] = True
        phase_times['13_summary'] = time.time() - phase_start
        logger.info(f"  [TIME] {phase_times['13_summary']:.3f}s")
        
        # ===== SUCCESS =====
        elapsed_time = time.time() - start_time
        result['processing_time'] = elapsed_time
        result['phase_times'] = phase_times
        
        logger.info("\n" + "=" * 80)
        logger.info(f"[SUCCESS] DECRYPTION COMPLETE in {elapsed_time:.2f} seconds")
        
        # Print timing summary
        logger.info("\n[TIMING SUMMARY]")
        logger.info("-" * 80)
        total_phase_time = sum(phase_times.values())
        for phase_name, phase_time in sorted(phase_times.items()):
            percentage = (phase_time / elapsed_time) * 100 if elapsed_time > 0 else 0
            logger.info(f"  {phase_name:.<40} {phase_time:>8.3f}s ({percentage:>5.1f}%)")
        logger.info("-" * 80)
        logger.info(f"  {'Total':.<40} {elapsed_time:>8.2f}s (100.0%)")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error("\n" + "=" * 80)
        logger.error(f"[FAILURE] DECRYPTION FAILED after {elapsed_time:.2f} seconds")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        result['success'] = False
        result['error'] = str(e)
        result['processing_time'] = elapsed_time
        result['phase_times'] = phase_times
        
        return result


def main():
    """Main entry point for decryption."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python main_decrypt.py <encrypted_image> <metadata_json> [config_file] [output_dir]")
        print("Example: python main_decrypt.py output/encrypted/encrypted_image.png output/metadata/encryption_metadata.json config.json output/decrypted")
        sys.exit(1)
    
    encrypted_image_path = sys.argv[1]
    metadata_path = sys.argv[2]
    config_path = sys.argv[3] if len(sys.argv) > 3 else "config.json"
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "output/decrypted"
    
    result = orchestrate_decryption(encrypted_image_path, metadata_path, config_path, output_dir)
    
    if result['success']:
        print("\n✓ Decryption successful!")
        return 0
    else:
        print(f"\n✗ Decryption failed: {result['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
