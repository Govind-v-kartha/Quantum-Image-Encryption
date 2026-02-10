"""
Verification Engine
Validates encryption quality and decryption accuracy.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional


class VerificationEngine:
    """Check entropy, randomness, and pixel-exact reconstruction."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('verification_engine', {})
        self.logger = logging.getLogger('verification_engine')
        self.is_initialized = False

    def initialize(self):
        self.is_initialized = True
        self.logger.info("  Verification Engine initialized")

    # ------------------------------------------------------------------
    # Encryption quality
    # ------------------------------------------------------------------

    def verify_encryption(
        self,
        original: np.ndarray,
        encrypted: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Assess encryption quality.

        Returns dict with metrics:
          entropy          : float  (ideal 8.0 for uniform)
          pixel_change_pct : float  (% pixels changed)
          correlation      : float  (should be near 0)
          npcr             : float  (Number of Pixel Change Rate, %)
          uaci             : float  (Unified Average Changing Intensity, %)
        """
        results = {}

        # Entropy
        results['entropy'] = float(self._shannon_entropy(encrypted))

        # Pixel change
        if original.shape == encrypted.shape:
            changed = np.count_nonzero(original != encrypted)
            total = original.size
            results['pixel_change_pct'] = changed / total * 100.0

            # NPCR
            diff_map = (original != encrypted).any(axis=-1) if original.ndim == 3 else (original != encrypted)
            results['npcr'] = np.count_nonzero(diff_map) / diff_map.size * 100.0

            # UACI
            results['uaci'] = float(
                np.mean(np.abs(original.astype(np.float64) - encrypted.astype(np.float64))) / 255.0 * 100.0
            )

            # Correlation
            o_flat = original.ravel().astype(np.float64)
            e_flat = encrypted.ravel().astype(np.float64)
            if np.std(o_flat) > 0 and np.std(e_flat) > 0:
                results['correlation'] = float(np.corrcoef(o_flat, e_flat)[0, 1])
            else:
                results['correlation'] = 0.0
        else:
            results['pixel_change_pct'] = None
            results['npcr'] = None
            results['uaci'] = None
            results['correlation'] = None

        self.logger.info(
            f"  Encryption metrics: entropy={results['entropy']:.4f}, "
            f"NPCR={results.get('npcr', 'N/A')}%, "
            f"UACI={results.get('uaci', 'N/A')}%"
        )

        return results

    # ------------------------------------------------------------------
    # Decryption quality
    # ------------------------------------------------------------------

    def verify_decryption(
        self,
        original: np.ndarray,
        decrypted: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Assess decryption accuracy.

        Returns dict with metrics:
          pixel_exact : bool   – True if every pixel matches
          mse         : float  – Mean Squared Error (0 = perfect)
          psnr        : float  – Peak SNR (inf = perfect)
          ssim_approx : float  – simplified SSIM
        """
        results = {}

        results['pixel_exact'] = bool(np.array_equal(original, decrypted))

        diff = original.astype(np.float64) - decrypted.astype(np.float64)
        mse = float(np.mean(diff ** 2))
        results['mse'] = mse

        if mse > 0:
            results['psnr'] = float(10 * np.log10(255.0**2 / mse))
        else:
            results['psnr'] = float('inf')

        # Simplified SSIM (luminance + contrast)
        mu_o = np.mean(original.astype(np.float64))
        mu_d = np.mean(decrypted.astype(np.float64))
        sig_o = np.std(original.astype(np.float64))
        sig_d = np.std(decrypted.astype(np.float64))
        sig_od = np.mean((original.astype(np.float64) - mu_o) * (decrypted.astype(np.float64) - mu_d))
        c1 = (0.01 * 255)**2
        c2 = (0.03 * 255)**2
        ssim = ((2*mu_o*mu_d + c1) * (2*sig_od + c2)) / ((mu_o**2 + mu_d**2 + c1) * (sig_o**2 + sig_d**2 + c2))
        results['ssim'] = float(ssim)

        self.logger.info(
            f"  Decryption metrics: pixel_exact={results['pixel_exact']}, "
            f"MSE={mse:.4f}, PSNR={results['psnr']:.2f}dB, SSIM={results['ssim']:.6f}"
        )

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _shannon_entropy(image: np.ndarray) -> float:
        """Compute Shannon entropy in bits per pixel."""
        flat = image.ravel()
        counts = np.bincount(flat, minlength=256)
        probs = counts[counts > 0] / flat.size
        return float(-np.sum(probs * np.log2(probs)))
