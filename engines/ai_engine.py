"""
Layer 2: AI Engine
Uses FlexiMo Vision Transformer for semantic segmentation to identify
Regions of Interest (ROI) vs Background (BG).
Falls back to contrast-based detection when model weights are unavailable.
"""

import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Try to import FlexiMo
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'repos' / 'fleximo_repo'))
    from fleximo.models_dwv import OFAViT
    FLEXIMO_AVAILABLE = True
except Exception:
    FLEXIMO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AIEngine:
    """Produce ROI/BG binary mask from input image using FlexiMo or fallback."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('ai_engine', {})
        self.logger = logging.getLogger('ai_engine')
        self.is_initialized = False
        self.model = None
        self.use_fleximo = False
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.saliency_sigma = self.config.get('saliency_sigma', 1.5)

    def initialize(self):
        """Try to initialise FlexiMo; if unavailable fall back to contrast."""
        if FLEXIMO_AVAILABLE and TORCH_AVAILABLE:
            try:
                # Check for model weights
                weights_dir = Path(__file__).parent.parent / 'repos' / 'fleximo_repo' / 'figure'
                weight_files = list(weights_dir.glob('*.pth')) + list(weights_dir.glob('*.pt'))
                if weight_files:
                    self.model = OFAViT(
                        img_size=224,
                        patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        num_classes=2,
                    )
                    checkpoint = torch.load(str(weight_files[0]), map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        self.model.load_state_dict(checkpoint['model'], strict=False)
                    else:
                        self.model.load_state_dict(checkpoint, strict=False)
                    self.model.eval()
                    self.use_fleximo = True
                    self.logger.info("  FlexiMo model loaded successfully")
                else:
                    self.logger.info("  No FlexiMo weights found -> contrast-based fallback")
            except Exception as e:
                self.logger.warning(f"  FlexiMo init failed ({e}) -> contrast-based fallback")
        else:
            reason = "FlexiMo" if not FLEXIMO_AVAILABLE else "PyTorch"
            self.logger.info(f"  {reason} not available -> contrast-based fallback")

        self.is_initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_roi_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a binary ROI mask (255 = ROI, 0 = BG).

        Parameters
        ----------
        image : np.ndarray  (H, W, 3) uint8 RGB

        Returns
        -------
        mask : np.ndarray  (H, W) uint8, values {0, 255}
        """
        if self.use_fleximo:
            return self._fleximo_segment(image)
        else:
            return self._contrast_fallback(image)

    # ------------------------------------------------------------------
    # FlexiMo segmentation
    # ------------------------------------------------------------------

    def _fleximo_segment(self, image: np.ndarray) -> np.ndarray:
        """Run FlexiMo ViT to obtain ROI mask."""
        import torch
        import torch.nn.functional as F
        from PIL import Image as PILImage

        h, w = image.shape[:2]

        # Preprocess: resize to model input, normalise
        pil_img = PILImage.fromarray(image)
        pil_resized = pil_img.resize((224, 224), PILImage.BILINEAR)
        tensor = torch.from_numpy(np.array(pil_resized)).float().permute(2, 0, 1) / 255.0
        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0)  # (1, 3, 224, 224)

        with torch.no_grad():
            logits = self.model(tensor)  # (1, 2, H', W')
            probs = F.softmax(logits, dim=1)
            roi_prob = probs[0, 1]  # foreground probability

        # Resize probability map back to original size
        roi_prob_resized = F.interpolate(
            roi_prob.unsqueeze(0).unsqueeze(0),
            size=(h, w), mode='bilinear', align_corners=False
        ).squeeze().numpy()

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[roi_prob_resized >= self.confidence_threshold] = 255

        roi_pct = np.count_nonzero(mask) / mask.size * 100
        self.logger.info(f"  FlexiMo ROI: {roi_pct:.1f}% of pixels")
        return mask

    # ------------------------------------------------------------------
    # Contrast-based fallback
    # ------------------------------------------------------------------

    def _contrast_fallback(self, image: np.ndarray) -> np.ndarray:
        """
        Generate ROI mask using multi-feature contrast analysis.

        Combines:
          1) Local contrast (std-dev in 8x8 blocks)
          2) Edge magnitude (Sobel approx)
          3) Color saliency
        """
        h, w = image.shape[:2]
        gray = np.mean(image.astype(np.float32), axis=2)

        # ------ 1. Local contrast (block std) ------
        block = 8
        contrast = np.zeros_like(gray)
        for r in range(0, h - block + 1, block):
            for c in range(0, w - block + 1, block):
                patch = gray[r:r+block, c:c+block]
                contrast[r:r+block, c:c+block] = np.std(patch)

        # normalise 0..1
        cmax = contrast.max()
        if cmax > 0:
            contrast /= cmax

        # ------ 2. Sobel edge magnitude ------
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        edge = np.sqrt(gx**2 + gy**2)
        emax = edge.max()
        if emax > 0:
            edge /= emax

        # ------ 3. Color saliency ------
        lab_approx = image.astype(np.float32)
        mean_color = lab_approx.mean(axis=(0, 1), keepdims=True)
        saliency = np.sqrt(np.sum((lab_approx - mean_color) ** 2, axis=2))
        smax = saliency.max()
        if smax > 0:
            saliency /= smax

        # Combine scores
        score = 0.4 * contrast + 0.3 * edge + 0.3 * saliency

        # Adaptive threshold via Otsu-like split
        threshold = self._otsu_threshold(score)
        threshold = max(threshold, self.confidence_threshold * score.max())

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[score >= threshold] = 255

        # Ensure at least 10 % ROI
        roi_pct = np.count_nonzero(mask) / mask.size
        if roi_pct < 0.10:
            threshold = np.percentile(score, 90)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[score >= threshold] = 255

        # Ensure at most 90 % ROI
        roi_pct = np.count_nonzero(mask) / mask.size
        if roi_pct > 0.90:
            threshold = np.percentile(score, 10)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[score >= threshold] = 255

        roi_pct = np.count_nonzero(mask) / mask.size * 100
        self.logger.info(f"  Contrast-based ROI: {roi_pct:.1f}% of pixels")
        return mask

    @staticmethod
    def _otsu_threshold(data: np.ndarray) -> float:
        """Simple Otsu threshold on continuous [0,1] data."""
        hist_bins = 256
        flat = data.ravel()
        hist, bin_edges = np.histogram(flat, bins=hist_bins, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0.5

        centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        w0 = 0.0
        sum0 = 0.0
        sum_total = np.sum(hist * centres)
        best_t = 0.5
        best_var = 0.0

        for i in range(hist_bins):
            w0 += hist[i]
            if w0 == 0:
                continue
            w1 = total - w0
            if w1 == 0:
                break
            sum0 += hist[i] * centres[i]
            m0 = sum0 / w0
            m1 = (sum_total - sum0) / w1
            var = w0 * w1 * (m0 - m1) ** 2
            if var > best_var:
                best_var = var
                best_t = centres[i]

        return best_t
