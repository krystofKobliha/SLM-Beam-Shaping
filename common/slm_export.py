# ──────────────────────────────────────────────────────────────────────────────
# File: slm_export.py          Project: DNN-SLM Phase Mask Generation
# Purpose: Phase tensor → SLM-ready BMP export and save helpers
# ──────────────────────────────────────────────────────────────────────────────
"""
Converts phase tensors to SLM-compatible bitmap files.

Provides:
  - phase_to_slm_bmp : 1024×1272 grayscale BMP (gray 0–213 = phase 0–2π)
  - phase_to_raw_bmp : 1024×1280 grayscale BMP (gray 0–255, for viewing)
  - save_tensor_as_png : normalised grayscale PNG from any tensor
"""
import math
import os
import numpy as np
import cv2
from PIL import Image

from config import SLM_ACTIVE_WIDTH_PX, SLM_HEIGHT_PX, SLM_GRAY_2PI
from config import SLM_CALIB_BMP, SLM_CALIB_GRAY_2PI
from utils import tensor_to_image


def _load_calibration_rad():
    """Load calibration mask as radians (1024×1272).  Cached lazily."""
    if not hasattr(_load_calibration_rad, '_cache'):
        if os.path.isfile(SLM_CALIB_BMP):
            cal = np.array(Image.open(SLM_CALIB_BMP).convert('L'),
                           dtype=np.float32)
            if cal.shape[1] == 1280:
                cal = cal[:, 4:-4]
            _load_calibration_rad._cache = cal * (2.0 * math.pi / SLM_CALIB_GRAY_2PI)
        else:
            _load_calibration_rad._cache = np.zeros(
                (SLM_HEIGHT_PX, SLM_ACTIVE_WIDTH_PX), dtype=np.float32)
    return _load_calibration_rad._cache


def phase_to_slm_bmp(phase_tensor, save_path):
    """
    Convert a [1, 1, 1024, 1280] phase tensor to SLM-ready BMP.

    Mapping: phase [0, 2π] → gray [0, SLM_GRAY_2PI].
    Cropping: 4 px from each side (1280 → 1272).
    Adds per-pixel calibration correction (from CAL BMP).
    """
    phase_np = phase_tensor.detach().cpu().squeeze().numpy()

    if phase_np.shape[1] == 1280:
        phase_np = phase_np[:, 4:-4]

    # Add calibration correction (Mathematica: Mod[holo + cal, 2π])
    cal_rad = _load_calibration_rad()
    phase_corrected = np.mod(phase_np + cal_rad, 2.0 * math.pi)

    gray = np.clip(phase_corrected / (2 * math.pi) * SLM_GRAY_2PI,
                   0, SLM_GRAY_2PI).astype(np.uint8)

    assert gray.shape == (SLM_HEIGHT_PX, SLM_ACTIVE_WIDTH_PX), (
        f"Expected ({SLM_HEIGHT_PX}, {SLM_ACTIVE_WIDTH_PX}), got {gray.shape}"
    )

    Image.fromarray(gray, mode='L').save(save_path)
    print(f"Saved SLM mask: {save_path}  shape={gray.shape}  "
          f"gray range=[{gray.min()}, {gray.max()}]")
    return gray


def phase_to_raw_bmp(phase_tensor, save_path):
    """Save the full 1024×1280 phase as a grayscale BMP (0–255 for viewing)."""
    phase_np = phase_tensor.detach().cpu().squeeze().numpy()
    gray = np.clip(phase_np / (2 * math.pi) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(gray, mode='L').save(save_path)
    print(f"Saved raw phase BMP: {save_path}  shape={gray.shape}")


def save_tensor_as_png(tensor, save_path):
    """Save a [1, 1, H, W] tensor as a normalised grayscale PNG."""
    img = tensor_to_image(tensor)
    if img.max() > 0:
        img = img / img.max()
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, img_uint8)
    print(f"Saved: {save_path}")
