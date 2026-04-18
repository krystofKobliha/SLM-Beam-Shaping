# ──────────────────────────────────────────────────────────────────────────────
# File: gs_generate.py          Project: DNN-SLM Phase Mask Generation
# Purpose: CLI runner for FFT-GS phase mask generation + CITL fine-tuning
# ──────────────────────────────────────────────────────────────────────────────
"""
Generates a phase mask from a target intensity using FFT Gerchberg-Saxton,
optionally refined with camera-in-the-loop (CITL) feedback.

HOW TO USE:
    1. Edit the USER CONFIGURATION section below (shape, size, position, etc.)
    2. Run:  python gs_generate.py
    3. Outputs are saved to gs_masks/<auto-named folder>/

Outputs:
    target.png          — target intensity image
    phase_mask_slm.bmp  — SLM-ready 1024×1272, gray 0–213 = phase 0–2π
    phase_mask_raw.bmp  — full 1024×1280 grayscale phase (0–255 for viewing)
    phase_mask.npy      — full 1024×1280 float phase [0, 2π]
    simulated.png       — physics-simulated reconstruction intensity
    comparison.png      — side-by-side: Target | Simulated Recon | Phase Mask
    dashboard.png       — 6-panel diagnostic dashboard
    blur_comparison.png — multi-sigma blur comparison

Module layout:
    target_generator.py  — target shape creation, image loading, blur helpers
    gerchberg_saxton.py  — FFT-based Gerchberg-Saxton phase retrieval
    slm_export.py        — phase tensor → SLM BMP conversion & save helpers
    citl_finetune.py     — camera-in-the-loop hardware fine-tuning
    gs_generate.py       — this file: user config, pipeline, CLI entry point
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import math
import numpy as np
import torch

from config import PHYSICS_CONFIG, SLM_HEIGHT_PX

from fft_physics import FFTPhysics
from utils import (
    save_visual_dashboard, visualize_comparison, save_blur_comparison,
    calculate_psnr,
)

from target_generator import make_target, load_target_image
from gerchberg_saxton import fft_gerchberg_saxton
from slm_export import phase_to_slm_bmp, phase_to_raw_bmp, save_tensor_as_png
from citl_finetune import camera_gs_finetune


# ---------------------------------------------------------------------------
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       USER CONFIGURATION                                 ║
# ║       Edit the values below, then run:  python gs_generate.py            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --- Target definition ---------------------------------------------------
# Pick ONE of:  'circle', 'ring', 'rect', 'text', or 'image'
SHAPE = 'text'

# Shape dimensions [pixels at 1024×1280 resolution]
RADIUS    = 150      # circle / ring: outer radius
WIDTH     = 250     # rect: full width
HEIGHT    = 50      # rect: full height
THICKNESS = 50       # ring: wall thickness
CHARACTER = 'HiLASE'      # text: single character (letter or digit)
FONT_SIZE = 100    # text: character height [px]

# Position: offset from the centre of the picture [pixels]
# (0, 0) = picture centre.  Positive CX → right, positive CY → down.
CX_OFFSET = 105
CY_OFFSET = -175

# If SHAPE = 'image', set the path to the grayscale target image:
TARGET_IMAGE_PATH = r"C:\Users\Administrator\Desktop\Krystof\Program PYTHON SLM\chessboard_2.bmp"  # e.g. r'C:\path\to\target.png'

# --- FFT Gerchberg-Saxton ------------------------------------------------
# Pure FFT-based GS that matches the working Mathematica code exactly.
# Uses fft2/ifft2 as forward/backward — exact mathematical inverses.
FFT_GS_STEPS = 150        # number of FFT-GS iterations
FFT_GS_GLOBAL_ITERS = 50   # iters with global constraint before MRAF

# --- CITL fine-tuning of GS result ---------------------------------------
CITL_FINETUNE = False         # set True to run CITL after GS

# Camera-GS parameters
# One-shot baseline correction removes the systematic beam/optics gradient,
# then a short iterative loop fine-tunes residual errors.
CAMGS_OUTER_ITERS = 10        # refinement iterations after one-shot correction
CAMGS_SHAPE_ITERS = 10        # (all iters use large blur)
CAMGS_BLUR_SIGMA_SHAPE = 0   # blur sigma (0 = auto from target feature size)
CAMGS_BLUR_SIGMA_FILL = 0    # unused (kept for API compat)
CAMGS_CORRECTION_STRENGTH = 0.1  # ratio exponent alpha

# Shared capture settings
CITL_CAPTURE_AVG = 30        # camera frames to average per capture
CITL_STABILIZATION_MS = 1500  # SLM settle time [ms]

# --- Output --------------------------------------------------------------
OUTPUT_ROOT = 'gs_masks'

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                 END OF USER CONFIGURATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def generate_phase_mask(target_tensor, device, outdir='gs_output',
                        fft_gs_steps=300, fft_gs_global_iters=50,
                        citl=False,
                        citl_capture_avg=10,
                        citl_stabilization_ms=1000,
                        camgs_outer_iters=20,
                        camgs_shape_iters=5,
                        camgs_blur_sigma_shape=0,
                        camgs_blur_sigma_fill=0,
                        camgs_correction_strength=0.1):
    """
    Full pipeline: FFT-GS → (optional Camera-GS CITL) → save.

    Phase 1 (FFT-GS): Pure FFT Gerchberg-Saxton produces a clean phase mask.
        Uses fft2/ifft2 as exact-inverse transforms, matching Mathematica.
    Phase 2 (Camera-GS): Iterative target pre-compensation on real hardware.

    Args:
        target_tensor:       [1, 1, H, W] target intensity in [0, 1].
        device:              torch device.
        outdir:              output directory.
        fft_gs_steps:        FFT-GS iterations.
        fft_gs_global_iters: FFT-GS global-constraint iterations before MRAF.
        citl:                run Camera-GS fine-tuning after GS.
        camgs_*:             Camera-GS hyper-parameters.

    Returns:
        (phase, recon) — both [1, 1, H, W] tensors.
    """
    os.makedirs(outdir, exist_ok=True)
    _, _, h, w = target_tensor.shape

    # 1. Save target
    save_tensor_as_png(target_tensor, os.path.join(outdir, 'target.png'))

    # 2. FFT physics model (single Fourier transform, matches GS model)
    physics = FFTPhysics(PHYSICS_CONFIG, (h, w), device)

    # 3. Phase 1 — FFT Gerchberg-Saxton
    waist_ratio = PHYSICS_CONFIG.get('waist_ratio', 0.45)
    print(f"Phase 1: FFT Gerchberg-Saxton ({fft_gs_steps} iters, "
          f"global={fft_gs_global_iters}) ...")
    phase = fft_gerchberg_saxton(
        target_tensor, waist_ratio=waist_ratio,
        steps=fft_gs_steps, global_iters=fft_gs_global_iters,
    )

    # 4. Phase 2 — Optional Camera-GS fine-tuning
    if citl:
        phase = camera_gs_finetune(
            target_tensor, device,
            initial_phase=phase,
            outer_iters=camgs_outer_iters,
            gs_steps=fft_gs_steps,
            gs_global_iters=fft_gs_global_iters,
            shape_iters=camgs_shape_iters,
            blur_sigma_shape=camgs_blur_sigma_shape,
            blur_sigma_fill=camgs_blur_sigma_fill,
            correction_strength=camgs_correction_strength,
            capture_avg=citl_capture_avg,
            stabilization_ms=citl_stabilization_ms,
            outdir=outdir,
        )

    # 4. Save phase masks (SLM-ready + raw + numpy)
    phase_to_slm_bmp(phase, os.path.join(outdir, 'phase_mask_slm.bmp'))
    phase_to_raw_bmp(phase, os.path.join(outdir, 'phase_mask_raw.bmp'))
    npy_path = os.path.join(outdir, 'phase_mask.npy')
    np.save(npy_path, phase.detach().cpu().squeeze().numpy())
    print(f"Saved phase array: {npy_path}")

    # 5. Simulated reconstruction + metrics
    with torch.no_grad():
        recon = physics(phase)
        s_t = target_tensor.sum(dim=(-1, -2), keepdim=True)
        s_r = recon.sum(dim=(-1, -2), keepdim=True)
        recon_scaled = recon * (s_t / (s_r + 1e-8))

        corr = torch.corrcoef(torch.stack([
            recon_scaled.reshape(-1), target_tensor.reshape(-1)
        ]))[0, 1].item()
        psnr = calculate_psnr(target_tensor, recon_scaled)
        print(f"Final:  Pearson r = {corr:.4f}  |  PSNR = {psnr:.1f} dB")

    # 6. Save visualisations
    save_tensor_as_png(recon_scaled, os.path.join(outdir, 'simulated.png'))
    visualize_comparison(
        target_tensor, recon_scaled, phase,
        os.path.join(outdir, 'comparison.png'),
    )
    save_visual_dashboard(
        target_tensor, recon_scaled, phase, 0,
        os.path.join(outdir, 'dashboard.png'),
    )
    save_blur_comparison(
        target_tensor, recon_scaled,
        os.path.join(outdir, 'blur_comparison.png'),
    )

    print(f"\nAll outputs saved to: {os.path.abspath(outdir)}/")
    return phase, recon_scaled


# ---------------------------------------------------------------------------
# Auto-naming & entry point
# ---------------------------------------------------------------------------
def _build_run_name(shape, **kwargs):
    """Build a compact folder name from run parameters."""
    parts = []

    if shape == 'image':
        img_path = kwargs.get('image_path') or 'target'
        base = os.path.splitext(os.path.basename(img_path))[0]
        parts.append(f"image_{base}")
    elif shape == 'circle':
        parts.append(f"circle_r{int(kwargs.get('radius', 0))}")
    elif shape == 'ring':
        parts.append(f"ring_r{int(kwargs.get('radius', 0))}")
        parts.append(f"t{int(kwargs.get('thickness', 0))}")
    elif shape == 'rect':
        parts.append(f"rect_{int(kwargs.get('width', 0))}x{int(kwargs.get('height', 0))}")
    elif shape == 'text':
        parts.append(f"text_{kwargs.get('character', '?')}_h{int(kwargs.get('font_size', 0))}")

    cx = int(kwargs.get('cx_offset', 0))
    cy = int(kwargs.get('cy_offset', 0))
    if cx != 0 or cy != 0:
        parts.append(f"cx{cx:+d}_cy{cy:+d}")

    parts.append(f"gs{int(kwargs.get('gs_steps', 300))}")

    if kwargs.get('citl'):
        parts.append(f"cgs{int(kwargs.get('camgs_outer_iters', 15))}")

    return "_".join(parts)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    resolution = (SLM_HEIGHT_PX, 1280)
    shape = SHAPE.lower().strip()

    # --- Build target -------------------------------------------------------
    if shape == 'image':
        if TARGET_IMAGE_PATH is None:
            raise ValueError("SHAPE is 'image' but TARGET_IMAGE_PATH is not set.")
        print(f"Loading target image: {TARGET_IMAGE_PATH}")
        target = load_target_image(TARGET_IMAGE_PATH, resolution, device)
        run_name = _build_run_name(
            shape, image_path=TARGET_IMAGE_PATH,
            gs_steps=FFT_GS_STEPS,
            citl=CITL_FINETUNE, camgs_outer_iters=CAMGS_OUTER_ITERS,
        )
    else:
        print(f"Generating '{shape}' target  "
              f"centre=({CX_OFFSET:+d}, {CY_OFFSET:+d}) px from picture centre")
        target = make_target(
            shape=shape, resolution=resolution, device=device,
            cx_offset=CX_OFFSET, cy_offset=CY_OFFSET,
            radius=RADIUS, width=WIDTH, height=HEIGHT, thickness=THICKNESS,
            character=CHARACTER, font_size=FONT_SIZE,
        )
        run_name = _build_run_name(
            shape, radius=RADIUS, width=WIDTH, height=HEIGHT,
            thickness=THICKNESS, cx_offset=CX_OFFSET, cy_offset=CY_OFFSET,
            gs_steps=FFT_GS_STEPS, character=CHARACTER, font_size=FONT_SIZE,
            citl=CITL_FINETUNE, camgs_outer_iters=CAMGS_OUTER_ITERS,
        )

    # --- Run pipeline -------------------------------------------------------
    outdir = os.path.join(OUTPUT_ROOT, run_name)
    print(f"Output folder: {os.path.abspath(outdir)}")

    generate_phase_mask(
        target, device, outdir=outdir,
        fft_gs_steps=FFT_GS_STEPS,
        fft_gs_global_iters=FFT_GS_GLOBAL_ITERS,
        citl=CITL_FINETUNE,
        citl_capture_avg=CITL_CAPTURE_AVG,
        citl_stabilization_ms=CITL_STABILIZATION_MS,
        camgs_outer_iters=CAMGS_OUTER_ITERS,
        camgs_shape_iters=CAMGS_SHAPE_ITERS,
        camgs_blur_sigma_shape=CAMGS_BLUR_SIGMA_SHAPE,
        camgs_blur_sigma_fill=CAMGS_BLUR_SIGMA_FILL,
        camgs_correction_strength=CAMGS_CORRECTION_STRENGTH,
    )


if __name__ == '__main__':
    main()
