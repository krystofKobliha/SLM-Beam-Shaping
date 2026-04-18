# ──────────────────────────────────────────────────────────────────────────────
# File: unet_generate.py       Project: DNN-SLM Phase Mask Generation
# Purpose: CLI runner for U-Net phase mask generation from trained model
# ──────────────────────────────────────────────────────────────────────────────
"""
Generates a phase mask from a target intensity using the trained U-Net,
with a single forward pass (no iterative optimisation).

HOW TO USE:
    1. Edit the USER CONFIGURATION section below (shape, size, position, etc.)
    2. Run:  python unet_generate.py
    3. Outputs are saved to unet_masks/<auto-named folder>/

Outputs (all inside the output folder):
    target.png          — target intensity image
    phase_mask_slm.bmp  — SLM-ready 1024×1272, gray 0–213 = phase 0–2π
    phase_mask_raw.bmp  — full 1024×1280 grayscale phase (0–255 for viewing)
    phase_mask.npy      — full 1024×1280 float phase [0, 2π]
    simulated.png       — physics-simulated reconstruction (scaled to target energy)
    comparison.png      — side-by-side: Target | Simulated Recon | Phase Mask
    dashboard.png       — 6-panel diagnostic dashboard
    blur_comparison.png — multi-sigma blur comparison
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import math
import numpy as np
import torch

from config import PHYSICS_CONFIG, SLM_HEIGHT_PX, SLM_PITCH
from model import HoloNet
from fft_physics import FFTPhysics
from utils import (
    save_visual_dashboard, visualize_comparison, save_blur_comparison,
    calculate_psnr, crop_edges,
)
from target_generator import make_target, load_target_image
from slm_export import phase_to_slm_bmp, phase_to_raw_bmp, save_tensor_as_png


# ---------------------------------------------------------------------------
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       USER CONFIGURATION                                 ║
# ║       Edit the values below, then run:  python unet_generate.py          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --- Model checkpoint -----------------------------------------------------
# 'ema' = EMA model (smoother weights, recommended)
# 'best' = best validation model (selected by val metric during training)
MODEL_TYPE = 'citl'

# Paths are resolved relative to the workspace root (parent of unet/)
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATHS = {
    'ema':  os.path.join(_WORKSPACE, 'holonet_ema_stage_7.pth'),
    'best': os.path.join(_WORKSPACE, 'holonet_best_S7.pth'),
    'citl': os.path.join(_WORKSPACE, 'unet_best_citl_ema.pth'),
}

# --- Target definition ---------------------------------------------------
# Pick ONE of:  'circle', 'ring', 'rect', 'text', or 'image'
SHAPE = 'text'

# Shape dimensions [pixels at 1024×1280 resolution]
RADIUS    = 150      # circle / ring: outer radius
WIDTH     = 175      # rect: full width
HEIGHT    = 175       # rect: full height
THICKNESS = 50       # ring: wall thickness
CHARACTER = 'V'      # text: single character (letter or digit)
FONT_SIZE = 200      # text: character height [px]

# Position: offset from the centre of the picture [pixels]
# (0, 0) = picture centre.  Positive CX → right, positive CY → down.
CX_OFFSET = 100
CY_OFFSET = -100

# If SHAPE = 'image', set the path to the grayscale target image:
TARGET_IMAGE_PATH = None  # e.g. r'C:\path\to\target.png'

# --- Output --------------------------------------------------------------
OUTPUT_ROOT = 'unet_masks'

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                 END OF USER CONFIGURATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_type, device):
    """Load the trained U-Net in eval mode.

    With GroupNorm (batch-independent), eval mode works perfectly for
    single-sample inference — no padded batches or BN hacks needed.

    Args:
        model_type: 'ema' or 'best'.
        device:     torch device.

    Returns:
        HoloNet model in eval mode.
    """
    path = MODEL_PATHS[model_type]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Make sure the .pth file exists in the workspace root."
        )
    print(f"Loading U-Net ({model_type}): {path}")
    model = HoloNet().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    del state

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def generate_phase_mask(model, target_tensor, device, outdir='unet_output'):
    """
    Full pipeline: U-Net forward pass → save.

    Args:
        model:          trained HoloNet in eval mode.
        target_tensor:  [1, 1, H, W] target intensity in [0, 1].
        device:         torch device.
        outdir:         output directory.

    Returns:
        (phase, recon) — both [1, 1, H, W] tensors.
    """
    os.makedirs(outdir, exist_ok=True)
    _, _, h, w = target_tensor.shape

    # 1. Save target
    tgt_path = os.path.join(outdir, 'target.png')
    save_tensor_as_png(target_tensor, tgt_path)

    # 2. U-Net forward pass
    print("Running U-Net forward pass ...")
    with torch.no_grad():
        phase = model(target_tensor.to(device))
    print(f"Phase mask shape: {phase.shape}  range: [{phase.min():.3f}, {phase.max():.3f}]")

    # 3. Save phase masks (SLM-ready + raw + numpy)
    phase_to_slm_bmp(phase, os.path.join(outdir, 'phase_mask_slm.bmp'))
    phase_to_raw_bmp(phase, os.path.join(outdir, 'phase_mask_raw.bmp'))
    npy_path = os.path.join(outdir, 'phase_mask.npy')
    np.save(npy_path, phase.detach().cpu().squeeze().numpy())
    print(f"Saved phase array: {npy_path}")

    # 4. Simulated reconstruction + metrics
    physics = FFTPhysics(PHYSICS_CONFIG, (h, w), device)
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

    # 5. Save visualisations
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
def _build_run_name(model_type, shape, **kwargs):
    """Build a compact folder name from run parameters."""
    parts = [f"unet_{model_type}"]

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

    return "_".join(parts)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    resolution = (SLM_HEIGHT_PX, 1280)
    shape = SHAPE.lower().strip()

    # --- Load model ---------------------------------------------------------
    model = load_model(MODEL_TYPE, device)

    # --- Build target -------------------------------------------------------
    if shape == 'image':
        if TARGET_IMAGE_PATH is None:
            raise ValueError("SHAPE is 'image' but TARGET_IMAGE_PATH is not set.")
        print(f"Loading target image: {TARGET_IMAGE_PATH}")
        target = load_target_image(TARGET_IMAGE_PATH, resolution, device)
        run_name = _build_run_name(
            MODEL_TYPE, shape, image_path=TARGET_IMAGE_PATH,
        )
    else:
        print(f"Generating '{shape}' target  "
              f"centre=({CX_OFFSET:+d}, {CY_OFFSET:+d}) px from picture centre")
        target = make_target(
            shape=shape, resolution=resolution, device=device,
            cx_offset=CX_OFFSET, cy_offset=CY_OFFSET,
            radius=RADIUS, width=WIDTH, height=HEIGHT, thickness=THICKNESS,
            character=CHARACTER, font_size=FONT_SIZE,
            dft_aspect_correct=False,
        )
        run_name = _build_run_name(
            MODEL_TYPE, shape,
            radius=RADIUS, width=WIDTH, height=HEIGHT,
            thickness=THICKNESS, cx_offset=CX_OFFSET, cy_offset=CY_OFFSET,
            character=CHARACTER, font_size=FONT_SIZE,
        )

    # --- Match training pre-processing (S7 margin = 0.08) -----------------
    # During training, a soft aperture zeros the edges of every target to
    # prevent the network from learning FFT-boundary artefacts (Gibbs ringing,
    # aliasing).  The margin shrinks with resolution (15 % at S1 → 8 % at S7).
    # At inference we apply the same S7 margin so the input distribution
    # matches what the trained model expects — without this, edge energy
    # leaks into the loss-free border and degrades reconstruction quality.
    target = crop_edges(target, 0.08)

    # --- Run pipeline -------------------------------------------------------
    outdir = os.path.join(OUTPUT_ROOT, run_name)
    print(f"Output folder: {os.path.abspath(outdir)}")

    generate_phase_mask(model, target, device, outdir=outdir)


if __name__ == '__main__':
    main()
