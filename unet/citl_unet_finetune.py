# ──────────────────────────────────────────────────────────────────────────────
# File: citl_unet_finetune.py   Project: DNN-SLM Phase Mask Generation
# Purpose: Camera-in-the-loop fine-tuning of the trained U-Net at full SLM res
# ──────────────────────────────────────────────────────────────────────────────
"""
Fine-tunes a pre-trained HoloNet U-Net using real SLM + camera hardware
feedback (Camera-in-the-Loop, CITL).

Workflow per iteration:
    1. Generate a random target (SyntheticGenerator)
    2. U-Net predicts a phase mask (differentiable)
    3. Upload the phase mask to the SLM, wait for settling, capture camera
    4. Run the same phase through differentiable FFTPhysics → sim. recon.
    5. Blur camera capture to remove speckle
    6. Compute camera-to-sim correction ratio (captures sim-reality gap)
    7. Apply correction to the differentiable simulation → "camera-corrected sim"
    8. Loss = Pearson(blur(corrected_sim), blur(target)) + TV + BG
    9. Backprop through the differentiable path → update U-Net weights

WHY THIS WORKS:
    The camera path (SLM→optics→camera) is NOT differentiable, but the
    simulated physics path IS.  The correction ratio (camera / sim) captures
    the systematic reality gap — beam profile, optics aberrations, SLM
    non-uniformity.  Multiplying the differentiable simulation by this
    (detached) correction gives a "camera-calibrated differentiable signal":
    the loss value reflects real hardware quality, while gradients flow
    through the simulation model to update U-Net weights.

    The beam-stop region (defined in config.py) is always zero in both the
    target and the camera capture — no targets are generated there and the
    camera zeros it out during preprocessing.

HOW TO USE:
    1. Edit the USER CONFIGURATION section below
    2. Ensure SLM and camera are connected and operational
    3. Run:  python citl_unet_finetune.py
    4. Outputs are saved to the configured output directory

Requires:  SLM hardware (hpkSLMdaLV.dll) + CinCam beam profiler (XML-RPC)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import math
import time
import copy
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import (
    PHYSICS_CONFIG, SLM_HEIGHT_PX,
    BEAM_STOP_X_CENTER, BEAM_STOP_X_HALF_WIDTH,
    BEAM_STOP_Y_START, BEAM_STOP_Y_END,
)
from model import HoloNet
from fft_physics import FFTPhysics
from dataset import SyntheticGenerator
from slm_export import phase_to_slm_bmp, phase_to_raw_bmp, save_tensor_as_png
from utils import (
    save_visual_dashboard, calculate_psnr, calculate_blurred_psnr,
    tensor_to_image,
)

# Optional hardware imports — gracefully fall back when unavailable
_HW_AVAILABLE = False
try:
    from slm_communicate import SLMController
    from extract_camera_data import get_single_frame
    from profile_adjustment import HitlPreprocessor
    _HW_AVAILABLE = True
except Exception:
    pass


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       USER CONFIGURATION                               ║
# ║     Edit the values below, then run:  python citl_unet_finetune.py     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --- Model checkpoint (pre-trained U-Net) --------------------------------
# Path to the HoloNet .pth checkpoint trained through progressive stages.
# Typically the S7 (full-resolution) checkpoint from simulation training.
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), '..',
    'holonet_best_S7.pth',
)

# --- Training parameters -------------------------------------------------
TOTAL_ITERS       = 2500      # total CITL training iterations
LEARNING_RATE     = 3e-5      # small LR — fine-tuning, not training from scratch
GRAD_ACCUM        = 4         # gradient accumulation steps (effective batch = 4)
EMA_DECAY         = 0.999     # exponential moving average for weight smoothing
WEIGHT_DECAY      = 1e-6      # AdamW weight decay

# --- CITL loss parameters ------------------------------------------------
# Blur sigma for comparing camera vs simulation vs target.
# Must be large enough to average out speckle (~8–15 px at full resolution).
BLUR_SIGMA        = 8.0

# Correction ratio clamp range (prevents extreme adjustments from noise)
CORRECTION_MIN    = 0.5
CORRECTION_MAX    = 2.0

# Weights for the two loss branches:
#   CITL: Pearson(blur(sim_corrected_by_camera), blur(target))
#   SIM : Pearson(blur(sim), blur(target))   [pure simulation, no camera]
# Sum should be 1.0.  Higher CITL_WEIGHT = more camera influence.
CITL_WEIGHT       = 0.7
SIM_WEIGHT        = 0.3

# --- Regularisation weights -----------------------------------------------
TV_WEIGHT         = 0.005     # total variation (phase smoothness)
BG_WEIGHT         = 1.0       # background suppression (energy outside target)
PHASE_VAR_WEIGHT  = 0.15      # anti-flat-phase penalty

# --- Hardware / capture settings ------------------------------------------
CAPTURE_AVG       = 20        # camera frames to average per capture
STABILIZATION_MS  = 1500      # SLM settle time [ms] after uploading a new phase

# --- Output ---------------------------------------------------------------
CHECKPOINT_INTERVAL = 50      # save model checkpoint every N iterations
VIS_INTERVAL        = 10      # save diagnostic dashboard every N iterations
OUTPUT_DIR          = 'citl_unet_outputs'

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    END OF USER CONFIGURATION                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_blur(x, sigma, cache={}):
    """Separable Gaussian blur on a [1,1,H,W] tensor. Cached kernels."""
    if sigma < 0.5:
        return x
    key = (sigma, x.device, x.dtype)
    if key not in cache:
        ks = int(6 * sigma + 1) | 1
        pad = ks // 2
        coords = torch.arange(ks, device=x.device, dtype=x.dtype) - pad
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        cache[key] = (g.reshape(1, 1, 1, ks), g.reshape(1, 1, ks, 1), pad)
    kw, kh, pad = cache[key]
    out = F.conv2d(x, kw, padding=(0, pad))
    out = F.conv2d(out, kh, padding=(pad, 0))
    return out


def _make_beam_stop_mask_tensor(h, w, device):
    """Create a float beam-stop mask [1,1,H,W]: 1 = valid, 0 = blocked."""
    mask = torch.ones(1, 1, h, w, device=device)
    if BEAM_STOP_X_HALF_WIDTH <= 0:
        return mask
    cx = int(round(w * BEAM_STOP_X_CENTER))
    hw = int(round(w * BEAM_STOP_X_HALF_WIDTH))
    y0 = int(round(h * BEAM_STOP_Y_START))
    y1 = int(round(h * BEAM_STOP_Y_END))
    x0 = max(0, cx - hw)
    x1 = min(w, cx + hw)
    mask[:, :, y0:y1, x0:x1] = 0.0
    return mask


def _pearson_loss(pred, target):
    """1 - Pearson correlation. Scale-invariant shape loss."""
    p = pred.reshape(1, -1)
    t = target.reshape(1, -1)
    p_m = p - p.mean(dim=1, keepdim=True)
    t_m = t - t.mean(dim=1, keepdim=True)
    corr = F.cosine_similarity(p_m, t_m, dim=1, eps=1e-8)
    return 1.0 - corr.mean()


def _cosine_tv_loss(phase):
    """Phase-wrapped total variation (smooth across 0/2π boundary)."""
    if phase.ndim == 3:
        phase = phase.unsqueeze(1)
    dh = phase[:, :, 1:, :] - phase[:, :, :-1, :]
    dw = phase[:, :, :, 1:] - phase[:, :, :, :-1]
    return torch.mean(1.0 - torch.cos(dh)) + torch.mean(1.0 - torch.cos(dw))


def _local_phase_variance_loss(phase, patch_size=32):
    """Penalise flat-phase regions (circular variance per patch)."""
    if phase.ndim == 3:
        phase = phase.unsqueeze(1)
    B, _, H, W = phase.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        phase = F.pad(phase, (0, pad_w, 0, pad_h), mode='reflect')
    patches = phase.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    R = torch.sqrt(
        torch.sin(patches).mean((-2, -1)) ** 2 +
        torch.cos(patches).mean((-2, -1)) ** 2 + 1e-8
    )
    circ_var = 1.0 - R
    min_var = circ_var.reshape(B, -1).min(dim=1).values
    return (1.0 - min_var).mean()


def _save_citl_dashboard(target, sim_recon, cam_recon, phase, iteration,
                         metrics, save_path):
    """6-panel CITL diagnostic: target, sim, camera, phase, profiles, metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    t_img = tensor_to_image(target)
    s_img = tensor_to_image(sim_recon)
    c_img = tensor_to_image(cam_recon) if cam_recon is not None else np.zeros_like(t_img)
    p_img = tensor_to_image(phase)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, figure=fig)

    # Row 1: Target, Simulated, Camera, Phase
    ax_t = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_p = fig.add_subplot(gs[0, 3])

    ax_t.imshow(t_img, cmap='hot', vmin=0, vmax=max(t_img.max(), 0.01))
    ax_t.set_title('Target')
    ax_t.axis('off')

    ax_s.imshow(s_img, cmap='hot', vmin=0, vmax=max(s_img.max(), 0.01))
    ax_s.set_title(f'Sim Recon (max={s_img.max():.2f})')
    ax_s.axis('off')

    ax_c.imshow(c_img, cmap='hot', vmin=0, vmax=max(c_img.max(), 0.01))
    ax_c.set_title(f'Camera (max={c_img.max():.2f})')
    ax_c.axis('off')

    im_p = ax_p.imshow(p_img, cmap='twilight', vmin=0, vmax=2 * np.pi)
    ax_p.set_title('Phase Mask')
    ax_p.axis('off')
    plt.colorbar(im_p, ax=ax_p, fraction=0.046)

    # Row 2: Blurred comparison + cross-section + metric history
    ax_blur = fig.add_subplot(gs[1, 0])
    ax_prof = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[1, 2:])

    # Blurred overlays
    sigma_vis = 8.0
    with torch.no_grad():
        t_blur = _gaussian_blur(target, sigma_vis).squeeze().cpu().numpy()
        s_blur = _gaussian_blur(sim_recon, sigma_vis).squeeze().cpu().numpy()
    ax_blur.imshow(t_blur, cmap='Greens', alpha=0.6, vmin=0,
                   vmax=max(t_blur.max(), 0.01))
    ax_blur.imshow(s_blur, cmap='Reds', alpha=0.4, vmin=0,
                   vmax=max(s_blur.max(), 0.01))
    ax_blur.set_title(f'Blur overlay (σ={sigma_vis})\nGreen=target, Red=sim')
    ax_blur.axis('off')

    # Cross-section (centre row)
    h = t_img.shape[0]
    cy = h // 2
    ax_prof.plot(t_img[cy, :], label='Target', color='black', alpha=0.6)
    ax_prof.plot(s_img[cy, :], label='Sim', color='tab:red', alpha=0.8)
    if cam_recon is not None:
        ax_prof.plot(c_img[cy, :], label='Camera', color='tab:blue', alpha=0.8)
    ax_prof.set_title('Cross-section (centre row)')
    ax_prof.legend(fontsize=8)
    ax_prof.set_xlabel('Pixel X')
    ax_prof.set_ylabel('Intensity')

    # Metric text
    txt_lines = [f'Iteration {iteration}', '']
    for k, v in metrics.items():
        if isinstance(v, float):
            txt_lines.append(f'{k}: {v:.5f}')
        else:
            txt_lines.append(f'{k}: {v}')
    ax_hist.text(0.05, 0.95, '\n'.join(txt_lines), transform=ax_hist.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax_hist.axis('off')
    ax_hist.set_title('Metrics')

    fig.suptitle(f'CITL U-Net Fine-Tuning — iter {iteration}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    plt.close('all')


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------
class EMAModel:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    def apply_to(self, model):
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# Main CITL fine-tuning loop
# ---------------------------------------------------------------------------
def citl_finetune_unet(
    checkpoint_path,
    total_iters=500,
    lr=3e-5,
    grad_accum=4,
    ema_decay=0.999,
    weight_decay=1e-6,
    blur_sigma=8.0,
    correction_clip=(0.5, 2.0),
    citl_weight=0.7,
    sim_weight=0.3,
    tv_weight=0.005,
    bg_weight=1.0,
    phase_var_weight=0.15,
    capture_avg=20,
    stabilization_ms=1500,
    checkpoint_interval=50,
    vis_interval=10,
    output_dir='citl_unet_outputs',
):
    """
    Camera-in-the-loop fine-tuning of a pre-trained HoloNet U-Net.

    Args:
        checkpoint_path: path to pre-trained .pth file
        total_iters: number of CITL training iterations
        lr: learning rate
        grad_accum: gradient accumulation steps
        ema_decay: EMA decay for weight smoothing
        weight_decay: AdamW weight decay
        blur_sigma: Gaussian blur sigma for speckle smoothing
        correction_clip: (min, max) for camera/sim correction ratio
        citl_weight: weight of camera-corrected loss branch
        sim_weight: weight of pure simulation loss branch
        tv_weight: total variation regularisation weight
        bg_weight: background suppression weight
        phase_var_weight: anti-flat-phase penalty weight
        capture_avg: camera frames to average per capture
        stabilization_ms: SLM settle time in ms
        checkpoint_interval: save checkpoint every N iters
        vis_interval: save dashboard every N iters
        output_dir: output directory
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Hardware available: {_HW_AVAILABLE}")

    if not _HW_AVAILABLE:
        print("ERROR: SLM/camera hardware not available. CITL fine-tuning "
              "requires real hardware. Exiting.")
        return

    resolution = (SLM_HEIGHT_PX, 1280)
    h, w = resolution
    stab_s = stabilization_ms / 1000.0
    corr_min, corr_max = correction_clip

    # --- Create output directory ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f'run_{timestamp}')
    snap_dir = os.path.join(run_dir, 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'training_log.txt')

    # --- Load pre-trained U-Net ---
    print(f"Loading checkpoint: {checkpoint_path}")
    model = HoloNet(n_channels=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Handle both raw state_dict and nested checkpoint formats
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.train()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # --- Differentiable physics model ---
    physics = FFTPhysics(PHYSICS_CONFIG, resolution, device)

    # --- Optimizer + EMA ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    ema = EMAModel(model, decay=ema_decay)

    # --- Target generator (full resolution) ---
    generator = SyntheticGenerator(
        device=device, resolution=resolution,
        pixel_pitch=PHYSICS_CONFIG['pixel_pitch'],
        wavelength=PHYSICS_CONFIG['wavelength'],
        focal_length=PHYSICS_CONFIG['f3'],
        waist_ratio=PHYSICS_CONFIG.get('waist_ratio', 0.45),
    )

    # --- Beam-stop mask (zero out blocked region in targets) ---
    beam_stop_mask = _make_beam_stop_mask_tensor(h, w, device)

    # --- Hardware initialisation ---
    print("Initialising SLM ...")
    slm = SLMController()

    print("Initialising camera preprocessor ...")
    preprocessor = HitlPreprocessor()
    _repo_root = os.path.join(os.path.dirname(__file__), '..')
    _latest_cal = os.path.normpath(
        os.path.join(_repo_root, 'calibration_outputs', 'latest.json'))
    if not preprocessor.aligner.load(_latest_cal):
        print("  WARNING: No calibration file — geometric correction disabled.")

    # --- Background capture (flat phase on SLM) ---
    print("Capturing background (flat phase) ...")
    flat_phase = torch.zeros(1, 1, h, w, device=device)
    slm.display_phase_tensor(flat_phase)
    time.sleep(stab_s * 3)
    preprocessor.capture_background(get_single_frame, n_frames=capture_avg * 2)

    # --- Intensity mapping calibration ---
    # Run a known pattern through UNet + SLM + camera to calibrate sim↔camera
    print("Calibrating intensity mapping ...")
    with torch.no_grad():
        cal_target = generator.sample_batch(1)
        cal_target = cal_target * beam_stop_mask
        cal_phase = model(cal_target)
        cal_sim = physics(cal_phase)

    slm.display_phase_tensor(cal_phase)
    time.sleep(stab_s * 2)
    cal_frames = []
    for _ in range(capture_avg * 2):
        f, _ = get_single_frame()
        if f is not None:
            cal_frames.append(f)
    if cal_frames:
        avg_cal = np.mean(cal_frames, axis=0).astype(np.uint8)
        cam_proc = preprocessor.subtract_background(avg_cal)
        if preprocessor.aligner.is_calibrated:
            ch, cw = cam_proc.shape[:2]
            cos_t = np.cos(preprocessor.aligner.rotation)
            sin_t = np.sin(preprocessor.aligner.rotation)
            s = preprocessor.aligner.scale
            M = np.float32([
                [s * cos_t, -s * sin_t, preprocessor.aligner.shift_x],
                [s * sin_t,  s * cos_t, preprocessor.aligner.shift_y]
            ])
            cam_proc = cv2.warpAffine(
                cam_proc, M, (cw, ch), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(np.float32)
        cam_resized = cv2.resize(cam_proc, (w, h), interpolation=cv2.INTER_LINEAR)
        sim_np = cal_sim.squeeze().cpu().numpy()
        sim_max = sim_np.max()
        if sim_max > 0:
            sim_np /= sim_max
        cam_max = cam_resized.max()
        if cam_max > 0:
            cam_resized_norm = cam_resized / cam_max
        else:
            cam_resized_norm = cam_resized
        preprocessor.compute_intensity_mapping([sim_np], [cam_resized_norm])
        print(f"  Intensity mapping: a={preprocessor.intensity_a:.4f}, "
              f"b={preprocessor.intensity_b:.6f}")
    else:
        print("  WARNING: Calibration capture failed, using identity mapping.")

    # --- Write log header ---
    with open(log_path, 'w') as lf:
        lf.write(f"CITL U-Net Fine-Tuning  —  {timestamp}\n")
        lf.write(f"Checkpoint: {checkpoint_path}\n")
        lf.write(f"lr={lr}  grad_accum={grad_accum}  blur_sigma={blur_sigma}  "
                 f"citl_w={citl_weight}  sim_w={sim_weight}  "
                 f"tv_w={tv_weight}  bg_w={bg_weight}  pv_w={phase_var_weight}\n")
        lf.write(f"correction_clip=[{corr_min},{corr_max}]  "
                 f"capture_avg={capture_avg}  stab_ms={stabilization_ms}\n")
        lf.write("-" * 90 + "\n")
        lf.write(f"{'iter':>5s}  {'loss':>8s}  {'l_citl':>8s}  {'l_sim':>8s}  "
                 f"{'l_tv':>7s}  {'l_bg':>7s}  {'sim_psnr':>8s}  {'cam_psnr':>8s}  "
                 f"{'sim_pearson':>11s}  {'cam_pearson':>11s}\n")
        lf.write("-" * 90 + "\n")

    # --- Training loop ---
    print(f"\nStarting CITL fine-tuning: {total_iters} iterations")
    print(f"  blur_sigma={blur_sigma}  citl_weight={citl_weight}  "
          f"sim_weight={sim_weight}")
    print(f"  grad_accum={grad_accum}  lr={lr}")
    print(f"  Output: {os.path.abspath(run_dir)}")
    print("-" * 70)

    best_cam_pearson = -1.0
    best_iter = 0
    loss_history = []
    cam_pearson_history = []
    sim_pearson_history = []
    citl_loss_history = []
    sim_loss_history = []
    tv_loss_history = []
    bg_loss_history = []
    cam_psnr_history = []
    sim_psnr_history = []

    optimizer.zero_grad()

    for iteration in range(1, total_iters + 1):
        iter_start = time.time()

        # 1. Generate random target (with beam-stop mask applied)
        target = generator.sample_batch(1)   # [1, 1, H, W]
        target = target * beam_stop_mask
        # Re-normalise after masking
        t_max = target.max()
        if t_max > 0:
            target = target / t_max

        # 2. U-Net forward pass → phase  (differentiable)
        phase = model(target)                # [1, 1, H, W]

        # 3. Differentiable simulation → reconstruction
        sim_recon = physics(phase)           # [1, 1, H, W]

        # Energy-normalise simulation to match target
        sim_sum = sim_recon.sum(dim=(-1, -2), keepdim=True) + 1e-8
        tgt_sum = target.sum(dim=(-1, -2), keepdim=True) + 1e-8
        scale = (tgt_sum / sim_sum).clamp(max=100.0).detach()
        sim_norm = sim_recon * scale

        # 4. Upload phase to SLM, wait, capture camera image
        with torch.no_grad():
            slm.display_phase_tensor(phase)
        time.sleep(stab_s)

        frames = []
        for _ in range(capture_avg):
            f, _ = get_single_frame()
            if f is not None:
                frames.append(f)

        capture_ok = len(frames) > 0
        if capture_ok:
            avg_frame = np.mean(frames, axis=0).astype(np.uint8)
            cam_tensor = preprocessor.camera_to_tensor(avg_frame, device, resolution)
        else:
            print(f"  iter {iteration}: camera capture failed — using sim only")
            cam_tensor = None

        # 5. Blur everything for comparison
        sim_blur = _gaussian_blur(sim_norm, blur_sigma)
        tgt_blur = _gaussian_blur(target, blur_sigma)

        # 6. Compute losses
        # 6a. Pure simulation loss (always available, differentiable)
        loss_sim = _pearson_loss(sim_blur, tgt_blur)

        # 6b. Camera-corrected simulation loss (CITL component)
        if capture_ok and cam_tensor is not None:
            with torch.no_grad():
                cam_blur = _gaussian_blur(cam_tensor, blur_sigma)
                sim_blur_detached = _gaussian_blur(sim_norm.detach(), blur_sigma)

                # Energy-scale camera to match target on valid pixels
                cam_sum = cam_blur.sum() + 1e-8
                tgt_sum_blur = tgt_blur.sum() + 1e-8
                cam_blur_scaled = cam_blur * (tgt_sum_blur / cam_sum)

                # Correction ratio: where camera differs from simulation
                # correction > 1 → camera brighter than sim → sim underestimates
                # correction < 1 → camera dimmer than sim → sim overestimates
                correction = cam_blur_scaled / (sim_blur_detached + 1e-8)
                correction = correction.clamp(corr_min, corr_max)

            # Apply correction to the differentiable simulation
            # The correction is detached — gradients flow through sim_norm only
            sim_corrected = sim_norm * correction
            sim_corrected_blur = _gaussian_blur(sim_corrected, blur_sigma)
            loss_citl = _pearson_loss(sim_corrected_blur, tgt_blur)
        else:
            loss_citl = loss_sim  # fallback to pure sim

        # 6c. Combined reconstruction loss
        loss_recon = citl_weight * loss_citl + sim_weight * loss_sim

        # 6d. TV regularisation
        loss_tv = _cosine_tv_loss(phase)

        # 6e. Background suppression
        with torch.no_grad():
            bg_mask = (target < 0.05 * target.max()).float()
        loss_bg = (sim_norm * bg_mask).mean()

        # 6f. Phase variance (anti-flat)
        loss_pv = _local_phase_variance_loss(phase)

        # Total loss
        total_loss = (loss_recon
                      + tv_weight * loss_tv
                      + bg_weight * loss_bg
                      + phase_var_weight * loss_pv)

        # 7. Backprop (with gradient accumulation)
        (total_loss / grad_accum).backward()

        if iteration % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(model)

        # 8. Metrics (detached, for logging)
        with torch.no_grad():
            sim_psnr = calculate_psnr(target, sim_norm)
            sim_blur_psnr = calculate_blurred_psnr(target, sim_norm, sigma=10.0)

            # Pearson correlations
            t_flat = target.reshape(-1)
            s_flat = sim_norm.reshape(-1)
            t_m = t_flat - t_flat.mean()
            s_m = s_flat - s_flat.mean()
            sim_pearson = F.cosine_similarity(
                t_m.unsqueeze(0), s_m.unsqueeze(0), eps=1e-8).item()

            if capture_ok and cam_tensor is not None:
                cam_psnr = calculate_blurred_psnr(target, cam_tensor, sigma=10.0)
                c_flat = cam_tensor.reshape(-1)
                c_m = c_flat - c_flat.mean()
                cam_pearson = F.cosine_similarity(
                    t_m.unsqueeze(0), c_m.unsqueeze(0), eps=1e-8).item()
            else:
                cam_psnr = 0.0
                cam_pearson = 0.0

        loss_val = total_loss.item()
        loss_history.append(loss_val)
        cam_pearson_history.append(cam_pearson)
        sim_pearson_history.append(sim_pearson)
        citl_loss_history.append(loss_citl.item())
        sim_loss_history.append(loss_sim.item())
        tv_loss_history.append(loss_tv.item())
        bg_loss_history.append(loss_bg.item())
        cam_psnr_history.append(cam_psnr)
        sim_psnr_history.append(sim_psnr)

        # 9. Track best (by camera Pearson)
        if cam_pearson > best_cam_pearson:
            best_cam_pearson = cam_pearson
            best_iter = iteration
            torch.save(model.state_dict(),
                       os.path.join(run_dir, 'unet_best_citl.pth'))
            torch.save(ema.state_dict(),
                       os.path.join(run_dir, 'unet_best_citl_ema.pth'))

        # 10. Console + file logging
        elapsed = time.time() - iter_start
        print(f"  iter {iteration:4d}/{total_iters}  "
              f"loss={loss_val:.5f}  "
              f"citl={loss_citl.item():.4f}  sim={loss_sim.item():.4f}  "
              f"tv={loss_tv.item():.3f}  bg={loss_bg.item():.4f}  "
              f"sim_P={sim_pearson:.4f}  cam_P={cam_pearson:.4f}  "
              f"[{elapsed:.1f}s]")

        with open(log_path, 'a') as lf:
            lf.write(f"{iteration:5d}  {loss_val:8.5f}  "
                     f"{loss_citl.item():8.5f}  {loss_sim.item():8.5f}  "
                     f"{loss_tv.item():7.4f}  {loss_bg.item():7.5f}  "
                     f"{sim_psnr:8.2f}  {cam_psnr:8.2f}  "
                     f"{sim_pearson:11.5f}  {cam_pearson:11.5f}\n")

        # 11. Diagnostic dashboard
        if iteration % vis_interval == 0 or iteration == 1:
            metrics = {
                'loss': loss_val,
                'loss_citl': loss_citl.item(),
                'loss_sim': loss_sim.item(),
                'loss_tv': loss_tv.item(),
                'loss_bg': loss_bg.item(),
                'sim_PSNR': f'{sim_psnr:.2f} dB',
                'sim_blur_PSNR': f'{sim_blur_psnr:.2f} dB',
                'cam_blur_PSNR': f'{cam_psnr:.2f} dB',
                'sim_Pearson': sim_pearson,
                'cam_Pearson': cam_pearson,
                'best_cam_Pearson': f'{best_cam_pearson:.5f} (iter {best_iter})',
            }
            dash_path = os.path.join(snap_dir, f'dashboard_{iteration:04d}.png')
            with torch.no_grad():
                _save_citl_dashboard(
                    target, sim_norm, cam_tensor, phase,
                    iteration, metrics, dash_path,
                )
            # Save camera capture
            if cam_tensor is not None:
                save_tensor_as_png(
                    cam_tensor, os.path.join(snap_dir, f'camera_{iteration:04d}.png'))
            # Save phase mask
            phase_snap = torch.remainder(phase.detach(), 2 * math.pi)
            phase_to_raw_bmp(
                phase_snap, os.path.join(snap_dir, f'phase_{iteration:04d}.bmp'))

        # 12. Checkpoint
        if iteration % checkpoint_interval == 0:
            ckpt_path = os.path.join(run_dir, f'unet_citl_iter{iteration}.pth')
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cam_pearson': best_cam_pearson,
                'best_iter': best_iter,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Free memory
        del target, phase, sim_recon, sim_norm, sim_blur, tgt_blur
        del total_loss, loss_recon, loss_sim, loss_citl, loss_tv, loss_bg, loss_pv
        if cam_tensor is not None:
            del cam_tensor
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    # --- Final saves ---
    print("\n" + "=" * 70)
    print(f"CITL fine-tuning complete. Best cam_Pearson={best_cam_pearson:.5f} "
          f"at iter {best_iter}")

    # Save final model
    final_path = os.path.join(run_dir, 'unet_citl_final.pth')
    torch.save({
        'iteration': total_iters,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_cam_pearson': best_cam_pearson,
        'best_iter': best_iter,
    }, final_path)
    print(f"Final model saved: {final_path}")

    # Save EMA model separately (for inference)
    ema_model = HoloNet(n_channels=1).to(device)
    ema.apply_to(ema_model)
    torch.save(ema_model.state_dict(),
               os.path.join(run_dir, 'unet_citl_final_ema.pth'))
    del ema_model

    # Save all per-iteration metrics to npz for later analysis
    metrics_path = os.path.join(run_dir, 'training_metrics.npz')
    np.savez(metrics_path,
             iteration=np.arange(1, len(loss_history) + 1),
             loss=np.array(loss_history),
             citl_loss=np.array(citl_loss_history),
             sim_loss=np.array(sim_loss_history),
             tv_loss=np.array(tv_loss_history),
             bg_loss=np.array(bg_loss_history),
             sim_pearson=np.array(sim_pearson_history),
             cam_pearson=np.array(cam_pearson_history),
             sim_psnr=np.array(sim_psnr_history),
             cam_psnr=np.array(cam_psnr_history),
             best_cam_pearson=best_cam_pearson,
             best_iter=best_iter)
    print(f"Metrics saved: {metrics_path}")

    # Save training history plot (3-panel: loss, Pearson, PSNR)
    # Style matching plot_dual_metrics from common/utils.py
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    iters = np.arange(1, len(loss_history) + 1)
    window = max(25, len(loss_history) // 200) if len(loss_history) > 25 else 1
    kernel = np.ones(window) / window

    def _ma(arr):
        if window <= 1:
            return iters, np.array(arr)
        ma = np.convolve(arr, kernel, mode='valid')
        return np.arange(window, len(arr) + 1), ma

    # Panel 1: Loss components
    ax = axes[0]
    x_ma, y_ma = _ma(loss_history)
    ax.plot(x_ma, y_ma, color='mediumblue', linewidth=1.5, alpha=0.85, label='Total Loss')
    x_ma, y_ma = _ma(citl_loss_history)
    ax.plot(x_ma, y_ma, color='darkviolet', linewidth=1.5, label='CITL Loss')
    x_ma, y_ma = _ma(sim_loss_history)
    ax.plot(x_ma, y_ma, color='darkgreen', linewidth=1.8, label='Sim Loss')
    ax.set_ylabel('Combined Loss [-]')
    ax.set_title('CITL U-Net Fine-Tuning Progress')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=True, edgecolor='lightgray')

    # Panel 2: Pearson correlations (dual metrics)
    ax = axes[1]
    x_ma, y_ma = _ma(sim_pearson_history)
    ax.plot(x_ma, y_ma, color='darkviolet', linewidth=1.5, label='Sim Pearson')
    x_ma, y_ma = _ma(cam_pearson_history)
    ax.plot(x_ma, y_ma, color='darkgreen', linewidth=1.8, label='Camera Pearson')
    ax.axhline(best_cam_pearson, color='darkorange', linestyle='--', linewidth=1.5,
               label=f'Best cam r={best_cam_pearson:.4f}\n@ iteration {best_iter}')
    ax.set_ylabel('Pearson Correlation [-]')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=True, edgecolor='lightgray')

    # Panel 3: PSNR (dual metrics)
    ax = axes[2]
    x_ma, y_ma = _ma(sim_psnr_history)
    ax.plot(x_ma, y_ma, color='darkviolet', linewidth=1.5, label='Sim PSNR')
    x_ma, y_ma = _ma(cam_psnr_history)
    ax.plot(x_ma, y_ma, color='darkgreen', linewidth=1.8, label='Camera PSNR')
    ax.set_ylabel('PSNR [dB]')
    ax.set_xlabel('Iteration [-]')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=True, edgecolor='lightgray')

    max_x = len(loss_history)
    xlims = (0, max_x * 1.02)
    for a in axes:
        a.set_xlim(xlims)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Log summary
    with open(log_path, 'a') as lf:
        lf.write("-" * 90 + "\n")
        lf.write(f"DONE  total_iters={total_iters}  "
                 f"best_cam_pearson={best_cam_pearson:.5f}  "
                 f"best_iter={best_iter}\n")

    # Close hardware
    slm.close()
    print(f"All outputs saved to: {os.path.abspath(run_dir)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    citl_finetune_unet(
        checkpoint_path=CHECKPOINT_PATH,
        total_iters=TOTAL_ITERS,
        lr=LEARNING_RATE,
        grad_accum=GRAD_ACCUM,
        ema_decay=EMA_DECAY,
        weight_decay=WEIGHT_DECAY,
        blur_sigma=BLUR_SIGMA,
        correction_clip=(CORRECTION_MIN, CORRECTION_MAX),
        citl_weight=CITL_WEIGHT,
        sim_weight=SIM_WEIGHT,
        tv_weight=TV_WEIGHT,
        bg_weight=BG_WEIGHT,
        phase_var_weight=PHASE_VAR_WEIGHT,
        capture_avg=CAPTURE_AVG,
        stabilization_ms=STABILIZATION_MS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        vis_interval=VIS_INTERVAL,
        output_dir=OUTPUT_DIR,
    )


if __name__ == '__main__':
    main()
