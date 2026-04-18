# ──────────────────────────────────────────────────────────────────────────────
# File: measure_obstruction.py  Project: DNN-SLM Phase Mask Generation
# Purpose: Measure the real beam-stop obstruction size using SLM + camera
# ──────────────────────────────────────────────────────────────────────────────
"""
Determines the true extent of the beam-stop obstruction by generating a tall,
narrow rectangular target centred on the optical axis, running it through the
GS algorithm to compute a phase mask, displaying it on the SLM, and analysing
what is missing in the camera capture.

Strategy:
  1. Generate a tall, narrow vertical strip target spanning the full height
     of the image, centred on the optical axis (where the obstruction is).
  2. Use Gerchberg-Saxton to compute a phase mask for this target.
  3. Display the phase on the SLM and capture the camera image.
  4. Compare what was expected vs. what the camera sees.
  5. The "missing" region = the obstruction.
  6. Detect the edges of the missing region and report the obstruction
     boundaries in both pixel and fractional coordinates.

Outputs (saved to obstruction_measurement/):
  - target.png              : the generated target
  - camera_capture.png      : raw camera image
  - comparison.png          : side-by-side diagnostic figure
  - obstruction_params.json : measured obstruction parameters
  - edge_profile_x.png      : horizontal intensity profile through centre
  - edge_profile_y.png      : vertical intensity profile through centre

Run:  python measure_obstruction.py
Requires: SLM hardware + CinCam beam profiler
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'iterative'))

import json
import time
import datetime
import numpy as np
import torch
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import (
    PHYSICS_CONFIG, SLM_HEIGHT_PX, SLM_ACTIVE_WIDTH_PX,
    BEAM_STOP_X_CENTER, BEAM_STOP_X_HALF_WIDTH,
    BEAM_STOP_Y_START, BEAM_STOP_Y_END,
)
from slm_communicate import SLMController
from extract_camera_data import get_single_frame
from profile_adjustment import HitlPreprocessor
from gerchberg_saxton import fft_gerchberg_saxton
from slm_export import phase_to_slm_bmp

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       USER CONFIGURATION                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Target strip parameters (in fractional coordinates of the image)
# A tall narrow strip centred on the optical axis — obstruction will cut
# through the middle of it.
STRIP_X_CENTER     = 0.50     # centre of the strip (fraction of W)
STRIP_X_HALF_WIDTH = 0.12     # half-width (fraction of W) — wider than expected obstruction
STRIP_Y_START      = 0.05     # top of strip (fraction of H)
STRIP_Y_END        = 0.95     # bottom of strip (fraction of H)
STRIP_INTENSITY    = 1.0      # target intensity inside the strip

# Resolution (full SLM resolution)
RESOLUTION = (SLM_HEIGHT_PX, 1280)   # (H, W)

# GS parameters
GS_STEPS = 300
GS_GLOBAL_ITERS = 50

# Camera capture
CAPTURE_AVG       = 30        # frames to average (more = cleaner)
STABILIZATION_SEC = 2.0       # SLM settle time [s]

# Edge detection threshold (fraction of peak intensity in the strip)
EDGE_THRESHOLD = 0.15

# Output directory
OUTPUT_DIR = 'obstruction_measurement'

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    END OF USER CONFIGURATION                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def generate_strip_target(h, w, device):
    """Create a tall narrow vertical strip target [1,1,H,W]."""
    target = torch.zeros(1, 1, h, w, device=device)
    x0 = int(round(w * (STRIP_X_CENTER - STRIP_X_HALF_WIDTH)))
    x1 = int(round(w * (STRIP_X_CENTER + STRIP_X_HALF_WIDTH)))
    y0 = int(round(h * STRIP_Y_START))
    y1 = int(round(h * STRIP_Y_END))
    x0 = max(0, x0)
    x1 = min(w, x1)
    y0 = max(0, y0)
    y1 = min(h, y1)
    target[:, :, y0:y1, x0:x1] = STRIP_INTENSITY
    return target


def capture_averaged(n_frames):
    """Capture and average n_frames from the camera. Returns float32 HxW."""
    accum = None
    count = 0
    for _ in range(n_frames):
        frame, _ = get_single_frame()
        if frame is None:
            continue
        if accum is None:
            accum = frame.astype(np.float64)
        else:
            accum += frame.astype(np.float64)
        count += 1
    if count == 0:
        raise RuntimeError("All camera captures failed")
    return (accum / count).astype(np.float32)


def find_obstruction_edges(profile, threshold_frac=0.15):
    """Find the start and end of a gap in a 1-D intensity profile.

    Returns (gap_start, gap_end) indices where intensity drops below
    threshold_frac * peak, or None if no clear gap is found.
    """
    peak = profile.max()
    if peak <= 0:
        return None
    threshold = peak * threshold_frac

    # Find the region where profile exceeds threshold
    above = profile > threshold

    # Look for a gap (contiguous below-threshold region) in the middle
    n = len(profile)
    mid = n // 2

    # Search outward from the centre for the gap boundaries
    # Find left edge of gap (going left from centre until above threshold)
    gap_left = mid
    while gap_left > 0 and not above[gap_left]:
        gap_left -= 1
    # gap_left is now the last above-threshold pixel to the left of the gap

    gap_right = mid
    while gap_right < n - 1 and not above[gap_right]:
        gap_right += 1
    # gap_right is now the first above-threshold pixel to the right of the gap

    if gap_right - gap_left < 3:
        return None  # No clear gap

    return (gap_left, gap_right)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h, w = RESOLUTION
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output directory
    out_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR, f'run_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── Step 1: Generate the strip target ──
    print("\n[Step 1] Generating vertical strip target ...")
    target = generate_strip_target(h, w, device)
    target_np = target.squeeze().cpu().numpy()

    # Save target
    cv2.imwrite(os.path.join(out_dir, 'target.png'),
                np.clip(target_np * 255, 0, 255).astype(np.uint8))
    print(f"  Strip: x=[{STRIP_X_CENTER - STRIP_X_HALF_WIDTH:.3f}, "
          f"{STRIP_X_CENTER + STRIP_X_HALF_WIDTH:.3f}]  "
          f"y=[{STRIP_Y_START:.3f}, {STRIP_Y_END:.3f}]")

    # ── Step 2: Compute GS phase mask ──
    print("\n[Step 2] Running Gerchberg-Saxton ...")
    phase = fft_gerchberg_saxton(
        target, waist_ratio=PHYSICS_CONFIG.get('waist_ratio', 0.45),
        steps=GS_STEPS, global_iters=GS_GLOBAL_ITERS,
    )
    print(f"  Phase shape: {phase.shape}")

    # Save the SLM-ready BMP
    slm_bmp_path = os.path.join(out_dir, 'phase_mask.bmp')
    phase_to_slm_bmp(phase, slm_bmp_path)

    # ── Step 3: Display on SLM and capture background ──
    print("\n[Step 3] Hardware capture ...")
    with SLMController() as slm:
        # Background capture (flat phase)
        print("  Capturing background (flat phase) ...")
        slm.display_bmp(os.path.join(out_dir, 'phase_mask.bmp'))
        # Actually display flat phase first for background
        flat_phase = torch.zeros(1, 1, h, w, device=device)
        slm.display_phase_tensor(flat_phase)
        time.sleep(STABILIZATION_SEC * 2)
        bg = capture_averaged(CAPTURE_AVG)

        # Display the strip phase mask
        print("  Displaying strip phase mask ...")
        slm.display_bmp(slm_bmp_path)
        time.sleep(STABILIZATION_SEC * 2)
        capture = capture_averaged(CAPTURE_AVG)

    # Background subtraction
    capture_sub = np.clip(capture - bg, 0, None)

    # Save raw capture
    cap_vis = capture_sub / (capture_sub.max() + 1e-8) * 255
    cv2.imwrite(os.path.join(out_dir, 'camera_capture.png'),
                cap_vis.astype(np.uint8))

    # ── Step 4: Geometric correction (apply calibration) ──
    print("\n[Step 4] Applying geometric calibration ...")
    preprocessor = HitlPreprocessor()
    _repo_root = os.path.join(os.path.dirname(__file__), '..')
    _latest_cal = os.path.normpath(
        os.path.join(_repo_root, 'calibration_outputs', 'latest.json'))
    if preprocessor.aligner.load(_latest_cal):
        ch, cw = capture_sub.shape[:2]
        cos_t = np.cos(preprocessor.aligner.rotation)
        sin_t = np.sin(preprocessor.aligner.rotation)
        s = preprocessor.aligner.scale
        M = np.float32([
            [s * cos_t, -s * sin_t, preprocessor.aligner.shift_x],
            [s * sin_t,  s * cos_t, preprocessor.aligner.shift_y]
        ])
        capture_aligned = cv2.warpAffine(
            capture_sub, M, (cw, ch), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(np.float32)
    else:
        print("  WARNING: No calibration — using raw camera image.")
        capture_aligned = capture_sub

    # Resize to simulation resolution
    capture_resized = cv2.resize(capture_aligned, (w, h),
                                 interpolation=cv2.INTER_LINEAR)

    # ── Step 5: Analyse the obstruction ──
    print("\n[Step 5] Analysing obstruction ...")

    # Horizontal profile: average across the strip's vertical extent
    # Use the middle portion of the strip for a clean profile
    strip_y0 = int(round(h * STRIP_Y_START))
    strip_y1 = int(round(h * STRIP_Y_END))
    strip_mid_y0 = int(round(h * 0.55))  # just below expected obstruction start
    strip_mid_y1 = int(round(h * 0.85))  # well into expected obstruction

    # Profile in the blocked region (lower half)
    profile_blocked = capture_resized[strip_mid_y0:strip_mid_y1, :].mean(axis=0)

    # Profile in the unblocked region (upper half, above obstruction)
    strip_upper_y0 = int(round(h * 0.15))
    strip_upper_y1 = int(round(h * 0.45))
    profile_upper = capture_resized[strip_upper_y0:strip_upper_y1, :].mean(axis=0)

    # Vertical profile: average across the strip's horizontal extent
    strip_x0 = int(round(w * (STRIP_X_CENTER - STRIP_X_HALF_WIDTH)))
    strip_x1 = int(round(w * (STRIP_X_CENTER + STRIP_X_HALF_WIDTH)))
    # For vertical profile, use a column range slightly left of centre to avoid
    # the x-centre of the obstruction
    # Actually we want through the centre to see where it starts vertically
    profile_v_left_x0 = int(round(w * (STRIP_X_CENTER - STRIP_X_HALF_WIDTH)))
    profile_v_left_x1 = int(round(w * (STRIP_X_CENTER - 0.03)))
    if profile_v_left_x1 > profile_v_left_x0:
        profile_v_left = capture_resized[:, profile_v_left_x0:profile_v_left_x1].mean(axis=1)
    else:
        profile_v_left = capture_resized[:, strip_x0:strip_x1].mean(axis=1)

    # Also get vertical profile through the centre (blocked column)
    centre_x0 = int(round(w * (STRIP_X_CENTER - 0.01)))
    centre_x1 = int(round(w * (STRIP_X_CENTER + 0.01)))
    profile_v_centre = capture_resized[:, centre_x0:centre_x1].mean(axis=1)

    # ── Find horizontal extent of obstruction ──
    # In the blocked row band, look for where the strip intensity drops
    gap_h = find_obstruction_edges(profile_blocked, EDGE_THRESHOLD)
    if gap_h:
        obs_x_left_px = gap_h[0]
        obs_x_right_px = gap_h[1]
        obs_x_left_frac = obs_x_left_px / w
        obs_x_right_frac = obs_x_right_px / w
        obs_x_center_frac = (obs_x_left_frac + obs_x_right_frac) / 2
        obs_x_half_width_frac = (obs_x_right_frac - obs_x_left_frac) / 2
        obs_x_width_px = obs_x_right_px - obs_x_left_px
        print(f"  Horizontal: x=[{obs_x_left_px}..{obs_x_right_px}] px "
              f"= [{obs_x_left_frac:.4f}..{obs_x_right_frac:.4f}] frac")
        print(f"    Centre: {obs_x_center_frac:.4f}  Half-width: {obs_x_half_width_frac:.4f}")
        print(f"    Width: {obs_x_width_px} px")
    else:
        print("  WARNING: Could not detect horizontal obstruction edges")
        obs_x_center_frac = BEAM_STOP_X_CENTER
        obs_x_half_width_frac = BEAM_STOP_X_HALF_WIDTH

    # ── Find vertical extent of obstruction ──
    # Compare the vertical profile through the centre vs. left edge
    # Where the centre profile drops but the left doesn't = obstruction start
    if profile_v_left.max() > 0:
        profile_v_left_norm = profile_v_left / profile_v_left.max()
    else:
        profile_v_left_norm = profile_v_left

    if profile_v_centre.max() > 0:
        profile_v_centre_norm = profile_v_centre / profile_v_centre.max()
    else:
        profile_v_centre_norm = profile_v_centre

    # Find where the centre profile drops to < threshold while left is still bright
    threshold = EDGE_THRESHOLD
    # Search downward from the top of the strip
    obs_y_start_px = None
    for y in range(strip_y0, strip_y1):
        if (profile_v_left_norm[y] > threshold * 2 and
                profile_v_centre_norm[y] < threshold):
            obs_y_start_px = y
            break

    # The obstruction extends to the bottom of the frame (known from physics)
    obs_y_end_px = h

    if obs_y_start_px is not None:
        obs_y_start_frac = obs_y_start_px / h
        obs_y_end_frac = obs_y_end_px / h
        print(f"  Vertical: y=[{obs_y_start_px}..{obs_y_end_px}] px "
              f"= [{obs_y_start_frac:.4f}..{obs_y_end_frac:.4f}] frac")
    else:
        print("  WARNING: Could not detect vertical obstruction start")
        obs_y_start_frac = BEAM_STOP_Y_START
        obs_y_end_frac = BEAM_STOP_Y_END

    # ── Step 6: Report ──
    print("\n" + "=" * 60)
    print("OBSTRUCTION MEASUREMENT RESULTS")
    print("=" * 60)
    print(f"  Current config values:")
    print(f"    BEAM_STOP_X_CENTER     = {BEAM_STOP_X_CENTER}")
    print(f"    BEAM_STOP_X_HALF_WIDTH = {BEAM_STOP_X_HALF_WIDTH}")
    print(f"    BEAM_STOP_Y_START      = {BEAM_STOP_Y_START}")
    print(f"    BEAM_STOP_Y_END        = {BEAM_STOP_Y_END}")
    print()
    print(f"  Measured values:")
    print(f"    BEAM_STOP_X_CENTER     = {obs_x_center_frac:.4f}")
    print(f"    BEAM_STOP_X_HALF_WIDTH = {obs_x_half_width_frac:.4f}")
    print(f"    BEAM_STOP_Y_START      = {obs_y_start_frac:.4f}")
    print(f"    BEAM_STOP_Y_END        = {obs_y_end_frac:.4f}")
    print()

    # Differences
    dx_center = obs_x_center_frac - BEAM_STOP_X_CENTER
    dx_width = obs_x_half_width_frac - BEAM_STOP_X_HALF_WIDTH
    dy_start = obs_y_start_frac - BEAM_STOP_Y_START
    print(f"  Differences (measured - config):")
    print(f"    X centre:     {dx_center:+.4f}  ({dx_center * w:+.1f} px)")
    print(f"    X half-width: {dx_width:+.4f}  ({dx_width * w:+.1f} px)")
    print(f"    Y start:      {dy_start:+.4f}  ({dy_start * h:+.1f} px)")
    print("=" * 60)

    # ── Save results ──
    results = {
        'timestamp': timestamp,
        'resolution': list(RESOLUTION),
        'current_config': {
            'BEAM_STOP_X_CENTER': BEAM_STOP_X_CENTER,
            'BEAM_STOP_X_HALF_WIDTH': BEAM_STOP_X_HALF_WIDTH,
            'BEAM_STOP_Y_START': BEAM_STOP_Y_START,
            'BEAM_STOP_Y_END': BEAM_STOP_Y_END,
        },
        'measured': {
            'BEAM_STOP_X_CENTER': round(obs_x_center_frac, 4),
            'BEAM_STOP_X_HALF_WIDTH': round(obs_x_half_width_frac, 4),
            'BEAM_STOP_Y_START': round(obs_y_start_frac, 4),
            'BEAM_STOP_Y_END': round(obs_y_end_frac, 4),
        },
        'differences': {
            'x_center': round(dx_center, 4),
            'x_half_width': round(dx_width, 4),
            'y_start': round(dy_start, 4),
        },
    }

    with open(os.path.join(out_dir, 'obstruction_params.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── Diagnostic plots ──
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Target
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(target_np, cmap='hot', vmin=0)
    ax1.set_title('Target (strip)')
    ax1.set_xlabel('x [px]')
    ax1.set_ylabel('y [px]')

    # 2. Camera capture (aligned, resized)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(capture_resized, cmap='hot', vmin=0)
    ax2.set_title('Camera (aligned)')
    ax2.set_xlabel('x [px]')
    ax2.set_ylabel('y [px]')

    # 3. Overlay: target contour on camera
    ax3 = fig.add_subplot(gs[0, 2])
    cam_norm = capture_resized / (capture_resized.max() + 1e-8)
    ax3.imshow(cam_norm, cmap='hot', vmin=0, vmax=1)
    # Draw current config beam-stop rectangle
    cfg_x0 = w * (BEAM_STOP_X_CENTER - BEAM_STOP_X_HALF_WIDTH)
    cfg_x1 = w * (BEAM_STOP_X_CENTER + BEAM_STOP_X_HALF_WIDTH)
    cfg_y0 = h * BEAM_STOP_Y_START
    cfg_y1 = h * BEAM_STOP_Y_END
    rect_cfg = plt.Rectangle((cfg_x0, cfg_y0), cfg_x1 - cfg_x0, cfg_y1 - cfg_y0,
                              linewidth=2, edgecolor='cyan', facecolor='none',
                              linestyle='--', label='Config')
    ax3.add_patch(rect_cfg)
    # Draw measured beam-stop rectangle
    meas_x0 = w * (obs_x_center_frac - obs_x_half_width_frac)
    meas_x1 = w * (obs_x_center_frac + obs_x_half_width_frac)
    meas_y0 = h * obs_y_start_frac
    meas_y1 = h * obs_y_end_frac
    rect_meas = plt.Rectangle((meas_x0, meas_y0), meas_x1 - meas_x0, meas_y1 - meas_y0,
                               linewidth=2, edgecolor='lime', facecolor='none',
                               linestyle='-', label='Measured')
    ax3.add_patch(rect_meas)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_title('Overlay: config vs measured')

    # 4. Horizontal profile (blocked rows)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(profile_blocked, 'r-', linewidth=1.5,
             label=f'Blocked rows (y={strip_mid_y0}..{strip_mid_y1})')
    ax4.plot(profile_upper, 'b-', linewidth=1.5, alpha=0.7,
             label=f'Upper rows (y={strip_upper_y0}..{strip_upper_y1})')
    if gap_h:
        ax4.axvline(obs_x_left_px, color='lime', linestyle='--', label='Measured edges')
        ax4.axvline(obs_x_right_px, color='lime', linestyle='--')
    ax4.axvline(w * (BEAM_STOP_X_CENTER - BEAM_STOP_X_HALF_WIDTH),
                color='cyan', linestyle=':', label='Config edges')
    ax4.axvline(w * (BEAM_STOP_X_CENTER + BEAM_STOP_X_HALF_WIDTH),
                color='cyan', linestyle=':')
    ax4.set_xlabel('x [px]')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Horizontal profile (avg over row bands)')
    ax4.legend(fontsize=9)
    ax4.set_xlim(w * 0.3, w * 0.7)  # zoom in around centre

    # 5. Vertical profile
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(profile_v_left, 'b-', linewidth=1.5,
             label='Left of centre (unblocked columns)')
    ax5.plot(profile_v_centre, 'r-', linewidth=1.5,
             label='Centre columns (blocked)')
    if obs_y_start_px is not None:
        ax5.axvline(obs_y_start_px, color='lime', linestyle='--',
                     label='Measured Y start')
    ax5.axvline(h * BEAM_STOP_Y_START, color='cyan', linestyle=':',
                label='Config Y start')
    ax5.set_xlabel('y [px]')
    ax5.set_ylabel('Intensity')
    ax5.set_title('Vertical profile (avg over column bands)')
    ax5.legend(fontsize=9)

    plt.suptitle(f'Obstruction Measurement — {timestamp}', fontsize=14)
    fig.savefig(os.path.join(out_dir, 'comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    print(f"\nResults saved to: {os.path.abspath(out_dir)}")
    print("If the measured values look correct, update config.py accordingly.")


if __name__ == '__main__':
    main()
