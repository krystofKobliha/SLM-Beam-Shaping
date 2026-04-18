# ──────────────────────────────────────────────────────────────────────────────
# # File: citl_finetune.py       Project: DNN-SLM Phase Mask Generation
# Purpose: Camera-in-the-loop (CITL) phase mask fine-tuning (Camera-GS)
# ──────────────────────────────────────────────────────────────────────────────
"""
Fine-tunes a GS-generated phase mask on the real optical setup using
iterative target pre-compensation (Camera-GS).

Each iteration captures the SLM output with a camera, computes the
multiplicative ratio between the effective target and the camera image,
adjusts the GS target by that ratio, and re-runs Gerchberg-Saxton.
This closes the loop through real hardware without requiring differentiable
gradients.

Requires hardware (SLM + camera).  If unavailable, returns the input phase
unchanged.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import math
import time
import numpy as np
import torch
import cv2

from slm_export import phase_to_raw_bmp, phase_to_slm_bmp, save_tensor_as_png

# Optional hardware imports â€” gracefully fall back when unavailable
_HW_AVAILABLE = False
try:
    from slm_communicate import SLMController
    from extract_camera_data import get_single_frame
    from profile_adjustment import HitlPreprocessor
    _HW_AVAILABLE = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Camera-GS: two-phase iterative target pre-compensation
# ---------------------------------------------------------------------------
def camera_gs_finetune(target_amp, device,
                       initial_phase=None,
                       outer_iters=15, gs_steps=300, gs_global_iters=50,
                       shape_iters=10,
                       blur_sigma_shape=0, blur_sigma_fill=0,
                       correction_strength=0.1,
                       capture_avg=30, stabilization_ms=1500,
                       outdir=None):
    """
    Two-phase target pre-compensation (shape, then fill).

    Phase 1 â€” Shape (iterations 1..shape_iters):
      Large blur sigma.  Fixes coarse shape distortion: edge contour,
      position offset, ellipticity, broad asymmetry.

    Phase 2 â€” Fill (iterations shape_iters+1..outer_iters):
      Small blur sigma.  Fills in internal intensity uniformity while
      preserving the corrected shape.

    Both phases use:
      - Normalized-convolution blur: blur(x*mask)/blur(mask) â€” correct
        local averages at support edges, no dark-background leakage.
      - Multiplicative ratio correction:
            ratio = effective_nc / camera_nc   (clamped to [0.5, 2.0])
            effective_target *= ratio^alpha
        With alpha=0.1, each iteration nudges gently (~Â±7 %).
      - Comparison against the EFFECTIVE target (evolving reference),
        preventing compounding when GS has structural bias.

    Args:
        target_amp:          [1,1,H,W] original target intensity.
        device:              torch device.
        initial_phase:       [1,1,H,W] pre-computed GS phase (optional).
        outer_iters:         total captureâ†’correctâ†’GS cycles.
        gs_steps:            GS iterations per cycle.
        gs_global_iters:     GS global-constraint iterations.
        shape_iters:         how many of outer_iters use large blur (shape).
        blur_sigma_shape:    shape-phase blur sigma [px] (0 = auto).
        blur_sigma_fill:     fill-phase blur sigma [px] (0 = auto).
        correction_strength: exponent alpha for ratio^alpha (0.1 typical).
        capture_avg:         camera frames to average per capture.
        stabilization_ms:    SLM settle time [ms].
        outdir:              output directory.

    Returns:
        Best phase [1,1,H,W] in [0, 2Ď€].
    """
    if not _HW_AVAILABLE:
        print("WARNING: SLM/camera hardware not available -- skipping Camera-GS.")
        from gerchberg_saxton import fft_gerchberg_saxton
        if initial_phase is not None:
            return initial_phase
        return fft_gerchberg_saxton(target_amp, steps=gs_steps,
                                    global_iters=gs_global_iters)

    from gerchberg_saxton import fft_gerchberg_saxton
    from fft_physics import FFTPhysics
    from config import PHYSICS_CONFIG
    import torch.nn.functional as F

    _, _, h, w = target_amp.shape
    stab_s = stabilization_ms / 1000.0
    _repo_root = os.path.join(os.path.dirname(__file__), '..')
    _latest_cal = os.path.normpath(
        os.path.join(_repo_root, 'calibration_outputs', 'latest.json'))
    waist_ratio = PHYSICS_CONFIG.get('waist_ratio', 0.45)

    # Ratio clamp limits for multiplicative correction
    RATIO_MIN, RATIO_MAX = 0.5, 2.0

    # --- Blur helper ---
    def _blur(x, sigma):
        """Gaussian blur a [1,1,H,W] tensor."""
        if sigma < 0.5:
            return x
        ks = int(6 * sigma + 1) | 1
        pad = ks // 2
        coords = torch.arange(ks, device=x.device, dtype=x.dtype) - pad
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        out = F.conv2d(x, g.reshape(1, 1, 1, ks), padding=(0, pad))
        out = F.conv2d(out, g.reshape(1, 1, ks, 1), padding=(pad, 0))
        return out

    def _nc_blur(x_2d, mask_f, mask_4d, sigma):
        """Normalized-convolution blur: blur(x*mask)/blur(mask).

        Returns correct local averages even at support edges, without
        dark-background leakage.  x_2d is [H,W], mask_f is [H,W] float.
        """
        num = _blur((x_2d * mask_f).unsqueeze(0).unsqueeze(0), sigma).squeeze()
        den = _blur(mask_4d, sigma).squeeze().clamp(min=1e-6)
        return num / den

    # --- Hardware init ---
    slm = SLMController()
    preprocessor = HitlPreprocessor()
    if not preprocessor.aligner.load(_latest_cal):
        print("Camera-GS WARNING: No calibration -- geometric correction disabled.")

    # Background capture
    print("Camera-GS: Capturing background ...")
    flat_phase = torch.zeros(1, 1, h, w, device=device)
    slm.display_phase_tensor(flat_phase)
    time.sleep(stab_s * 3)
    preprocessor.capture_background(get_single_frame, n_frames=capture_avg * 2)

    # --- Snapshot/log directory ---
    snap_dir = None
    log_path = None
    if outdir is not None:
        snap_dir = os.path.join(outdir, 'citl_snapshots')
        os.makedirs(snap_dir, exist_ok=True)
        log_path = os.path.join(outdir, 'citl_loss_log.txt')

    # --- Fixed support mask from original target ---
    with torch.no_grad():
        orig_raw = target_amp.squeeze()
        orig_support = (orig_raw > 0.05 * orig_raw.max())
    support_f = orig_support.float()
    mask_4d = support_f.unsqueeze(0).unsqueeze(0)

    # --- Distance transform for auto-sigma and edge ramp ---
    support_np = orig_support.cpu().numpy().astype(np.uint8)
    dist_np = cv2.distanceTransform(support_np, cv2.DIST_L2, 5)
    half_thickness = max(float(dist_np.max()), 3.0)

    # Auto-compute blur sigmas from feature size
    sigma_shape = blur_sigma_shape if blur_sigma_shape > 0 \
        else round(half_thickness * 0.5, 1)
    sigma_fill = blur_sigma_fill if blur_sigma_fill > 0 \
        else min(10.0, round(half_thickness * 0.15, 1))
    sigma_fill = min(sigma_fill, sigma_shape)
    print(f"  Auto-sigma: half_thickness={half_thickness:.1f}px -> "
          f"sigma_shape={sigma_shape}, sigma_fill={sigma_fill}")

    # Edge ramp: controls how deep corrections reach into boundary.
    #   Shape phase uses a wide ramp (half_thickness * 0.4, min 8 px) so
    #   corrections reach the full boundary region.
    #   Fill phase uses a narrow ramp (3 px) â€” only interior uniformity.
    _EDGE_RAMP_SHAPE_PX = max(8.0, half_thickness * 0.4)
    _EDGE_RAMP_FILL_PX = 3.0
    edge_weight_shape = torch.from_numpy(
        np.clip(dist_np / _EDGE_RAMP_SHAPE_PX, 0.0, 1.0)
    ).float().to(device)
    edge_weight_fill = torch.from_numpy(
        np.clip(dist_np / _EDGE_RAMP_FILL_PX, 0.0, 1.0)
    ).float().to(device)

    # Metric blur sigma (small, fixed â€” smooths speckle for evaluation)
    sigma_metric = max(sigma_fill, 3.0)

    # Write log header
    if log_path is not None:
        with open(log_path, 'w') as lf:
            lf.write("Camera-GS: two-phase target pre-compensation\n")
            lf.write(f"outer_iters={outer_iters}  gs_steps={gs_steps}  "
                     f"shape_iters={shape_iters}  "
                     f"sigma_shape={sigma_shape}  sigma_fill={sigma_fill}  "
                     f"alpha={correction_strength}  "
                     f"clip=[{RATIO_MIN},{RATIO_MAX}]  "
                     f"capture_avg={capture_avg}\n")
            lf.write("-" * 70 + "\n")
            lf.write(f"{'iter':>5s}  {'phase':>6s}  {'pearson':>8s}  "
                     f"{'unif_cv':>8s}  {'tgt_rng':>14s}\n")
            lf.write("-" * 70 + "\n")

    # --- Use pre-computed phase or run GS ---
    effective_target = target_amp.clone()
    physics = FFTPhysics(PHYSICS_CONFIG, (h, w), device)

    if initial_phase is not None:
        print("  Using pre-computed GS phase from pipeline.")
        phase = initial_phase.clone()
    else:
        print("  Running initial GS ...")
        phase = fft_gerchberg_saxton(effective_target, waist_ratio=waist_ratio,
                                     steps=gs_steps, global_iters=gs_global_iters,
                                     silent=True)

    # --- Display initial phase and calibrate intensity mapping ---
    slm.display_phase_tensor(phase)
    time.sleep(stab_s * 2)
    cal_frames = []
    for _ in range(capture_avg * 2):
        f, _ = get_single_frame()
        if f is not None:
            cal_frames.append(f)
    if cal_frames:
        avg_cal = np.mean(cal_frames, axis=0).astype(np.uint8)
        # Apply same preprocessing as camera_to_tensor (bg subtract +
        # geometric correction + resize) so that pixel correspondence
        # between sim and camera is correct for the intensity mapping fit.
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
                cam_proc, M, (cw, ch),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0).astype(np.float32)
        cam_resized = cv2.resize(cam_proc, (w, h), interpolation=cv2.INTER_LINEAR)
        with torch.no_grad():
            sim_np = physics(phase).squeeze().cpu().numpy()
        sim_max = sim_np.max()
        if sim_max > 0:
            sim_np /= sim_max
        preprocessor.compute_intensity_mapping(
            [sim_np], [cam_resized / max(cam_resized.max(), 1e-8)])
        print(f"  Intensity mapping: a={preprocessor.intensity_a:.4f}, "
              f"b={preprocessor.intensity_b:.6f}")

    # --- Helper: compute metrics (Pearson + CV on support via nc_blur) ---
    def _compute_metrics(cam_tensor_2d):
        """Return (pearson, unif_cv) for a camera capture vs original target.

        Uses normalized-convolution blur within the support mask so that
        edge pixels get correct local averages without dark-background leakage.
        Both Pearson and CV are computed on the support region only.
        """
        with torch.no_grad():
            # NC-blur smooths speckle while respecting support boundary
            cam_nc = _nc_blur(cam_tensor_2d, support_f, mask_4d, sigma_metric)
            tgt_nc = _nc_blur(orig_raw, support_f, mask_4d, sigma_metric)
            c_v = cam_nc[orig_support]
            t_v = tgt_nc[orig_support]

            # Energy-scale camera to match target on support
            scale = t_v.sum() / (c_v.sum() + 1e-8)
            c_v = c_v * scale

            # Pearson correlation (support-only)
            c_m, t_m = c_v.mean(), t_v.mean()
            cov = ((c_v - c_m) * (t_v - t_m)).mean()
            pearson = (cov / (c_v.std() * t_v.std() + 1e-8)).item()

            # Uniformity CV: std/mean of the energy-scaled ratio
            r_v = c_v / (t_v + 1e-8)
            unif_cv = (r_v.std() / (r_v.mean() + 1e-8)).item()
        return pearson, unif_cv

    # --- BASELINE CAPTURE ---
    print("  Baseline capture (no correction) ...")
    time.sleep(stab_s)
    baseline_frames = []
    for _ in range(capture_avg):
        f, _ = get_single_frame()
        if f is not None:
            baseline_frames.append(f)

    bl_pearson = -1.0
    if baseline_frames:
        avg_baseline = np.mean(baseline_frames, axis=0).astype(np.uint8)
        baseline_cam = preprocessor.camera_to_tensor(avg_baseline, device, (h, w))
        bl_pearson, bl_unif = _compute_metrics(baseline_cam.squeeze())

        print(f"  BASELINE: Pearson={bl_pearson:.4f}  uniformity_CV={bl_unif:.4f}")

        if snap_dir is not None:
            save_tensor_as_png(baseline_cam.detach(),
                               os.path.join(snap_dir, 'cgs000_baseline_camera.png'))
            phase_snap = torch.remainder(phase.detach(), 2 * math.pi)
            phase_to_raw_bmp(phase_snap,
                             os.path.join(snap_dir, 'cgs000_baseline_phase.bmp'))
            np.save(os.path.join(snap_dir, 'cgs000_baseline_phase.npy'),
                    phase_snap.cpu().squeeze().numpy())

        if log_path is not None:
            with open(log_path, 'a') as lf:
                lf.write(f"    0  {bl_pearson:8.5f}  "
                         f"{bl_unif:8.5f}  {'[0.000,1.000]':>14s}\n")

    best_pearson = bl_pearson
    best_phase = phase.clone()

    # --- One-shot coarse correction from baseline ---
    if baseline_frames:
        with torch.no_grad():
            bl_cam_2d = baseline_cam.squeeze().clamp(min=0)
            bl_nc = _nc_blur(bl_cam_2d, support_f, mask_4d, sigma_shape)
            eff_nc_bl = _nc_blur(effective_target.squeeze(), support_f,
                                 mask_4d, sigma_shape)

            # Energy-scale baseline to match target on support
            bl_sum = bl_nc[orig_support].sum()
            eff_bl_sum = eff_nc_bl[orig_support].sum()
            bl_scaled = bl_nc * (eff_bl_sum / (bl_sum + 1e-8))

            # Partial ratio correction â€” strong enough to remove most of the
            # systematic gradient but gentle enough not to overshoot.
            ratio_bl = eff_nc_bl / (bl_scaled + 1e-8)
            ratio_bl = ratio_bl.clamp(RATIO_MIN, RATIO_MAX)

            correction_bl = ratio_bl ** 0.5  # half-strength one-shot

            eff = effective_target.squeeze().clone()
            eff[orig_support] *= correction_bl[orig_support]
            eff[~orig_support] = 0.0
            eff = eff.clamp(min=0)
            effective_target = eff.unsqueeze(0).unsqueeze(0)

        # Re-run GS on corrected target
        phase = fft_gerchberg_saxton(effective_target, waist_ratio=waist_ratio,
                                     steps=gs_steps, global_iters=gs_global_iters,
                                     silent=True)
        tgt_rng = f"[{effective_target.min().item():.3f},{effective_target.max().item():.3f}]"
        print(f"  One-shot baseline correction applied: target_range={tgt_rng}")

        if snap_dir is not None:
            save_tensor_as_png(effective_target.detach(),
                               os.path.join(snap_dir, 'oneshot_corrected_target.png'))

    # --- Main loop (all iters use sigma_shape to avoid speckle pickup) ---
    print(f"Camera-GS: {outer_iters} iters (sigma={sigma_shape}, "
          f"edge_ramp={_EDGE_RAMP_SHAPE_PX:.0f}px), alpha={correction_strength}")
    best_meas = None  # best camera capture tensor

    for outer in range(outer_iters):
        sigma = sigma_shape
        edge_weight = edge_weight_shape

        # 1. Capture current result
        slm.display_phase_tensor(phase)
        time.sleep(stab_s)
        frames = []
        for _ in range(capture_avg):
            f, _ = get_single_frame()
            if f is not None:
                frames.append(f)
        if not frames:
            print(f"  iter {outer+1}: capture failed -- skipping")
            continue
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        cam_tensor = preprocessor.camera_to_tensor(avg_frame, device, (h, w))

        # 2. Multiplicative ratio correction within original support only
        with torch.no_grad():
            cam_2d = cam_tensor.squeeze().clamp(min=0)

            # Normalized-convolution blur of both effective target and
            # camera capture within support mask.
            eff_nc = _nc_blur(effective_target.squeeze(), support_f,
                              mask_4d, sigma)
            cam_nc = _nc_blur(cam_2d, support_f, mask_4d, sigma)

            # Scale camera energy to match effective target on support
            eff_sum = eff_nc[orig_support].sum()
            cam_sum = cam_nc[orig_support].sum()
            energy_scale = eff_sum / (cam_sum + 1e-8)
            cam_nc_scaled = cam_nc * energy_scale

            # Ratio: where camera < eff â†’ ratio > 1 â†’ boost
            ratio = eff_nc / (cam_nc_scaled + 1e-8)
            ratio = ratio.clamp(RATIO_MIN, RATIO_MAX)

            # Gentle multiplicative correction with phase-dependent edge ramp
            alpha = correction_strength
            correction = ratio ** alpha
            corr_blend = 1.0 + edge_weight * (correction - 1.0)

            eff = effective_target.squeeze().clone()
            eff[orig_support] *= corr_blend[orig_support]
            # Enforce strict zero outside support â€” prevents halo buildup
            eff[~orig_support] = 0.0
            eff = eff.clamp(min=0)

            effective_target = eff.unsqueeze(0).unsqueeze(0)

        # 3. Run GS on corrected target
        phase = fft_gerchberg_saxton(effective_target, waist_ratio=waist_ratio,
                                     steps=gs_steps,
                                     global_iters=gs_global_iters,
                                     silent=True)

        # 4. Capture again for metrics (display new phase first)
        slm.display_phase_tensor(phase)
        time.sleep(stab_s)
        mframes = []
        for _ in range(capture_avg):
            f, _ = get_single_frame()
            if f is not None:
                mframes.append(f)
        if mframes:
            avg_m = np.mean(mframes, axis=0).astype(np.uint8)
            meas = preprocessor.camera_to_tensor(avg_m, device, (h, w))
        else:
            meas = cam_tensor  # fallback

        pearson, unif_cv = _compute_metrics(meas.squeeze())

        # 5. Track best
        if pearson > best_pearson:
            best_pearson = pearson
            best_phase = phase.clone()
            best_meas = meas.clone()

        # 6. Log
        eff_min = effective_target.min().item()
        eff_max_val = effective_target.max().item()
        tgt_rng = f"[{eff_min:.3f},{eff_max_val:.3f}]"
        print(f"  iter {outer+1:3d}/{outer_iters}  "
              f"sigma={sigma:5.1f}  Pearson={pearson:.4f}  CV={unif_cv:.4f}  "
              f"target_range={tgt_rng}")

        if log_path is not None:
            with open(log_path, 'a') as lf:
                lf.write(f"{outer+1:5d}  "
                         f"{pearson:8.5f}  {unif_cv:8.5f}  "
                         f"{tgt_rng:>14s}\n")

        # 7. Snapshots
        if snap_dir is not None:
            phase_snap = torch.remainder(phase.detach(), 2 * math.pi)
            phase_to_raw_bmp(phase_snap,
                             os.path.join(snap_dir, f'cgs{outer+1:03d}_phase.bmp'))
            np.save(os.path.join(snap_dir, f'cgs{outer+1:03d}_phase.npy'),
                    phase_snap.cpu().squeeze().numpy())
            save_tensor_as_png(meas.detach(),
                               os.path.join(snap_dir, f'cgs{outer+1:03d}_camera.png'))
            meas_blur_4d = _blur(meas, sigma_metric)
            save_tensor_as_png(meas_blur_4d.detach(),
                               os.path.join(snap_dir, f'cgs{outer+1:03d}_cam_blur.png'))
            save_tensor_as_png(effective_target.detach(),
                               os.path.join(snap_dir, f'cgs{outer+1:03d}_eff_target.png'))

    # --- Final saves (best iteration) ---
    if outdir is not None:
        save_tensor_as_png(effective_target.detach(),
                           os.path.join(outdir, 'corrected_target.png'))
        # Save best phase mask (raw, SLM-ready, numpy)
        best_phase_snap = torch.remainder(best_phase.detach(), 2 * math.pi)
        phase_to_raw_bmp(best_phase_snap,
                         os.path.join(outdir, 'best_phase.bmp'))
        phase_to_slm_bmp(best_phase_snap,
                         os.path.join(outdir, 'best_phase_slm.bmp'))
        np.save(os.path.join(outdir, 'best_phase.npy'),
                best_phase_snap.cpu().squeeze().numpy())
        # Save best camera capture
        if best_meas is not None:
            save_tensor_as_png(best_meas.detach(),
                               os.path.join(outdir, 'best_camera.png'))
            best_meas_blur = _blur(best_meas, sigma_metric)
            save_tensor_as_png(best_meas_blur.detach(),
                               os.path.join(outdir, 'best_camera_blur.png'))
        # Save best simulated reconstruction
        with torch.no_grad():
            best_sim = physics(best_phase).squeeze()
        save_tensor_as_png(best_sim.unsqueeze(0).unsqueeze(0).detach(),
                           os.path.join(outdir, 'best_simulated.png'))

    slm.close()
    if log_path is not None:
        with open(log_path, 'a') as lf:
            lf.write("-" * 70 + "\n")
            lf.write(f"DONE  best_pearson={best_pearson:.5f}\n")
        print(f"  Log saved: {log_path}")
    print(f"Camera-GS done (best Pearson={best_pearson:.4f})")
    return best_phase
