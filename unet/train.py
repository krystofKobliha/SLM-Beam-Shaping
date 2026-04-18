# ──────────────────────────────────────────────────────────────────────────────
# File: train.py               Project: DNN-SLM Phase Mask Generation
# Purpose: Progressive synthetic training loop (physics-only, end-to-end)
# ──────────────────────────────────────────────────────────────────────────────
"""
Progressive synthetic training loop.

Pipeline per iteration:
  1. SyntheticGenerator produces a random target I_target
  2. HoloNet predicts a phase mask φ
  3. HolographyPhysics forward-propagates φ → I_sim
  4. HoloLoss computes the combined loss (blur-Pearson + blur-MSE + TV)
  5. Gradients accumulated over accum_steps, then Adam update

Training is divided into resolution stages (config.SCHEDULE).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import torch
import torch.optim as optim
import numpy as np
import gc
import copy
import time
import math
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime

from config import PHYSICS_CONFIG, SCHEDULE, SLM_PITCH, OUTPUT_DIR, TRAIN_CONFIG, LOSS_CONFIG, RESUME_CONFIG
from dataset import SyntheticGenerator
from model import HoloNet
from utils import crop_edges, calculate_psnr, calculate_blurred_psnr, save_visual_dashboard, plot_dual_metrics, plot_lr_history, visualize_comparison, save_blur_comparison
from loss import HoloLoss
from fft_physics import FFTPhysics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'iterative'))
from gerchberg_saxton import fft_gerchberg_saxton

# --- LOGGING SETUP ---
log_file = open(os.path.join(OUTPUT_DIR, "training_log.txt"), "a", encoding="utf-8")
log_file.write("\n" + "=" * 80 + "\n")
log_file.write(f"NEW RUN START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write("=" * 80 + "\n")
log_file.flush()

def log_print(message):
    print(message)
    sys.stdout.flush()
    log_file.write(message + "\n")
    log_file.flush()


@torch.no_grad()
def update_ema(ema_model, model, decay, step=None, warmup=200):
    """Update EMA model parameters with warmup.
    During warmup, effective decay ramps from 0 to target decay."""
    if step is not None and step < warmup:
        decay = min(decay, step / warmup * decay)
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

def generate_refined_labels(target_amp, waist_ratio=0.45, init_phase=None, steps=80):
    """
    FFT-GS Teacher: produces a clean teacher phase via centred-FFT Gerchberg-Saxton.

    Uses fft2/ifft2 (exact inverses) so no noise accumulates — produces
    near-perfect phases (r≈0.999) at any resolution without degradation.
    Much better teacher than the old PaddedPhysics GS which degraded at
    higher resolutions due to inexact forward/backward transforms.

    Handles batched targets: processes each sample independently.
    """
    device = target_amp.device
    B = target_amp.shape[0]
    phases = []
    for b in range(B):
        sample = target_amp[b:b+1]                    # [1, 1, H, W]
        phase_b = fft_gerchberg_saxton(
            sample, waist_ratio=waist_ratio,
            steps=steps, global_iters=min(50, steps // 6),
            silent=True,
        )
        if init_phase is not None:
            # Blend: use init_phase structure but refine with FFT-GS result.
            # This leverages the UNet's warm-start while still benefiting
            # from the clean FFT-GS phase.
            pass  # FFT-GS already optimal; init_phase not needed
        phases.append(phase_b)
    return torch.cat(phases, dim=0).detach()


def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f"Starting Training on {device}")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- 1. SETUP MODEL & OPTIMIZER ---
    model = HoloNet().to(device)
    ema_model = copy.deepcopy(model)
    ema_decay = TRAIN_CONFIG.get('ema_decay', 0.999)

    # --- 1b. RESUME FROM CHECKPOINT (optional) ---
    start_stage = 1  # 1-indexed
    resume_path = RESUME_CONFIG.get('resume_from')
    if resume_path and os.path.isfile(resume_path):
        start_stage = RESUME_CONFIG.get('start_stage', 1)
        log_print(f"Resuming from checkpoint: {resume_path}")
        log_print(f"  -> Skipping to Stage {start_stage} (stages 1-{start_stage-1} skipped)")
        state = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        ema_model.load_state_dict(state)
        del state
    elif resume_path:
        log_print(f"WARNING: resume checkpoint not found: {resume_path} — training from scratch")

    criterion = HoloLoss(
        tv_weight=LOSS_CONFIG['tv_weight'],
        pearson_weight=LOSS_CONFIG['pearson_weight'],
        supervised_physics_weight=LOSS_CONFIG['supervised_physics_weight'],
        bg_suppression_weight=LOSS_CONFIG.get('bg_suppression_weight', 1.0),
    ).to(device)
    # Override phase_variance_weight from config if provided
    if 'phase_variance_weight' in LOSS_CONFIG:
        criterion.phase_variance_weight = LOSS_CONFIG['phase_variance_weight']
    
    # --- 3. HISTORY TRACKING ---
    train_loss_hist, train_psnr_hist = [], []
    train_eval_loss_hist, train_eval_psnr_hist = [], []
    val_steps, val_loss_hist, val_psnr_hist = [], [], []
    val_blur_psnr_hist = []
    lr_history = []  # <--- NEW: To track LR over time
    stage_end_steps = []
    global_step = 0
    
    # --- 4. STAGE LOOP ---
    for stage_idx, cfg in enumerate(SCHEDULE):
        # Skip stages before start_stage (1-indexed)
        if (stage_idx + 1) < start_stage:
            log_print(f"\n=== STAGE {stage_idx+1}: SKIPPED (resumed from checkpoint) ===")
            continue

        full_res = cfg['res']
        h, w = full_res
        
        # Physics Info
        current_sensor_width = w * SLM_PITCH
        current_sensor_height = h * SLM_PITCH
        beam_waist_m = min(current_sensor_width, current_sensor_height) * PHYSICS_CONFIG['waist_ratio']
        beam_waist_px = beam_waist_m / SLM_PITCH
        
        log_print(f"\n=== STAGE {stage_idx+1}: {h}x{w} ===")
        log_print(f" > Fixed Pitch: {SLM_PITCH*1e6:.2f} um")
        log_print(f" > Simulated Sensor: {current_sensor_width*1e3:.1f} x {current_sensor_height*1e3:.1f} mm")
        log_print(f" > Beam waist: {beam_waist_m*1e3:.2f} mm ({beam_waist_px:.0f} px)")
        
        # Re-init Physics for new resolution (FFTPhysics matches the GS teacher)
        physics = FFTPhysics(PHYSICS_CONFIG, (h, w), device)
        generator = SyntheticGenerator(device, (h, w), 
                                       pixel_pitch=SLM_PITCH,
                                       wavelength=PHYSICS_CONFIG['wavelength'],
                                       focal_length=PHYSICS_CONFIG['f3'],
                                       waist_ratio=PHYSICS_CONFIG['waist_ratio'])

        # Fixed Validation Batch (32 samples for stable metrics)
        with torch.no_grad():
            fixed_val_batch = generator.sample_batch(32) 
            fixed_val_batch = crop_edges(fixed_val_batch, cfg['margin'])
            fixed_train_eval_batch = generator.sample_batch(32)
            fixed_train_eval_batch = crop_edges(fixed_train_eval_batch, cfg['margin'])

        # Per-stage setup
        criterion.tv_weight = cfg['tv']
        criterion.set_stage_resolution((h, w))
        model.train()
        best_val_loss = float('inf')
        best_val_psnr = float('-inf')
        best_ema_state = copy.deepcopy(ema_model.state_dict())
        val_psnr_stage = []
        accum_steps = cfg.get('accum', TRAIN_CONFIG.get('accum_steps', 1))
        max_iters = cfg['iters']
        min_iters = cfg.get('min_iters', int(max_iters * 0.6))
        warmup_iters = int(max_iters * TRAIN_CONFIG.get('warmup_fraction', 0.05))
        teacher_warmup_fraction = cfg.get('teacher_warmup_fraction', TRAIN_CONFIG.get('teacher_warmup_fraction', 0.0))
        teacher_warmup_iters = int(max_iters * teacher_warmup_fraction)
        plateau_window = int(TRAIN_CONFIG.get('plateau_window_iters', 3000))
        plateau_delta = float(TRAIN_CONFIG.get('plateau_delta_psnr', 0.10))
        base_lr = cfg['lr']

        phase_noise_max = float(TRAIN_CONFIG.get('phase_noise_std', 0.0))

        # Per-stage optimizer: simple Adam + cosine annealing with warmup
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        update_steps_done = 0

        # Cosine annealing with linear warmup + warm restarts
        min_lr = TRAIN_CONFIG.get('scheduler_min_lr', 1e-6)
        min_factor = min_lr / base_lr if base_lr > 0 else 0
        total_stage_updates = max(1, math.ceil(max_iters / accum_steps))
        warmup_updates = max(1, math.ceil(warmup_iters / accum_steps))
        restart_period_iters = TRAIN_CONFIG.get('restart_period_iters', 0)  # 0 = no restarts
        restart_updates = max(1, math.ceil(restart_period_iters / accum_steps)) if restart_period_iters > 0 else 0
        restart_decay = TRAIN_CONFIG.get('restart_decay', 0.85)  # peak LR decays each restart

        def lr_lambda(step, _wu=warmup_updates, _tot=total_stage_updates,
                       _mf=min_factor, _rp=restart_updates, _rd=restart_decay):
            if step < _wu:
                return max(_mf, (step + 1) / _wu)
            post_warmup = step - _wu
            if _rp > 0:
                # Cosine warm restarts: cycle within each restart period
                cycle = post_warmup // _rp
                progress = (post_warmup % _rp) / _rp
                peak = _rd ** cycle  # decaying peak each restart
                return _mf + 0.5 * (peak - _mf) * (1 + math.cos(math.pi * progress))
            else:
                # Single cosine decay (original behavior)
                progress = post_warmup / max(1, _tot - _wu)
                return _mf + 0.5 * (1.0 - _mf) * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        log_print(
            f" > Accum steps: {accum_steps} (eff. batch={cfg['batch']*accum_steps}) | "
            f"Warmup: {warmup_iters} iters | Teacher warmup: {teacher_warmup_iters} iters | "
            f"Phase noise: {phase_noise_max:.3f} rad"
        )
        
        # --- 5. ITERATION LOOP ---
        iter_start = time.time()
        i = 0
        while i < max_iters:
            # Zero gradients at accumulation boundaries
            if i % accum_steps == 0:
                optimizer.zero_grad()
            
            # A. Data Generation
            target = generator.sample_batch(cfg['batch'])
            target = crop_edges(target, cfg['margin'])
            
            # B. Student Prediction (Forward, float32)
            pred_phase = model(target)

            # Phase diversity: inject annealed noise to escape local minima
            if phase_noise_max > 0:
                noise_std = phase_noise_max * (1.0 - i / max_iters)
                if noise_std > 1e-6:
                    pred_phase = pred_phase + torch.randn_like(pred_phase) * noise_std

            # A: Update blur annealing (coarse→fine within each stage)
            stage_progress = i / max(1, max_iters)
            criterion.set_blur_anneal(stage_progress)

            # C. Optional FFT-GS teacher warm-start at beginning of each stage
            #    Smooth decay over 5000 iters after warmup (prevents abrupt gradient loss)
            gt_phase = None
            teacher_decay_iters = 5000
            if teacher_warmup_iters > 0 and i < teacher_warmup_iters:
                criterion.supervised_boost = 3.0  # boost teacher signal to escape flat phase
                with torch.no_grad():
                    gt_phase = generate_refined_labels(
                        target_amp=target,
                        waist_ratio=PHYSICS_CONFIG.get('waist_ratio', 0.45),
                        steps=cfg.get('opt_steps', 80),
                    )
            elif teacher_warmup_iters > 0 and i < teacher_warmup_iters + teacher_decay_iters:
                # Smooth linear decay: 3.0 → 1.0 over teacher_decay_iters
                t = (i - teacher_warmup_iters) / teacher_decay_iters
                criterion.supervised_boost = 3.0 * (1.0 - t) + 1.0 * t
                with torch.no_grad():
                    gt_phase = generate_refined_labels(
                        target_amp=target,
                        waist_ratio=PHYSICS_CONFIG.get('waist_ratio', 0.45),
                        steps=cfg.get('opt_steps', 80),
                    )
            else:
                criterion.supervised_boost = 1.0

            # D. Physics loss (optionally mixed with teacher supervision)
            recon_intensity = physics(pred_phase.float())
            total_loss, _ = criterion(pred_phase, gt_phase, recon_intensity, target)
            
            # E. Skip NaN/Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                optimizer.zero_grad()  # discard any accumulated gradients
                global_step += 1
                i += 1
                continue
            
            # F. Backprop + Gradient accumulation support
            scaled_loss = total_loss / accum_steps
            scaled_loss.backward()
            
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAIN_CONFIG['grad_clip'])
                optimizer.step()
                update_ema(ema_model, model, ema_decay, step=i)
                scheduler.step()
                update_steps_done += 1
            
            # Metrics
            if not torch.isnan(total_loss):
                train_loss_hist.append(total_loss.item())
                
            with torch.no_grad():
                s_t = target.sum(dim=(-1,-2), keepdim=True)
                s_r = recon_intensity.sum(dim=(-1,-2), keepdim=True)
                recon_scaled = recon_intensity * (s_t / (s_r + 1e-8))
                train_psnr_hist.append(calculate_psnr(target, recon_scaled))
            
            # Brief progress indicator every 50 iterations
            if i > 0 and i % 50 == 0 and i % TRAIN_CONFIG['val_interval'] != 0:
                elapsed = time.time() - iter_start
                it_per_sec = i / elapsed
                log_print(f"  ... iter {i}/{max_iters} ({it_per_sec:.1f} it/s, loss={total_loss.item():.3f})")

            # --- 6. VALIDATION & LOGGING ---
            if i % TRAIN_CONFIG['val_interval'] == 0 or i == max_iters - 1:
                # Validate on EMA model (smooth weights → stable metrics)
                # Chunked forward to avoid OOM at high resolutions
                # EMA in train mode: BN uses batch stats (correct for EMA weights)
                with torch.no_grad():
                    val_chunk = cfg['batch']  # same chunk size as training batch
                    all_v_phase, all_v_recon = [], []
                    all_t_phase, all_t_recon = [], []
                    for vc in range(0, len(fixed_val_batch), val_chunk):
                        vb = fixed_val_batch[vc:vc+val_chunk]
                        vp = ema_model(vb)
                        vr = physics(vp)
                        all_v_phase.append(vp)
                        all_v_recon.append(vr)

                    for tc in range(0, len(fixed_train_eval_batch), val_chunk):
                        tb = fixed_train_eval_batch[tc:tc+val_chunk]
                        tp = ema_model(tb)
                        tr = physics(tp)
                        all_t_phase.append(tp)
                        all_t_recon.append(tr)

                    v_phase = torch.cat(all_v_phase, dim=0)
                    v_recon = torch.cat(all_v_recon, dim=0)
                    t_phase = torch.cat(all_t_phase, dim=0)
                    t_recon = torch.cat(all_t_recon, dim=0)

                    v_loss, v_metrics = criterion(v_phase, None, v_recon, fixed_val_batch)
                    t_loss_cmp, _ = criterion(t_phase, None, t_recon, fixed_train_eval_batch)
                    
                    t_st = fixed_train_eval_batch.sum(dim=(-1,-2), keepdim=True)
                    t_sr = t_recon.sum(dim=(-1,-2), keepdim=True)
                    t_psnr_cmp = calculate_psnr(fixed_train_eval_batch, t_recon * (t_st / (t_sr + 1e-8)))
                    
                    v_st = fixed_val_batch.sum(dim=(-1,-2), keepdim=True)
                    v_sr = v_recon.sum(dim=(-1,-2), keepdim=True)
                    v_recon_scaled = v_recon * (v_st / (v_sr + 1e-8))
                    v_psnr = calculate_psnr(fixed_val_batch, v_recon_scaled)

                    # H: Blurred PSNR diagnostic (separates speckle from actual shape quality)
                    blur_sigma_diag = max(5.0, 10.0 * criterion.blur_scale)
                    v_blur_psnr = calculate_blurred_psnr(fixed_val_batch, v_recon_scaled, sigma=blur_sigma_diag)

                    val_steps.append(global_step)
                    train_eval_loss_hist.append(t_loss_cmp.item())
                    train_eval_psnr_hist.append(t_psnr_cmp)
                    val_loss_hist.append(v_loss.item())
                    val_psnr_hist.append(v_psnr)
                    val_blur_psnr_hist.append(v_blur_psnr)
                # model stays in train mode (EMA used for validation)

                # --- SAVE BEST EMA MODEL ---
                if v_loss.item() < best_val_loss:
                    best_val_loss = v_loss.item()
                    best_val_psnr = v_psnr
                    best_ema_state = copy.deepcopy(ema_model.state_dict())
                    torch.save(ema_model.state_dict(), os.path.join(OUTPUT_DIR, f"holonet_best_S{stage_idx+1}.pth"))
                    log_print(f"  [Best] Saved EMA (val_loss={v_loss.item():.4f})")
                
                # --- RECORD CURRENT LR ---
                current_lr = optimizer.param_groups[0]['lr']
                lr_history.append(current_lr) # Save for plotting

                # Logging
                l_pearson = v_metrics.get('pearson', 0.0)
                l_tv      = v_metrics.get('tv', 0.0)
                l_tv_b    = v_metrics.get('tv_b', 0.0)
                l_pv      = v_metrics.get('pv', 0.0)
                l_bg      = v_metrics.get('bg', 0.0)
                mem_used = torch.cuda.max_memory_allocated() / 1024**3 
                recent_train_psnr = np.mean(train_psnr_hist[-10:]) if train_psnr_hist else 0.0

                log_msg = (
                    f"[S{stage_idx+1} {i:05d}/{max_iters}] "
                    f"Loss TrEval:{t_loss_cmp.item():.4f} Val:{v_loss.item():.4f} "
                    f"(Prsn:{l_pearson:.3f} TV:{l_tv:.3f}/{l_tv_b:.3f} BG:{l_bg:.3f} PV:{l_pv:.3f}) | "
                    f"PSNR Tr:{recent_train_psnr:.1f} Val:{v_psnr:.1f} BlurVal:{v_blur_psnr:.1f} | "
                    f"LR: {current_lr:.1e} | Mem: {mem_used:.1f}GB"
                )
                log_print(log_msg)
                val_psnr_stage.append((i, v_psnr))

                # --- PLOTTING ---
                # 1. Metrics Plot
                plot_dual_metrics(
                    train_loss_hist, train_psnr_hist,
                    val_steps, train_eval_loss_hist, train_eval_psnr_hist,
                    val_loss_hist, val_psnr_hist,
                    os.path.join(OUTPUT_DIR, "training_metrics.png"),
                    stage_boundaries=stage_end_steps,
                    val_blur_psnr=val_blur_psnr_hist
                )
                
                # 2. Learning Rate Plot
                plot_lr_history(
                    lr_history, 
                    val_steps, 
                    os.path.join(OUTPUT_DIR, "lr_history.png"),
                    stage_boundaries=stage_end_steps
                )

                # 3. Visual Dashboard (uses EMA model for stable output)
                if i % TRAIN_CONFIG['vis_interval'] == 0:
                    with torch.no_grad():
                        vis_target = generator.sample_batch(1)
                        vis_target = crop_edges(vis_target, cfg['margin'])
                        
                        vis_phase = ema_model(vis_target)
                        vis_recon = physics(vis_phase)
                        
                        s_vt = vis_target.sum()
                        s_vr = vis_recon.sum()
                        vis_recon_scaled = vis_recon * (s_vt / (s_vr + 1e-8))

                        dash_path = os.path.join(OUTPUT_DIR, f"dash_S{stage_idx+1}_{i:05d}.png")
                        save_visual_dashboard(vis_target, vis_recon_scaled, vis_phase, global_step, dash_path)

                        comp_path = os.path.join(OUTPUT_DIR, f"comp_S{stage_idx+1}_{i:05d}.png")
                        visualize_comparison(vis_target, vis_recon_scaled, vis_phase, comp_path)

                        # Blurred comparison (what the loss actually sees)
                        blur_path = os.path.join(OUTPUT_DIR, f"blur_S{stage_idx+1}_{i:05d}.png")
                        save_blur_comparison(vis_target, vis_recon_scaled, blur_path)

                # Convergence-gated transition: require minimal PSNR gain over window.
                if i >= min_iters and len(val_psnr_stage) >= 2:
                    start_psnr = None
                    for old_i, old_psnr in val_psnr_stage:
                        if old_i <= i - plateau_window:
                            start_psnr = old_psnr
                        else:
                            break
                    if start_psnr is not None and (v_psnr - start_psnr) < plateau_delta:
                        log_print(
                            f"[S{stage_idx+1}] Converged by gate: "
                            f"PSNR gain {v_psnr - start_psnr:.3f} < {plateau_delta:.3f} "
                            f"over {plateau_window} iters at iter {i}."
                        )
                        break
            
            global_step += 1
            i += 1
            
        # Carry the best EMA state forward into the next stage.
        # (HF loss disabled)
        model.load_state_dict(best_ema_state)
        ema_model.load_state_dict(best_ema_state)

        stage_end_steps.append(global_step)
        log_print(
            f"[S{stage_idx+1}] Stage summary: best Val loss={best_val_loss:.4f}, "
            f"best Val PSNR={best_val_psnr:.2f}, end step={global_step}."
        )

        # Save both training and EMA models at end of stage
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"holonet_stage_{stage_idx+1}.pth"))
        torch.save(ema_model.state_dict(), os.path.join(OUTPUT_DIR, f"holonet_ema_stage_{stage_idx+1}.pth"))
        log_print(f"  [Stage {stage_idx+1}] Saved EMA checkpoint.")

    log_print("Training Complete.")
    log_file.close()

if __name__ == "__main__":
    run_training()
