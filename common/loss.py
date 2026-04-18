# ──────────────────────────────────────────────────────────────────────────────
# File: loss.py                Project: DNN-SLM Phase Mask Generation
# Purpose: Composite loss (supervised + physics + blur-Pearson + TV)
# ──────────────────────────────────────────────────────────────────────────────
"""
Composite loss function for holographic phase-mask training.

Strategy: coherent speckle prevents pixel-level matching, so the main
gradient signal comes from **Pearson correlation on Gaussian-blurred** images
at multiple scales.  Blurring averages out speckle and reveals the underlying
envelope pattern.

Combines:
  1. Supervised cosine-distance loss  (pred phase  vs.  GS teacher phase)
  2. Multi-scale blur-Pearson loss  (primary shape signal) with intra-stage annealing
  3. Cosine total-variation (TV) regularisation  (phase smoothness)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def local_phase_variance_loss(phase, patch_size=32):
    """Penalise local flat phase regions using patch-based circular variance.
    Unlike global variance, this detects and penalises the FLATTEST patch,
    catching localized flat regions that global metrics miss."""
    if phase.ndim == 3:
        phase = phase.unsqueeze(1)
    B, _, H, W = phase.shape
    # Pad to make divisible by patch_size
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        phase = F.pad(phase, (0, pad_w, 0, pad_h), mode='reflect')
    # Unfold into patches
    patches = phase.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches: (B, 1, nH, nW, patch_size, patch_size)
    sin_p = torch.sin(patches)
    cos_p = torch.cos(patches)
    # Circular variance per patch: 1 - |mean(exp(i*phi))|
    R = torch.sqrt(sin_p.mean((-2, -1))**2 + cos_p.mean((-2, -1))**2 + 1e-8)
    circ_var = 1.0 - R  # (B, 1, nH, nW) — 0=flat, 1=spread
    # Penalise minimum variance across patches (i.e. the flattest patch)
    min_var = circ_var.reshape(B, -1).min(dim=1).values  # (B,)
    return (1.0 - min_var).mean()  # high when any patch is flat

class HoloLoss(nn.Module):
    # sigmas for Gaussian blur: fine → coarse (wider to average out speckle at high res)
    BLUR_SIGMAS = [0.5, 2.0, 5.0]
    # relative weight of each blur level (coarser = stronger signal early on)
    BLUR_WEIGHTS = [0.3, 0.4, 0.3]

    def __init__(self, tv_weight=0.005, pearson_weight=1.0,
                 supervised_physics_weight=0.2,
                 bg_suppression_weight=1.0):
        super().__init__()
        self.tv_weight = tv_weight
        self.pearson_weight = pearson_weight
        self.supervised_physics_weight = supervised_physics_weight
        self.bg_suppression_weight = bg_suppression_weight
        self.phase_variance_weight = 0.15
        # Bounded TV: only penalise TV above this floor (prevents push toward flat)
        self.tv_floor = 0.3
        # Boosted supervised weight during teacher warmup
        self.supervised_boost = 1.0  # set by trainer (e.g. 3.0 during warmup)
        self.base_blur_sigmas = [float(s) for s in self.BLUR_SIGMAS]
        self.blur_weights = [float(w) for w in self.BLUR_WEIGHTS]
        self.stage_resolution = None
        self.blur_scale = 1.0
        # Blur annealing: multiplier on sigma (starts high, anneals to 1.0)
        self.blur_anneal_multiplier = 1.3
        # Cache blur kernels (built on first forward)
        self._blur_cache = {}

    def set_stage_resolution(self, resolution, base_height=256):
        """Update blur scale so loss preserves the same physical smoothing across stages.
        Uses log2 scaling — much gentler than sqrt, prevents loss-landscape flattening."""
        h, _ = resolution
        self.stage_resolution = resolution
        ratio = float(h) / float(base_height)
        self.blur_scale = min(1.5, 1.0 + 0.3 * math.log2(max(ratio, 1.0)))

    def set_blur_anneal(self, progress):
        """Set blur annealing multiplier based on stage progress [0..1].
        Starts at 1.05× and anneals to 1×."""
        self.blur_anneal_multiplier = 1.0 + 0.3 * max(0.0, 1.0 - progress)

    def _get_blur_kernels_1d(self, sigma, device, dtype):
        """Return (kernel_h, kernel_w) for separable 1D Gaussian blur."""
        key = (sigma, device, dtype)
        if key not in self._blur_cache:
            ks = int(6 * sigma + 1) | 1
            coords = torch.arange(ks, device=device, dtype=dtype) - ks // 2
            g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            g = g / g.sum()
            # Horizontal: [1, 1, 1, ks]
            kw = g.reshape(1, 1, 1, ks)
            # Vertical:   [1, 1, ks, 1]
            kh = g.reshape(1, 1, ks, 1)
            self._blur_cache[key] = (kh, kw, ks // 2)
        return self._blur_cache[key]

    def _blur(self, x, sigma):
        """Fast separable Gaussian blur on [B,1,H,W] tensor."""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        kh, kw, pad = self._get_blur_kernels_1d(sigma, x.device, x.dtype)
        # Horizontal pass then vertical pass (separable = ~30x faster for large σ)
        x = F.conv2d(x, kw, padding=(0, pad))
        x = F.conv2d(x, kh, padding=(pad, 0))
        return x

    def forward(self, pred_phase, gt_phase, recon_amp, target_amp):
        total_loss = 0.0
        loss_dict = {}

        # 1. SUPERVISED LOSS (Student vs Teacher) — boosted during warmup
        #    Wrapped MSE: stronger (quadratic) gradients for large phase errors
        #    vs cosine which saturates at 1.  Normalized so max ≈ 1.
        if gt_phase is not None:
            if pred_phase.shape != gt_phase.shape:
                if pred_phase.ndim == 3: pred_phase = pred_phase.unsqueeze(1)
                if gt_phase.ndim == 3: gt_phase = gt_phase.unsqueeze(1)
            diff = pred_phase - gt_phase
            diff = diff - 2 * math.pi * torch.round(diff / (2 * math.pi))
            sup_loss = torch.mean(diff ** 2) / (math.pi ** 2)
            loss_dict['sup'] = sup_loss
            total_loss += self.supervised_boost * sup_loss

        # 2. PHYSICS LOSS (Reconstruction vs Target)
        if recon_amp is not None and target_amp is not None:
            # Energy-normalize reconstruction to target scale
            recon_sum = recon_amp.sum(dim=(-1, -2), keepdim=True) + 1e-8
            target_sum = target_amp.sum(dim=(-1, -2), keepdim=True) + 1e-8
            scale = (target_sum / recon_sum).clamp(max=100.0).detach()
            recon_norm = recon_amp * scale

            # Ensure 4D
            r4 = recon_norm.unsqueeze(1) if recon_norm.ndim == 3 else recon_norm
            t4 = target_amp.unsqueeze(1) if target_amp.ndim == 3 else target_amp

            weight = self.supervised_physics_weight if gt_phase is not None else 1.0

            # Multi-scale Gaussian-blur Pearson (with intra-stage annealing)
            # ROI-weighted: compute Pearson on full image AND on ROI where target > 0.05
            agg_pearson = 0.0
            anneal = getattr(self, 'blur_anneal_multiplier', 1.0)
            scaled_sigmas = [max(1.0, s * self.blur_scale * anneal) for s in self.base_blur_sigmas]

            # Build ROI weight mask: higher weight where target has content
            with torch.no_grad():
                roi_mask = (t4 > 0.05 * t4.max()).float()
                # Dilate ROI slightly so edges are included (blur the mask)
                roi_sigma = max(3.0, 5.0 * self.blur_scale)
                roi_mask = self._blur(roi_mask, roi_sigma)
                roi_mask = roi_mask / (roi_mask.max() + 1e-8)
                # Blend: background weight 0.3, foreground weight 1.0
                roi_weight = 0.3 + 0.7 * roi_mask

            for sigma, bw in zip(scaled_sigmas, self.blur_weights):
                r_blur = self._blur(r4, sigma)
                t_blur = self._blur(t4, sigma)
                # Standard Pearson
                p_loss = self.pearson_loss(r_blur, t_blur)
                # ROI-weighted Pearson
                p_roi = self.pearson_loss(r_blur * roi_weight, t_blur * roi_weight)
                # Blend: 50% global + 50% ROI-weighted
                agg_pearson = agg_pearson + bw * (0.5 * p_loss + 0.5 * p_roi)

            loss_dict['pearson'] = agg_pearson
            total_loss += weight * self.pearson_weight * agg_pearson

            # BACKGROUND SUPPRESSION — penalise energy outside the target region
            if self.bg_suppression_weight > 0:
                with torch.no_grad():
                    bg_mask = (t4 < 0.05 * t4.max()).float()
                bg_energy = (r4 * bg_mask).mean()
                loss_dict['bg'] = bg_energy
                total_loss += weight * self.bg_suppression_weight * bg_energy

        # 3. BOUNDED TV REGULARIZATION (only penalise above floor)
        if self.tv_weight > 0:
            tv = self.cosine_tv_loss(pred_phase)
            bounded_tv = torch.clamp(tv - self.tv_floor, min=0.0)
            total_loss += self.tv_weight * bounded_tv
            loss_dict['tv'] = tv           # log raw TV for diagnostics
            loss_dict['tv_b'] = bounded_tv  # bounded (what's actually penalised)

        # 4. PHASE VARIANCE (anti-flat-phase)
        if self.phase_variance_weight > 0:
            pv = local_phase_variance_loss(pred_phase)
            total_loss += self.phase_variance_weight * pv
            loss_dict['pv'] = pv

        return total_loss, loss_dict

    @staticmethod
    def pearson_loss(pred, target):
        """1 - Pearson correlation (batch-averaged). Scale-invariant shape loss."""
        b = pred.size(0)
        if target.numel() == pred.numel():
            target = target.reshape(pred.shape)
        elif target.shape[0] != b:
            target = target.expand_as(pred)

        p_flat = pred.reshape(b, -1)
        t_flat = target.reshape(b, -1)

        p_mean = p_flat.mean(dim=1, keepdim=True)
        t_mean = t_flat.mean(dim=1, keepdim=True)

        correlation = F.cosine_similarity(
            p_flat - p_mean, t_flat - t_mean, dim=1, eps=1e-8)
        return 1.0 - torch.mean(correlation)

    @staticmethod
    def cosine_tv_loss(phase):
        """Smoothness that respects 2π wrapping."""
        if phase.ndim == 3:
            phase = phase.unsqueeze(1)
        d_h = phase[:, :, 1:, :] - phase[:, :, :-1, :]
        d_w = phase[:, :, :, 1:] - phase[:, :, :, :-1]
        return torch.mean(1.0 - torch.cos(d_h)) + torch.mean(1.0 - torch.cos(d_w))
