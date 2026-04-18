# ──────────────────────────────────────────────────────────────────────────────
# File: utils.py               Project: DNN-SLM Phase Mask Generation
# Purpose: Visualisation helpers, metrics, and tensor utilities
# ──────────────────────────────────────────────────────────────────────────────
"""
Utility / visualisation helpers.

Contains:
  - crop_edges        : zero-out border pixels to avoid edge artefacts
  - calculate_psnr    : peak signal-to-noise ratio between two tensors
  - tensor_to_image   : robust tensor → 2-D numpy conversion
  - plot_dual_metrics : training-history dual-panel plot (loss + PSNR)
  - plot_lr_history   : learning-rate schedule plot
  - save_visual_dashboard : 6-panel diagnostic figure
  - visualize_comparison  : quick 3-panel target|recon|phase figure
"""
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server / headless training
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import os

def crop_edges(tensor, margin_percent):
    """Prevents training on edge artifacts."""
    if margin_percent <= 0: return tensor
    
    # Handle both 3D (B, H, W) and 4D (B, C, H, W)
    if tensor.ndim == 4:
        b, c, h, w = tensor.shape
        h_margin = int(h * margin_percent)
        w_margin = int(w * margin_percent)
        mask = torch.zeros_like(tensor)
        mask[:, :, h_margin:h-h_margin, w_margin:w-w_margin] = 1.0
    elif tensor.ndim == 3:
        b, h, w = tensor.shape
        h_margin = int(h * margin_percent)
        w_margin = int(w * margin_percent)
        mask = torch.zeros_like(tensor)
        mask[:, h_margin:h-h_margin, w_margin:w-w_margin] = 1.0
    else:
        return tensor

    return tensor * mask

def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculates PSNR while handling logical boundaries.
    Args:
        img1, img2: Tensors or Arrays of shape (N, C, H, W) or (N, H, W)
        data_range: The max possible value of the data (1.0 for floats)
    """
    # 1. Ensure MSE is not Zero to avoid Divide By Zero (Infinity)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0

    # 2. Calculate PSNR
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    
    # 3. Logical Boundary Check (Optional)
    psnr = torch.clamp(psnr, min=0, max=100)
    
    return psnr.item()

def calculate_blurred_psnr(img1, img2, sigma=10.0, data_range=1.0):
    """PSNR on Gaussian-blurred images — separates speckle from shape quality."""
    import torch.nn.functional as F
    import math
    ksize = max(3, int(6 * sigma) | 1)
    coords = torch.arange(ksize, dtype=img1.dtype, device=img1.device) - ksize // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g = g / g.sum()
    kh = g.reshape(1, 1, ksize, 1)
    kw = g.reshape(1, 1, 1, ksize)
    pad_h, pad_w = ksize // 2, ksize // 2
    # Ensure 4D
    if img1.ndim == 3: img1 = img1.unsqueeze(1)
    if img2.ndim == 3: img2 = img2.unsqueeze(1)
    b1 = F.conv2d(F.conv2d(img1, kw, padding=(0, pad_w)), kh, padding=(pad_h, 0))
    b2 = F.conv2d(F.conv2d(img2, kw, padding=(0, pad_w)), kh, padding=(pad_h, 0))
    return calculate_psnr(b1, b2, data_range)


def tensor_to_image(tensor):
    """
    Robustly converts a Torch tensor (2D, 3D, or 4D) to a 2D Numpy array for plotting.
    Handles: [H, W], [1, H, W], [C, H, W], [1, C, H, W], [B, C, H, W]
    """
    # 1. Detach and move to CPU
    t = tensor.detach().cpu()
    
    # 2. Handle Dimensions
    if t.ndim == 4:
        # Assumes [Batch, Channel, Height, Width] -> Take first sample, first channel
        return t[0, 0, :, :].numpy()
    elif t.ndim == 3:
        # Assumes [Batch/Channel, Height, Width] -> Take first index
        return t[0, :, :].numpy()
    elif t.ndim == 2:
        # Assumes [Height, Width]
        return t.numpy()
    else:
        raise ValueError(f"Unsupported tensor shape for visualization: {t.shape}")

def _moving_average(values, window):
    if not values:
        return np.array([])
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode='valid')


def plot_dual_metrics(train_loss, train_psnr, val_steps, train_eval_loss, train_eval_psnr,
                      val_loss, val_psnr, save_path, stage_boundaries=None,
                      val_blur_psnr=None):
    """
    Saves the training history graphs (Loss & PSNR).
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    train_window = max(25, len(train_loss) // 200) if train_loss else 25
    smooth_loss = _moving_average(train_loss, train_window)
    smooth_psnr = _moving_average(train_psnr, train_window)
    smooth_steps = np.arange(len(smooth_loss)) + train_window - 1 if len(smooth_loss) else np.array([])
    
    # --- SUBPLOT 1: LOSS ---
    if len(smooth_loss):
        ax1.plot(smooth_steps, smooth_loss, color='mediumblue', label='Train Loss', alpha=0.85, linewidth=1.5)
    if val_steps:
        if train_eval_loss:
            ax1.plot(val_steps, train_eval_loss, color='darkviolet', linewidth=1.5, label='Train-Eval Loss')
        ax1.plot(val_steps, val_loss, color='darkgreen', linewidth=1.8, label='Validation Loss')

    ax1.set_ylabel("Combined Loss [-]")
    ax1.set_title("Training vs. Validation Performance")
    
    # --- SUBPLOT 2: PSNR ---
    if len(smooth_psnr):
        ax2.plot(smooth_steps, smooth_psnr, color='mediumblue', label='Train PSNR', alpha=0.85, linewidth=1.5)
    if val_steps:
        if train_eval_psnr:
            ax2.plot(val_steps, train_eval_psnr, color='darkviolet', linewidth=1.5, label='Train-Eval PSNR')
        ax2.plot(val_steps, val_psnr, color='darkgreen', linewidth=1.8, label='Validation PSNR')
        if val_blur_psnr:
            ax2.plot(val_steps, val_blur_psnr, color='darkorange', linewidth=1.5,
                     linestyle='--', label='Blur PSNR (shape quality)')
    
    ax2.set_ylabel("PSNR [dB]")
    ax2.set_xlabel("Iteration [-]")

    # --- STAGE BOUNDARIES ---
    max_x = max(len(train_loss), val_steps[-1] if val_steps else 0)
    if stage_boundaries is not None and len(stage_boundaries) > 0:
        max_x = max(max_x, max(stage_boundaries))
    
    xlims = (0, max_x * 1.02)
    ax1.set_xlim(xlims)
    ax2.set_xlim(xlims)

    if stage_boundaries is not None:
        colors = ['red', 'orange', 'purple', 'brown', 'magenta', 'olive', 'cyan', 'navy', 'teal']
        for i, boundary in enumerate(stage_boundaries):
            if boundary <= xlims[1]:
                    ax1.axvline(x=boundary, color=colors[i % len(colors)], 
                                linestyle='--', alpha=0.5)
                    ax2.axvline(x=boundary, color=colors[i % len(colors)], 
                                linestyle='--', alpha=0.5, label=f'End Stage {i+1}')
    
    # Place legends outside the plot area (right side, framed)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=True, edgecolor='lightgray')
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=True, edgecolor='lightgray')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig) 
    plt.close('all')
    
def plot_lr_history(lr_history, val_steps, save_path, stage_boundaries=None):
    """
    Plots the Learning Rate schedule over validation steps.
    """
    if not lr_history:
        return
        
    fig, ax = plt.subplots(figsize=(8, 4))
    # Plot data
    ax.plot(val_steps, lr_history, label='Learning Rate', color='purple', linewidth=2)
    
    # Stage boundaries
    if stage_boundaries is not None:
        colors = ['red', 'orange', 'purple', 'brown', 'magenta', 'olive', 'cyan', 'navy', 'teal']
        for i, boundary in enumerate(stage_boundaries):
            ax.axvline(x=boundary, color=colors[i % len(colors)],
                       linestyle='--', alpha=0.5, label=f'End Stage {i+1}')

    # Formatting
    ax.set_yscale('log')
    ax.set_xlabel('Iteration [-]')
    ax.set_ylabel('Learning Rate [-]')
    ax.set_title('Adaptive Learning Rate Schedule')
    ax.grid(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=True, edgecolor='lightgray')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def save_visual_dashboard(target, recon, phase, step, save_path):
    """
    Saves the current visual state of the network.
    Includes: Target, Recon, Phase, Zoom (for speckle), Histogram, and Profile.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- CRITICAL FIX: Robust Tensor Conversion ---
    # Use helper function instead of hardcoded indices [0,0]
    t_img = tensor_to_image(target)
    r_img = tensor_to_image(recon)
    p_img = tensor_to_image(phase)
    
    # Setup Figure (2 Rows)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 4, figure=fig)
    
    # --- ROW 1: IMAGES ---
    ax_t = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[0, 2])
    ax_z = fig.add_subplot(gs[0, 3])

    # Target
    ax_t.imshow(t_img, cmap='gray', vmin=0, vmax=1)
    ax_t.set_title("Target Intensity")
    ax_t.axis('off')
    
    # Reconstruction
    ax_r.imshow(r_img, cmap='gray') # Autoscale to see details
    ax_r.set_title(f"Reconstruction\n(Max: {r_img.max():.2f})")
    ax_r.axis('off')
    
    # Phase (Cyclic Colormap)
    im_p = ax_p.imshow(p_img, cmap='twilight', vmin=0, vmax=2*np.pi)
    ax_p.set_title(r"Phase Mask (0-2$\pi$)")
    ax_p.axis('off')
    plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)
    
    # Zoom (Center 100x100)
    h, w = r_img.shape
    cy, cx = h//2, w//2
    y1, y2 = max(0, cy-50), min(h, cy+50)
    x1, x2 = max(0, cx-50), min(w, cx+50)
    zoom_slice = r_img[y1:y2, x1:x2]
    
    ax_z.imshow(zoom_slice, cmap='gray')
    ax_z.set_title("Speckle Zoom (Center)")
    ax_z.axis('off')

    # --- ROW 2: DIAGNOSTICS ---
    ax_hist = fig.add_subplot(gs[1, :2])
    ax_prof = fig.add_subplot(gs[1, 2:])
    
    # Histogram
    ax_hist.hist(p_img.flatten(), bins=100, range=(0, 2*np.pi), color='purple', alpha=0.7)
    ax_hist.set_title("Phase Value Distribution")
    ax_hist.set_xlabel("Phase [rad]")
    ax_hist.set_ylabel("Count [-]")
    ax_hist.set_xlim(0, 2*np.pi)
    
    # Intensity Profile (Center Row)
    ax_prof.plot(t_img[cy, :], label='Target', color='black', alpha=0.5, linewidth=1)
    ax_prof.plot(r_img[cy, :], label='Recon', color='green', alpha=0.8, linewidth=1)
    ax_prof.set_title("Cross-Section (Center Row)")
    ax_prof.set_xlabel("Pixel X [-]")
    ax_prof.set_ylabel("Intensity [a.u.]")
    ax_prof.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_comparison(target, reconstruction, phase_mask, save_path):
    """Saves a visual comparison: Target | Reconstruction | Phase."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # --- CRITICAL FIX: Robust Tensor Conversion ---
    t = tensor_to_image(target)
    r = tensor_to_image(reconstruction)
    p = tensor_to_image(phase_mask)
    
    # Normalize phase for display (0 to 1) for Viridis colormap
    p_display = (p % (2 * np.pi)) / (2 * np.pi)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Target
    axs[0].imshow(t, cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Target Intensity")
    axs[0].axis('off')
    
    # Reconstruction
    axs[1].imshow(r, cmap='gray') 
    axs[1].set_title("Reconstructed Intensity")
    axs[1].axis('off')
    
    # Phase
    axs[2].imshow(p_display, cmap='gray', vmin=0, vmax=1)
    axs[2].set_title("Phase Mask (SLM)")
    axs[2].axis('off')
    
    plt.tight_layout()    
    plt.savefig(save_path, dpi=250)
    plt.close()


def save_blur_comparison(target, recon, save_path, sigmas=(3, 8)):
    """Show target vs recon at multiple Gaussian blur levels.
    This reveals the actual pattern the loss function sees (speckle removed)."""
    import torch.nn.functional as Futil
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    t4 = target if target.ndim == 4 else target.unsqueeze(1)
    r4 = recon if recon.ndim == 4 else recon.unsqueeze(1)

    n_sig = len(sigmas)
    fig, axes = plt.subplots(2, n_sig + 1, figsize=(4 * (n_sig + 1), 8))

    # Column 0: raw
    t_np = tensor_to_image(t4)
    r_np = tensor_to_image(r4)
    vmax = max(t_np.max(), 0.01)
    axes[0, 0].imshow(t_np, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 0].set_title('Target (raw)')
    axes[1, 0].imshow(r_np, cmap='hot', vmin=0, vmax=vmax)
    axes[1, 0].set_title('Recon (raw)')

    for col, sigma in enumerate(sigmas, 1):
        ks = int(6 * sigma + 1) | 1
        coords = torch.arange(ks, device=t4.device, dtype=t4.dtype) - ks // 2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        g = g / g.sum()
        kern = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        pad = ks // 2
        tb = Futil.conv2d(t4, kern, padding=pad)
        rb = Futil.conv2d(r4, kern, padding=pad)
        tb_np = tensor_to_image(tb)
        rb_np = tensor_to_image(rb)
        vmax_b = max(tb_np.max(), 0.01)
        axes[0, col].imshow(tb_np, cmap='hot', vmin=0, vmax=vmax_b)
        axes[0, col].set_title(f'Target (σ={sigma})')
        axes[1, col].imshow(rb_np, cmap='hot', vmin=0, vmax=vmax_b)
        axes[1, col].set_title(f'Recon (σ={sigma})')

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    fig.suptitle('Blur comparison (top: target, bottom: reconstruction)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()