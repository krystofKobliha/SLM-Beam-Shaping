# ──────────────────────────────────────────────────────────────────────────────
# File: gerchberg_saxton.py    Project: DNN-SLM Phase Mask Generation
# Purpose: FFT-based Gerchberg–Saxton phase retrieval (centred FT)
# ──────────────────────────────────────────────────────────────────────────────
"""
FFT-based Gerchberg–Saxton algorithm for computing SLM phase masks.

Models the 3-lens relay as a single Fourier transform (physically correct
for a 4f system).  Uses centred FTs (DC at array centre) so pixel positions
map directly to SLM/camera positions.

Provides:
    fft_gerchberg_saxton  — the only GS function used in the project.
      Called by gs_generate.py (CLI runner) and train.py (FFT-GS teacher).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import math
import torch


# ---------------------------------------------------------------------------
# Pure FFT-based GS  (matches the working Mathematica implementation)
# ---------------------------------------------------------------------------
def fft_gerchberg_saxton(target_amp, waist_ratio=0.45, steps=300,
                         global_iters=50, silent=False):
    """
    Pure FFT-based Gerchberg-Saxton matching the Mathematica algorithm.

    Uses *centered* Fourier transforms throughout: DC stays at the array
    centre in every domain.  This eliminates all fftshift/ifftshift at
    domain boundaries and ensures that pixel positions in the algorithm
    map directly to physical positions on the SLM and camera.

    The 3-lens relay is a 4f system whose transfer function is a scaled
    Fourier transform, so the FFT model is physically appropriate.

    Two-phase constraint (same as Mathematica):
      * Phase 1 (< global_iters): global amplitude constraint
      * Phase 2 (≥ global_iters): MRAF — signal region only

    Args:
        target_amp:   [1, 1, H, W] target *intensity* in [0, 1].
        waist_ratio:  Gaussian 1/e beam radius as a fraction of min(H, W)
                      (must match FFTPhysics so the phase is compatible).
        steps:        number of GS iterations.
        global_iters: iterations with global constraint before MRAF.
        silent:       suppress progress logging (for use as training teacher).

    Returns:
        Phase tensor [1, 1, H, W] in [0, 2π], centred layout (beam centre
        at array centre — directly usable by FFTPhysics and the SLM).
    """
    device = target_amp.device
    target_2d = target_amp.squeeze()          # [H, W]
    H, W = target_2d.shape

    target_amplitude = torch.sqrt(target_2d.clamp(min=0))
    signal_mask = (target_amplitude > 1e-4)

    # Gaussian source beam — centred layout (peak at array centre)
    waist_px = min(H, W) * waist_ratio
    gy = torch.arange(H, device=device, dtype=torch.float32) - H / 2
    gx = torch.arange(W, device=device, dtype=torch.float32) - W / 2
    yy, xx = torch.meshgrid(gy, gx, indexing='ij')
    r2 = xx ** 2 + yy ** 2
    source = torch.exp(-r2 / waist_px ** 2)

    # Circular aperture (must match FFTPhysics): sigmoid cutoff at 95% of
    # shorter grid half-extent to eliminate rectangular edge artifacts.
    half_min = min(H, W) / 2
    aperture_radius = 0.95 * half_min
    edge_width = 0.05 * half_min
    r = torch.sqrt(r2)
    aperture = torch.sigmoid((aperture_radius - r) / edge_width)
    source = source * aperture

    # Centred Fourier transforms (DC stays at array centre in both domains)
    def cft2(x):
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))

    def icft2(x):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))

    # Initialise: A = ICFT(target)  (Mathematica-style structured init)
    A = icft2(target_amplitude.to(torch.complex64))

    best_corr = -1.0
    best_A = A.clone()

    for u in range(steps):
        # 1. SLM-plane constraint: keep phase, replace amplitude with source
        b = source * torch.exp(1j * torch.angle(A))

        # 2. Forward propagation: centred FT  (SLM → sensor)
        c = cft2(b)
        c = c / (c.abs().max() + 1e-12)

        # 3. Sensor-plane amplitude constraint
        if u < global_iters:
            d = target_amplitude * torch.exp(1j * torch.angle(c))
        else:
            d = c.clone()
            d[signal_mask] = (target_amplitude[signal_mask]
                              * torch.exp(1j * torch.angle(c[signal_mask])))

        # 4. Backward propagation: centred IFT  (sensor → SLM) + normalise
        A = icft2(d)
        A = A / (A.abs().max() + 1e-12)

        # --- Track best & progress logging (every 50 iters) ---
        if (u + 1) % 50 == 0 or u == 0:
            with torch.no_grad():
                recon = c.abs() ** 2
                corr = torch.corrcoef(torch.stack([
                    recon.reshape(-1), target_2d.reshape(-1)
                ]))[0, 1].item()
            if not silent:
                print(f"  FFT-GS iter {u+1:4d}/{steps}  Pearson r = {corr:.4f}")
            if corr > best_corr:
                best_corr = corr
                best_A = A.clone()

    if not silent:
        print(f"  FFT-GS done — best Pearson r = {best_corr:.4f}")

    # Phase already in centred layout — no shift needed
    phase = torch.remainder(torch.angle(best_A), 2 * math.pi)
    return phase.unsqueeze(0).unsqueeze(0)
