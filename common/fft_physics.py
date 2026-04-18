# ──────────────────────────────────────────────────────────────────────────────
# File: fft_physics.py         Project: DNN-SLM Phase Mask Generation
# Purpose: Simple FFT-based physics model (single Fourier transform)
# ──────────────────────────────────────────────────────────────────────────────
"""
Minimal differentiable physics model that models the 3-lens relay as a single
Fourier transform — matching the Mathematica GS implementation.

Why this works:  A 4f relay (lens–propagate–lens–propagate) is mathematically
equivalent to a scaled 2-D Fourier transform.  The three-lens system
(f1→L1→f1+f2→L2→f2+f3→L3→f3) is two 4f relays in series.  The end-to-end
transfer function is a scaled Fourier transform with magnification f3/f1.

Why PaddedPhysics fails here:  Each angular-spectrum step pads with zeros,
propagates, and crops.  Forward/backward are NOT exact inverses → GS
accumulates noise.  The full model also produces ghost copies (3×3 grid)
that the FFT model avoids.

This module provides:
  FFTPhysics — drop-in replacement for HolographyPhysics with the same API
      .forward_field(phase_mask) → complex field at sensor
      .backward_field(sensor_field) → complex field at SLM
      .forward(phase_mask)  → intensity |U|² at sensor
"""
import math
import numpy as np
import torch
import torch.nn as nn


class FFTPhysics(nn.Module):
    """Centred-FFT physics model matching the GS algorithm.

    Uses centred Fourier transforms throughout: DC stays at the array
    centre in both SLM and sensor domains.  This ensures pixel positions
    in the model map directly to physical positions on the SLM/camera.
    """

    def __init__(self, config, resolution, device):
        super().__init__()
        self.config = config
        self.device = device
        self.H, self.W = resolution
        self.pitch = config['pixel_pitch']

        # Gaussian source beam — centred layout (peak at array centre)
        phys_H = self.H * self.pitch
        phys_W = self.W * self.pitch
        waist = min(phys_H, phys_W) * config.get('waist_ratio', 0.45)

        y, x = torch.meshgrid(
            torch.linspace(-self.H / 2, self.H / 2, self.H,
                           device=device) * self.pitch,
            torch.linspace(-self.W / 2, self.W / 2, self.W,
                           device=device) * self.pitch,
            indexing='ij',
        )
        r2 = x ** 2 + y ** 2
        amplitude = torch.exp(-r2 / waist ** 2)
        self.register_buffer('amplitude', amplitude.unsqueeze(0).unsqueeze(0))

        # Circular aperture apodization: eliminates rectangular edge phase
        # artifacts that cause cross-shaped diffraction patterns in the FFT.
        # Smooth sigmoid cutoff at 95% of the shorter grid half-extent.
        half_min = min(phys_H, phys_W) / 2
        aperture_radius = 0.95 * half_min
        edge_width = 0.05 * half_min
        r = torch.sqrt(r2)
        aperture = torch.sigmoid((aperture_radius - r) / edge_width)
        self.register_buffer('aperture', aperture.unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _cft2(x):
        """Centred 2-D Fourier transform (DC stays at array centre)."""
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))

    @staticmethod
    def _icft2(x):
        """Centred 2-D inverse Fourier transform."""
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))

    def forward_field(self, phase_mask):
        """SLM → sensor via centred FT with circular aperture."""
        slm_field = self.amplitude * self.aperture * torch.exp(1j * phase_mask)
        sensor_field = self._cft2(slm_field)
        # Detach max-norm so gradients flow through the brightest pixel
        sensor_field = sensor_field / (sensor_field.abs().max().detach() + 1e-12)
        return sensor_field

    def backward_field(self, sensor_field):
        """Sensor → SLM via centred IFT."""
        slm_field = self._icft2(sensor_field)
        slm_field = slm_field / (slm_field.abs().max().detach() + 1e-12)
        return slm_field

    def forward(self, phase_mask):
        """Simulated intensity |U|² at the sensor plane."""
        field = self.forward_field(phase_mask)
        return field.real ** 2 + field.imag ** 2
