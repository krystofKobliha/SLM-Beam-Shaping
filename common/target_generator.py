# ──────────────────────────────────────────────────────────────────────────────
# File: target_generator.py    Project: DNN-SLM Phase Mask Generation
# Purpose: Target intensity shape generation and image utilities
# ──────────────────────────────────────────────────────────────────────────────
"""
Synthetic target generator for the Gerchberg–Saxton pipeline.

Produces uniform-intensity shapes (circle, ring, rect, text) and loads
external images, all masked to the valid beam-waist region.
Also contains low-level image helpers (Gaussian blur) used across modules.
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from config import PHYSICS_CONFIG, SLM_PITCH


# ---------------------------------------------------------------------------
# Gaussian blur helpers
# ---------------------------------------------------------------------------
def gaussian_blur_tensor(tensor, sigma):
    """Apply separable Gaussian blur to a [1, 1, H, W] tensor."""
    if sigma <= 0:
        return tensor
    size = int(6 * sigma + 1) | 1
    coords = torch.arange(size, device=tensor.device, dtype=tensor.dtype) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kw = g.reshape(1, 1, 1, size)
    kh = g.reshape(1, 1, size, 1)
    out = F.conv2d(tensor, kw, padding=(0, size // 2))
    out = F.conv2d(out, kh, padding=(size // 2, 0))
    return out


def gaussian_blur_complex(field, sigma):
    """Gaussian blur on a complex phasor field (preserves phase cyclicity)."""
    if sigma <= 0:
        return field
    real = field.real
    imag = field.imag
    needs_expand = real.ndim == 2
    if needs_expand:
        real = real.unsqueeze(0).unsqueeze(0)
        imag = imag.unsqueeze(0).unsqueeze(0)
    blurred = torch.complex(
        gaussian_blur_tensor(real, sigma),
        gaussian_blur_tensor(imag, sigma),
    )
    if needs_expand:
        blurred = blurred.squeeze(0).squeeze(0)
    return blurred


# ---------------------------------------------------------------------------
# Physics mask (beam-waist hard aperture)
# ---------------------------------------------------------------------------
def physics_mask(h, w, device, pixel_pitch=None):
    """Hard circular mask at 1× beam waist (matches dataset.py convention)."""
    pp = pixel_pitch if pixel_pitch is not None else SLM_PITCH
    phys_h = h * pp
    phys_w = w * pp
    waist_m = min(phys_h, phys_w) * PHYSICS_CONFIG.get('waist_ratio', 0.45)
    limit_px = waist_m / pp

    cy, cx = h / 2, w / 2
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    return (dist_sq <= limit_px ** 2).float()


# ---------------------------------------------------------------------------
# Shape targets
# ---------------------------------------------------------------------------
def make_target(shape, resolution, device,
                cx_offset=0, cy_offset=0,
                radius=None, width=None, height=None, thickness=None,
                character=None, font_size=None, pixel_pitch=None,
                dft_aspect_correct=True):
    """
    Create a uniform-intensity target at the specified position.

    Coordinate system: picture centre = (0, 0).
      cx_offset > 0 → shape moves RIGHT
      cy_offset > 0 → shape moves DOWN

    All dimensions are in pixels at the simulation resolution (1024×1280).
    If pixel_pitch is provided, it overrides the global SLM_PITCH for
    beam-waist calculation (use REAL_SLM_PITCH for full-resolution targets).

    DFT aspect correction (on by default):
      The FFT output has non-square physical pixel pitch when the grid is
      non-square (e.g. 1280×1024).  Output pixel pitch in y is larger by
      factor W/H ≈ 1.25.  This makes a 150×150 px target appear ~25 %
      taller than wide on camera.  When dft_aspect_correct=True, the
      target x-coordinates are pre-scaled by W/H so that WIDTH, HEIGHT,
      and RADIUS specify the *physical* (camera-visible) dimensions.

    Returns:
        [1, 1, H, W] float tensor in [0, 1].
    """
    h, w = resolution
    pic_cy, pic_cx = h / 2.0, w / 2.0
    scx = pic_cx + cx_offset
    scy = pic_cy + cy_offset

    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # DFT aspect ratio: on a non-square grid the output pixel pitch differs
    # between x and y.  dft_ax scales x-coordinates so that shape dimensions
    # specified in pixels correspond to equal physical sizes on camera.
    #   Δξ_y / Δξ_x = W / H  →  x dimensions must be stretched by W/H.
    dft_ax = (w / h) if dft_aspect_correct else 1.0

    # Beam waist in pixels (for default size hints)
    pp = pixel_pitch if pixel_pitch is not None else SLM_PITCH
    phys_h = h * pp
    phys_w = w * pp
    waist_m = min(phys_h, phys_w) * PHYSICS_CONFIG.get('waist_ratio', 0.45)
    beam_wpx = waist_m / pp

    if shape == 'circle':
        r = radius if radius is not None else beam_wpx * 0.3
        # Anisotropic distance: scale x by h/w so the circle appears round
        dx = (xx - scx) * (1.0 / dft_ax)
        dy = yy - scy
        dist = torch.sqrt(dx ** 2 + dy ** 2)
        canvas = torch.sigmoid((r - dist) * 2.0)

    elif shape == 'ring':
        r_outer = radius if radius is not None else beam_wpx * 0.4
        th = thickness if thickness is not None else r_outer * 0.25
        r_inner = r_outer - th
        dx = (xx - scx) * (1.0 / dft_ax)
        dy = yy - scy
        dist = torch.sqrt(dx ** 2 + dy ** 2)
        canvas = (torch.sigmoid((dist - r_inner) * 2.0)
                  * torch.sigmoid((r_outer - dist) * 2.0))

    elif shape == 'rect':
        rw = (width if width is not None else beam_wpx * 0.5) / 2.0 * dft_ax
        rh = (height if height is not None else beam_wpx * 0.4) / 2.0
        mask_x = torch.sigmoid((rw - torch.abs(xx - scx)) * 2.0)
        mask_y = torch.sigmoid((rh - torch.abs(yy - scy)) * 2.0)
        canvas = mask_x * mask_y

    elif shape == 'triangle':
        r = radius if radius is not None else beam_wpx * 0.35
        # Equilateral triangle with vertices at 3 equally-spaced angles
        # Scale x-coordinates by dft_ax for aspect correction
        angles = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3]
        pts = np.array([[int(scx + r * np.cos(a) * dft_ax),
                         int(scy - r * np.sin(a))]
                        for a in angles], dtype=np.int32)
        img_np = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(img_np, [pts], 255)
        canvas = torch.from_numpy(img_np).float().to(device) / 255.0

    elif shape == 'text':
        char = character if character is not None else 'A'
        fs = font_size if font_size is not None else beam_wpx * 0.6
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = fs / 22.0
        line_th = max(1, int(font_scale * 2))
        (tw, th_t), _ = cv2.getTextSize(char, font, font_scale, line_th)
        # Render text at native aspect, then stretch x for DFT correction
        img_np = np.zeros((h, w), dtype=np.uint8)
        tx = int(scx - tw * dft_ax / 2)
        ty = int(scy + th_t / 2)
        cv2.putText(img_np, char, (tx, ty), font, font_scale, 255, line_th)
        if dft_aspect_correct and abs(dft_ax - 1.0) > 0.01:
            # Stretch the rendered text horizontally by dft_ax
            new_w = int(round(w * dft_ax))
            stretched = cv2.resize(img_np, (new_w, h),
                                   interpolation=cv2.INTER_LINEAR)
            # Crop/pad back to original width, centred on scx
            cx_in_stretched = int(round(scx * dft_ax))
            x0 = cx_in_stretched - int(scx)
            x0 = max(0, x0)
            x1 = x0 + w
            if x1 > new_w:
                x1 = new_w
                x0 = x1 - w
            img_np = stretched[:, x0:x1]
        canvas = torch.from_numpy(img_np).float().to(device) / 255.0

    else:
        raise ValueError(f"Unknown shape '{shape}'. Use circle, ring, rect, triangle, or text.")

    # Apply physics mask and normalise
    canvas = canvas * physics_mask(h, w, device, pixel_pitch=pp)
    t_max = canvas.max()
    if t_max > 0:
        canvas = canvas / t_max

    target = canvas.unsqueeze(0).unsqueeze(0)
    return target


# ---------------------------------------------------------------------------
# External image loading
# ---------------------------------------------------------------------------
def load_target_image(path, resolution, device, pixel_pitch=None):
    """Load a grayscale target image, resize, apply physics mask."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    h, w = resolution
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    pp = pixel_pitch if pixel_pitch is not None else SLM_PITCH
    tensor = tensor * physics_mask(h, w, device, pixel_pitch=pp).unsqueeze(0).unsqueeze(0)

    t_max = tensor.max()
    if t_max > 0:
        tensor = tensor / t_max
    return tensor
