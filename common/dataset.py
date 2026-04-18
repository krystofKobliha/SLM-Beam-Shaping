# ──────────────────────────────────────────────────────────────────────────────
# File: dataset.py             Project: DNN-SLM Phase Mask Generation
# Purpose: On-the-fly synthetic target generation (physics-constrained)
# ──────────────────────────────────────────────────────────────────────────────
"""
On-the-fly synthetic target generator for training.

Produces random shapes (circles, rings, rectangles) and text characters that
respect the physical constraints of the optical system: diffraction limit and
Gaussian beam waist. Everything outside the valid zone is zeroed out.
"""
import torch
import random
import numpy as np
import cv2
import torchvision.transforms as T
import math

class SyntheticGenerator:
    def __init__(self, device, resolution, input_energy=1.0, 
                 wavelength=515e-9, 
                 pixel_pitch=12.42e-6,
                 focal_length=0.075,
                 waist_ratio=0.45):  
        self.device = device
        self.H, self.W = resolution
        
        self.pixel_pitch = pixel_pitch
        self.sensor_width = self.W * self.pixel_pitch
        self.sensor_height = self.H * self.pixel_pitch

        # Valid zone = 1× Gaussian beam waist (I drops to 1/e² ≈ 13.5 % at r = w)
        waist_meters = min(self.sensor_width, self.sensor_height) * waist_ratio
        self.limit_px = waist_meters / self.pixel_pitch   # beam waist in pixels

        # Safe radius for placing object CENTRES (80 % of beam waist so that
        # finite-size objects don't protrude beyond the valid zone).
        self.safe_radius_px = self.limit_px * 0.80

        # Minimum feature size (Nyquist: λ f_equiv / SLM_aperture, ≈ 1-2 px).
        # Not restrictive in practice; stored for information only.
        slm_aperture = max(self.sensor_width, self.sensor_height)
        self.min_feature_m = wavelength * focal_length / slm_aperture
        self.min_feature_px = max(2.0, self.min_feature_m / self.pixel_pitch)

        # Pre-calculate Grid for Masking
        self.y_grid, self.x_grid = torch.meshgrid(
            torch.arange(self.H, device=self.device, dtype=torch.float32),
            torch.arange(self.W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Hard physics mask — circular, radius = beam waist
        center_y, center_x = self.H / 2, self.W / 2
        dist_sq = (self.x_grid - center_x)**2 + (self.y_grid - center_y)**2
        self.physics_mask = (dist_sq <= self.limit_px**2).float()

        print(f"Generator Init: {self.H}x{self.W}px | Pitch: {self.pixel_pitch*1e6:.1f}µm | "
              f"Beam waist: {self.limit_px:.0f}px | Safe radius: {self.safe_radius_px:.0f}px | "
              f"Min feature: {self.min_feature_px:.1f}px")

    def _get_safe_center(self):
        r = random.uniform(0, self.safe_radius_px * 0.5)
        theta = random.uniform(0, 2 * math.pi)
        cx = self.W / 2 + r * math.cos(theta)
        cy = self.H / 2 + r * math.sin(theta)
        return cx, cy

    def get_random_shape_tensor(self):
        cx, cy = self._get_safe_center()
        # Scale object size based on the SAFE ZONE, not the full image
        base_size = max(2.0, self.safe_radius_px * 0.5)
        
        img_tensor = torch.zeros((self.H, self.W), device=self.device)
        shape_type = random.choice(['circle', 'ring', 'rect'])

        if shape_type == 'circle':
            r = random.uniform(base_size * 0.5, base_size)
            dist_sq = (self.x_grid - cx)**2 + (self.y_grid - cy)**2
            mask = torch.sigmoid((r - torch.sqrt(dist_sq)) * 2.0)
            img_tensor = mask

        elif shape_type == 'ring':
            r_outer = random.uniform(base_size * 0.6, base_size)
            thickness = r_outer * random.uniform(0.1, 0.4)
            r_inner = r_outer - thickness
            dist = torch.sqrt((self.x_grid - cx)**2 + (self.y_grid - cy)**2)
            mask = torch.sigmoid((dist - r_inner)*2) * torch.sigmoid((r_outer - dist)*2)
            img_tensor = mask

        elif shape_type == 'rect':
            w = random.uniform(base_size * 0.5, base_size)
            h = random.uniform(base_size * 0.5, base_size)
            mask_x = torch.sigmoid((w/2 - torch.abs(self.x_grid - cx)) * 2.0)
            mask_y = torch.sigmoid((h/2 - torch.abs(self.y_grid - cy)) * 2.0)
            img_tensor = mask_x * mask_y
    
        return img_tensor

    def get_random_text_cv2(self):
        img_np = np.zeros((self.H, self.W), dtype=np.uint8)
        text = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        
        # CRITICAL FIX: Scale font to fit inside the Valid Zone
        # Font scale 1.0 is approx 22 pixels high in Hershey Simplex
        target_height_px = max(5, self.safe_radius_px * 1.2) 
        font_scale = target_height_px / 22.0
        thickness = max(1, int(font_scale * 2))
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cx, cy = self._get_safe_center()
        
        x = int(cx - tw // 2)
        y = int(cy + th // 2)
        
        cv2.putText(img_np, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness)
        return torch.from_numpy(img_np).float().to(self.device) / 255.0

    def apply_texture(self, canvas):
        """Applies a consistent texture 'pane' across the existing shapes."""
        noise = torch.rand((self.H, self.W), device=self.device)
        blurrer = T.GaussianBlur(kernel_size=15, sigma=5)
        texture = blurrer(noise.unsqueeze(0)).squeeze(0)
        
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        texture = texture * 0.8 + 0.2 
        
        return canvas * texture

    def sample_batch(self, batch_size):
        targets = torch.zeros((batch_size, 1, self.H, self.W), device=self.device)
        
        for b in range(batch_size):
            # --- MODIFIED: Always exactly 1 object ---
            canvas = torch.zeros((self.H, self.W), device=self.device)
            
            # 50% chance of Shape, 50% chance of Text
            if random.random() < 0.5:
                obj = self.get_random_shape_tensor()
            else:
                obj = self.get_random_text_cv2()
            
            intensity = random.uniform(0.7, 1.0)
            canvas = torch.maximum(canvas, obj * intensity)
            
            # Optional texture
            if random.random() < 0.8: 
                canvas = self.apply_texture(canvas)

            # Safety check
            if canvas.max() < 0.1:
                canvas = self.get_random_shape_tensor()
            
            # --- FINAL PHYSICS MASK ---
            # Strictly enforce the limit. No signal allowed outside.
            canvas = canvas * self.physics_mask

            targets[b, 0] = canvas
            
        return targets