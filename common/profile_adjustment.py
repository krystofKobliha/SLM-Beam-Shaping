# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# File: profile_adjustment.py   Project: DNN-SLM Phase Mask Generation
# Purpose: Camera <-> simulation alignment via ORB feature matching + affine warp
# ------------------------------------------------------------------------------
import cv2
import json
import os
import numpy as np
import torch

from config import (BEAM_STOP_X_CENTER, BEAM_STOP_X_HALF_WIDTH,
                    BEAM_STOP_Y_START, BEAM_STOP_Y_END)


def _make_beam_stop_mask(h, w):
    """Return uint8 mask (HxW): 255 everywhere except beam-stop shadow (0)."""
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if BEAM_STOP_X_HALF_WIDTH <= 0:
        return mask
    cx = int(round(w * BEAM_STOP_X_CENTER))
    hw = int(round(w * BEAM_STOP_X_HALF_WIDTH))
    y0 = int(round(h * BEAM_STOP_Y_START))
    y1 = int(round(h * BEAM_STOP_Y_END))
    x0 = max(0, cx - hw)
    x1 = min(w, cx + hw)
    mask[y0:y1, x0:x1] = 0
    return mask


class IntensityAligner:
    def __init__(self):
        self.scale = 1.0
        self.shift_x = 0.0
        self.shift_y = 0.0
        self.rotation = 0.0          # rotation angle in radians
        self.is_calibrated = False

    # ------------------------------------------------------------------
    # Persistence: save / load calibration to JSON
    # ------------------------------------------------------------------

    def save(self, path):
        """Save calibrated transform parameters to a JSON file."""
        import json
        data = {
            'scale':         self.scale,
            'shift_x':       self.shift_x,
            'shift_y':       self.shift_y,
            'rotation_rad':  self.rotation,
            'is_calibrated': self.is_calibrated,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[IntensityAligner] Saved calibration -> {path}")

    def load(self, path):
        """Load calibration from a JSON file saved by save().

        Returns True on success, False if the file is missing or invalid.
        """
        import json
        if not os.path.isfile(path):
            return False
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.scale        = float(data['scale'])
            self.shift_x      = float(data['shift_x'])
            self.shift_y      = float(data['shift_y'])
            self.rotation     = float(data['rotation_rad'])
            self.is_calibrated = bool(data.get('is_calibrated', True))
            print(f"[IntensityAligner] Loaded calibration from {path}")
            print(f"  scale={self.scale:.6f}  shift=({self.shift_x:.1f}, "
                  f"{self.shift_y:.1f})  rot={np.degrees(self.rotation):.4f} deg")
            return True
        except Exception as e:
            print(f"[IntensityAligner] WARNING: could not load {path}: {e}")
            return False

    # ------------------------------------------------------------------
    # Centroid extraction (robust to sparse holographic patterns)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_centroid(img, beam_stop_mask=None):
        """Compute the intensity-weighted centroid of the bright region.

        Returns (cx, cy) or None if the image is too dark.

        Steps:
            1. Gaussian blur to smooth speckle noise.
            2. Apply beam-stop mask (zero out blocked region).
            3. Threshold at 15 % of the frame maximum.
            4. Compute intensity-weighted centroid via cv2.moments.
        """
        blurred = cv2.GaussianBlur(img.astype(np.float32), (15, 15), 4)
        if beam_stop_mask is not None:
            blurred = blurred * (beam_stop_mask.astype(np.float32) / 255.0)
        peak = blurred.max()
        if peak < 3:
            return None
        thresh_val = peak * 0.15
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        M = cv2.moments(binary)
        if M['m00'] < 1:
            return None
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        return (cx, cy)

    def find_centroid_pair(self, target_img, measured_img):
        """Return (target_centroid, camera_centroid) or None if either fails."""
        h_t, w_t = target_img.shape[:2]
        h_m, w_m = measured_img.shape[:2]
        mask_tgt  = _make_beam_stop_mask(h_t, w_t)
        mask_meas = _make_beam_stop_mask(h_m, w_m)
        tgt_c = self._get_centroid(target_img,  mask_tgt)
        cam_c = self._get_centroid(measured_img, mask_meas)
        if tgt_c is None or cam_c is None:
            return None
        return tgt_c, cam_c

    def calibrate_by_centroids(self, target_list, measured_list):
        """Fit a similarity transform from centroid correspondences.

        For each (target, camera) pair, extracts the intensity-weighted
        centroid.  With N >= 2 successful pairs at different positions, solves
        for the 4-parameter similarity transform (s, theta, tx, ty) using
        cv2.estimateAffinePartial2D with RANSAC.

        This replaces ORB feature matching, which fails on sparse holographic
        patterns with very few textured pixels.
        """
        if len(target_list) != len(measured_list):
            raise ValueError("Target and measured lists must have equal length.")

        tgt_pts = []
        cam_pts = []

        print(f"Centroid calibration with {len(target_list)} pairs ...")
        for i, (tgt, meas) in enumerate(zip(target_list, measured_list)):
            result = self.find_centroid_pair(tgt, meas)
            if result is not None:
                tc, cc = result
                tgt_pts.append(tc)
                cam_pts.append(cc)
                print(f"  Pair {i+1}: target=({tc[0]:.1f}, {tc[1]:.1f})  "
                      f"camera=({cc[0]:.1f}, {cc[1]:.1f})")
            else:
                print(f"  Pair {i+1}: centroid extraction failed.")

        if len(tgt_pts) < 2:
            raise RuntimeError(
                f"Centroid calibration failed: only {len(tgt_pts)} centroids "
                f"found (need >= 2).")

        src = np.float32(cam_pts).reshape(-1, 1, 2)
        dst = np.float32(tgt_pts).reshape(-1, 1, 2)

        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=15.0)

        if matrix is None:
            raise RuntimeError("estimateAffinePartial2D returned None.")

        a = matrix[0, 0]
        b = matrix[1, 0]
        self.scale    = float(np.sqrt(a**2 + b**2))
        self.rotation = float(np.arctan2(b, a))
        self.shift_x  = float(matrix[0, 2])
        self.shift_y  = float(matrix[1, 2])
        self.is_calibrated = True

        n_inliers = int(inliers.sum()) if inliers is not None else len(tgt_pts)
        print("-" * 40)
        print("Centroid calibration complete:")
        print(f"  Pairs used:  {n_inliers} / {len(tgt_pts)} (inliers/total)")
        print(f"  Scale:       {self.scale:.6f}")
        print(f"  Shift X:     {self.shift_x:.2f} px")
        print(f"  Shift Y:     {self.shift_y:.2f} px")
        print(f"  Rotation:    {np.degrees(self.rotation):.4f}  deg")

        return {
            'n_pairs': len(tgt_pts),
            'n_inliers': n_inliers,
            'tgt_pts': tgt_pts,
            'cam_pts': cam_pts,
        }

    def apply_correction(self, measured_img):
        """
        Applies the calibrated Scale and Shifts to a new measured image.
        Returns the corrected image.
        """
        if not self.is_calibrated:
            raise RuntimeError("Aligner not calibrated. Run calibrate() first.")

        height, width = measured_img.shape[:2]
        
        # Construct the full similarity transformation matrix
        cos_t = np.cos(self.rotation)
        sin_t = np.sin(self.rotation)
        s = self.scale
        M = np.float32([
            [s * cos_t, -s * sin_t, self.shift_x],
            [s * sin_t,  s * cos_t, self.shift_y]
        ])

        # Warp the image
        corrected_img = cv2.warpAffine(measured_img, M, (width, height))
        return corrected_img


# ---------------------------------------------------------------------------
# HitlPreprocessor: Full camera->tensor pipeline for HITL training
# ---------------------------------------------------------------------------

class HitlPreprocessor:
    """Wraps IntensityAligner with HITL-specific preprocessing.

    Pipeline: background subtraction -> geometric correction -> resize ->
    intensity mapping -> torch.Tensor [1,1,H,W].
    """

    def __init__(self, aligner=None):
        self.aligner = aligner or IntensityAligner()
        self.background = None             # average dark frame (float64)
        self.intensity_a = 1.0             # linear mapping: camera ~= a * sim + b
        self.intensity_b = 0.0
        # Beam-stop validity mask (float32, values in {0, 1}).  Lazily built the
        # first time camera_to_tensor is called and cached for that resolution.
        self._beam_stop_mask_cache: dict = {}   # (H, W) -> np.ndarray float32

    # --- Background ---
    def capture_background(self, camera_func, n_frames=10):
        """Capture n dark frames (laser blocked / flat phase) and average.

        Args:
            camera_func: callable returning (np.ndarray HxW uint8, raw_max).
            n_frames: number of frames to average.
        """
        accum = None
        count = 0
        for _ in range(n_frames):
            frame, _ = camera_func()
            if frame is None:
                continue
            if accum is None:
                accum = frame.astype(np.float64)
            else:
                accum += frame.astype(np.float64)
            count += 1
        if count == 0:
            raise RuntimeError("capture_background: all frames failed")
        self.background = accum / count

    def subtract_background(self, frame):
        """Subtract stored background, clip to 0."""
        if self.background is None:
            return frame.astype(np.float32)
        return np.clip(frame.astype(np.float64) - self.background, 0, None).astype(np.float32)

    # --- Intensity mapping ---
    def compute_intensity_mapping(self, sim_tensors, camera_tensors):
        """Compute linear regression  camera ~= a * sim + b.

        Args:
            sim_tensors:    list of 2-D numpy float arrays (simulated intensities,
                            already energy-normalised to [0, 1]).
            camera_tensors: list of 2-D numpy float arrays (preprocessed camera
                            frames, same spatial size, geometric-corrected).
        """
        x_all, y_all = [], []
        for s, c in zip(sim_tensors, camera_tensors):
            x_all.append(s.ravel())
            y_all.append(c.ravel())
        x = np.concatenate(x_all)
        y = np.concatenate(y_all)

        # Least-squares: y = a*x + b
        A = np.vstack([x, np.ones_like(x)]).T
        result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self.intensity_a = float(result[0])
        self.intensity_b = float(result[1])

    # --- Full pipeline ---
    def camera_to_tensor(self, camera_frame, device, resolution):
        """Convert a raw camera frame to a physics-compatible tensor.

        Steps:
            1. Background subtract
            2. Geometric correction (IntensityAligner)
            3. Resize to target resolution
            4. Intensity mapping (linear inverse: sim ~= (cam - b) / a)
            5. Clamp to non-negative, normalise to [0, 1]

        Args:
            camera_frame: np.ndarray (H, W) uint8 from camera
            device: torch device
            resolution: (H, W) physics resolution

        Returns:
            torch.Tensor [1, 1, H, W] float32 on `device`
        """
        # 1. Background subtract
        frame = self.subtract_background(camera_frame)

        # 2. Geometric correction (keep float32 to avoid quantisation loss)
        if self.aligner.is_calibrated:
            height, width = frame.shape[:2]
            cos_t = np.cos(self.aligner.rotation)
            sin_t = np.sin(self.aligner.rotation)
            s = self.aligner.scale
            M = np.float32([
                [s * cos_t, -s * sin_t, self.aligner.shift_x],
                [s * sin_t,  s * cos_t, self.aligner.shift_y]
            ])
            frame = cv2.warpAffine(frame, M, (width, height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0).astype(np.float32)

        # 3. Resize
        target_h, target_w = resolution
        if frame.shape != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 4. Intensity mapping (invert: sim ~= (cam - b) / a)
        a = self.intensity_a if abs(self.intensity_a) > 1e-12 else 1.0
        frame = (frame - self.intensity_b) / a

        # 5. Zero-out beam-stop shadow (pixels in the blocked lower region
        #    carry no real optical signal and must not contribute to the loss).
        key = (target_h, target_w)
        if key not in self._beam_stop_mask_cache:
            np_mask = _make_beam_stop_mask(target_h, target_w).astype(np.float32) / 255.0
            self._beam_stop_mask_cache[key] = np_mask
        frame = frame * self._beam_stop_mask_cache[key]

        # 6. Clamp & normalise (over valid pixels only)
        frame = np.clip(frame, 0, None)
        fmax = frame.max()
        if fmax > 0:
            frame = frame / fmax

        tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).to(device)
        return tensor


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Load your images here (grayscale recommended for intensity)
    # This is dummy code to demonstrate usage
    
    # 1. Load your data (Replace with your actual image loading logic)
    # targets = [cv2.imread(f'target_{i}.png', 0) for i in range(3)]
    # measured = [cv2.imread(f'measured_{i}.png', 0) for i in range(3)]
    
    # For demonstration, let's create a fake example
    img = np.zeros((1024, 1280), dtype=np.uint8)
    cv2.circle(img, (600, 500), 100, 255, -1) # Target Blob
    
    # Create a "Measured" image that is shifted and scaled down
    M_fake = np.float32([[0.8, 0, 50], [0, 0.8, 100]]) # Scale 0.8, Shift +50, +100
    measured_img = cv2.warpAffine(img, M_fake, (1280, 1024))
    
    targets = [img]
    measured_list = [measured_img]

    # 2. Run the Aligner
    aligner = IntensityAligner()
    aligner.calibrate(targets, measured_list)

    # 3. Apply to new image
    corrected = aligner.apply_correction(measured_list[0])
    
    # Verify results (Optional visualization)
    # cv2.imshow("Target", img)
    # cv2.imshow("Measured", measured_list[0])
    # cv2.imshow("Corrected", corrected)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()