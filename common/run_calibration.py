# ------------------------------------------------------------------------------
# File: run_calibration.py      Project: DNN-SLM Phase Mask Generation
# Purpose: Full camera <-> simulation calibration run using the 30 random
#          mask/target pairs in random_targets_and_masks/.
#
# Run from repo root:  python run_calibration.py
#
# Output structure (timestamped, under calibration_outputs/):
#   run_YYYYMMDD_HHMMSS/
#     calibration_summary.txt       <- final transform + per-pair table
#     01_raw_camera_profiles/       <- raw_01.png + raw_01.npy  per mask
#     02_per_pair_transforms/       <- per_pair_transforms.csv + .json
#     03_corrected_profiles/        <- corrected_01.png + corrected_01.npy
# ------------------------------------------------------------------------------
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'common'))

import re
import glob
import time
import csv
import json
import datetime
import numpy as np
import cv2
from PIL import Image

from slm_communicate import SLMController
from extract_camera_data import get_single_frame
from profile_adjustment import IntensityAligner

# -- Settings ------------------------------------------------------------------
N_CAPTURE_FRAMES  = 5     # camera frames averaged per mask
STABILIZATION_SEC = 3.0   # SLM settle time after each display [s]

# -- Paths ---------------------------------------------------------------------
MASKS_DIR   = os.path.join(REPO_ROOT, "random_targets_and_masks", "random_masks")
TARGETS_DIR = os.path.join(REPO_ROOT, "random_targets_and_masks", "random_targets")

# -- Output directories --------------------------------------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_root  = os.path.join(REPO_ROOT, "calibration_outputs", f"run_{timestamp}")
dir_raw   = os.path.join(out_root, "01_raw_camera_profiles")
dir_xform = os.path.join(out_root, "02_per_pair_transforms")
dir_corr  = os.path.join(out_root, "03_corrected_profiles")
for d in (dir_raw, dir_xform, dir_corr):
    os.makedirs(d, exist_ok=True)

print(f"Output directory: {out_root}\n")


# -- Helpers -------------------------------------------------------------------

def _extract_number(path):
    """Pull the trailing integer from e.g. 'mask_7.bmp' -> 7."""
    m = re.search(r'_(\d+)\.[^.]+$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _save_frame(frame, name, folder):
    """Save *frame* (uint8 HxW) as both a PNG (visual) and a NPY (lossless)."""
    cv2.imwrite(os.path.join(folder, f"{name}.png"), frame)
    np.save(os.path.join(folder, f"{name}.npy"), frame)


def _load_target_bmp(path, cam_h, cam_w):
    """Load a target BMP as uint8 grayscale, resized to match camera resolution.

    ORB keypoints are in pixel coordinates, so both images should be at the
    same scale for the affine matrix to be meaningful in camera-pixel units.
    """
    img = np.array(Image.open(path).convert('L'), dtype=np.uint8)
    if img.shape != (cam_h, cam_w):
        img = cv2.resize(img, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
    return img


# -- Discover and sort pairs ---------------------------------------------------
mask_paths   = sorted(glob.glob(os.path.join(MASKS_DIR,   "mask_*.bmp")),
                      key=_extract_number)
target_paths = sorted(glob.glob(os.path.join(TARGETS_DIR, "target_*.bmp")),
                      key=_extract_number)

mask_by_num   = {_extract_number(p): p for p in mask_paths}
target_by_num = {_extract_number(p): p for p in target_paths}
common_nums   = sorted(set(mask_by_num) & set(target_by_num))

if not common_nums:
    print("ERROR: No matching mask/target pairs found. Check folder paths.")
    sys.exit(1)

print(f"Found {len(common_nums)} matching pairs: {common_nums}\n")


# -- Phase 1 - capture raw profiles --------------------------------------------
print("=" * 60)
print("PHASE 1 - Capture raw camera profiles")
print("=" * 60)

raw_frames       = {}   # num -> uint8 HxW
target_imgs      = {}   # num -> uint8 HxW  (resized to camera resolution)

aligner = IntensityAligner()

with SLMController() as slm:
    for num in common_nums:
        mask_path   = mask_by_num[num]
        target_path = target_by_num[num]

        # --- Display mask on SLM ---
        print(f"\n[Pair {num:2d}] {os.path.basename(mask_path)}")
        slm.display_bmp(mask_path)
        time.sleep(STABILIZATION_SEC)

        # --- Capture and average N frames ---
        accum    = None
        captured = 0
        for frame_idx in range(N_CAPTURE_FRAMES):
            frame, raw_max = get_single_frame()
            if frame is None:
                print(f"  WARNING: frame {frame_idx + 1} capture failed - skipping")
                continue
            if accum is None:
                accum = frame.astype(np.float64)
            else:
                accum += frame.astype(np.float64)
            captured += 1

        if captured == 0:
            print(f"  ERROR: all captures failed for pair {num} - skipping pair")
            continue

        avg_frame = np.clip(accum / captured, 0, 255).astype(np.uint8)
        cam_h, cam_w = avg_frame.shape[:2]
        raw_frames[num] = avg_frame

        # --- Load and resize target ---
        tgt = _load_target_bmp(target_path, cam_h, cam_w)
        target_imgs[num] = tgt

        # --- Save raw profile ---
        _save_frame(avg_frame, f"raw_{num:02d}", dir_raw)
        print(f"  Captured  ({captured}/{N_CAPTURE_FRAMES} frames averaged)  "
              f"mean={avg_frame.mean():.1f}  max={avg_frame.max()}")


# -- Phase 2 - centroid extraction + save per-pair data ------------------------
print("\n" + "=" * 60)
print("PHASE 2 - Extract centroids & save per-pair data")
print("=" * 60)

centroid_results = {}   # num -> {'tgt': (cx,cy), 'cam': (cx,cy)} or None

for num in common_nums:
    if num not in raw_frames or num not in target_imgs:
        centroid_results[num] = None
        continue
    result = aligner.find_centroid_pair(target_imgs[num], raw_frames[num])
    if result is not None:
        tc, cc = result
        centroid_results[num] = {
            'tgt_cx': float(tc[0]), 'tgt_cy': float(tc[1]),
            'cam_cx': float(cc[0]), 'cam_cy': float(cc[1]),
        }
        print(f"  Pair {num:2d}: target=({tc[0]:.1f}, {tc[1]:.1f})  "
              f"camera=({cc[0]:.1f}, {cc[1]:.1f})")
    else:
        centroid_results[num] = None
        print(f"  Pair {num:2d}: centroid extraction failed")

csv_path  = os.path.join(dir_xform, "per_pair_centroids.csv")
json_path = os.path.join(dir_xform, "per_pair_centroids.json")

fieldnames = ['pair', 'tgt_cx', 'tgt_cy', 'cam_cx', 'cam_cy', 'status']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for num in common_nums:
        c = centroid_results.get(num)
        if c:
            writer.writerow({'pair': num, **c, 'status': 'ok'})
        else:
            writer.writerow({
                'pair': num,
                'tgt_cx': '', 'tgt_cy': '', 'cam_cx': '', 'cam_cy': '',
                'status': 'failed' if num in raw_frames else 'no_capture',
            })

with open(json_path, 'w') as f:
    json.dump({str(k): v for k, v in centroid_results.items()}, f, indent=2)

print(f"\n  CSV  -> {csv_path}")
print(f"  JSON -> {json_path}")


# -- Phase 3 - fit similarity transform from centroids -------------------------
print("\n" + "=" * 60)
print("PHASE 3 - Fit similarity transform (centroid-based)")
print("=" * 60)

# Build lists for the pairs that have valid centroids
good_nums = [n for n in common_nums if centroid_results.get(n) is not None]
if len(good_nums) < 2:
    print("ERROR: Need at least 2 centroid pairs - cannot compute transform.")
    sys.exit(1)

# Use calibrate_by_centroids (operates on the full image lists)
ordered_tgts  = [target_imgs[n] for n in good_nums]
ordered_cams  = [raw_frames[n]  for n in good_nums]
cal_info = aligner.calibrate_by_centroids(ordered_tgts, ordered_cams)

print(f"\n  Pairs used:   {cal_info['n_inliers']} / {cal_info['n_pairs']}")
print(f"  Scale:        {aligner.scale:.6f}")
print(f"  Shift X:      {aligner.shift_x:.3f} px")
print(f"  Shift Y:      {aligner.shift_y:.3f} px")
print(f"  Rotation:     {np.degrees(aligner.rotation):.4f}  deg")

# Save to timestamped folder AND as latest.json at a fixed path
timestamped_json = os.path.join(out_root, "calibration.json")
latest_json      = os.path.join(REPO_ROOT, "calibration_outputs", "latest.json")
aligner.save(timestamped_json)
aligner.save(latest_json)


# -- Phase 4 - apply final transform and save corrected profiles ---------------
print("\n" + "=" * 60)
print("PHASE 4 - Apply final transform -> corrected profiles")
print("=" * 60)

for num in common_nums:
    if num not in raw_frames:
        print(f"  [Pair {num:2d}] skipped - no captured frame")
        continue
    corrected = aligner.apply_correction(raw_frames[num])
    _save_frame(corrected, f"corrected_{num:02d}", dir_corr)
    print(f"  [Pair {num:2d}] corrected  "
          f"mean={corrected.mean():.1f}  max={corrected.max()}")


# -- Phase 5 - write human-readable summary -----------------------------------
summary_path = os.path.join(out_root, "calibration_summary.txt")
with open(summary_path, 'w') as f:
    f.write(f"Calibration run : {timestamp}\n")
    f.write(f"Method          : centroid-based (intensity-weighted)\n")
    f.write(f"Masks dir       : {MASKS_DIR}\n")
    f.write(f"Targets dir     : {TARGETS_DIR}\n")
    f.write(f"Pairs found     : {len(common_nums)}\n")
    f.write(f"Pairs captured  : {len(raw_frames)}\n")
    f.write(f"Centroids found : {len(good_nums)}\n")
    f.write(f"RANSAC inliers  : {cal_info['n_inliers']}\n")
    f.write(f"Frames averaged : {N_CAPTURE_FRAMES}\n")
    f.write(f"SLM settle [s]  : {STABILIZATION_SEC}\n")
    f.write("\n")
    f.write("-" * 50 + "\n")
    f.write("FINAL TRANSFORM  (centroid RANSAC fit)\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Scale       : {aligner.scale:.6f}\n")
    f.write(f"  Shift X     : {aligner.shift_x:.3f} px\n")
    f.write(f"  Shift Y     : {aligner.shift_y:.3f} px\n")
    f.write(f"  Rotation    : {np.degrees(aligner.rotation):.4f}  deg\n")
    f.write("\n")
    f.write("-" * 50 + "\n")
    f.write("PER-PAIR CENTROIDS\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Pair':>5}  {'Tgt cx':>8}  {'Tgt cy':>8}  "
            f"{'Cam cx':>8}  {'Cam cy':>8}  Status\n")
    for num in common_nums:
        c = centroid_results.get(num)
        if c:
            f.write(f"{num:5d}  {c['tgt_cx']:8.1f}  {c['tgt_cy']:8.1f}  "
                    f"{c['cam_cx']:8.1f}  {c['cam_cy']:8.1f}  ok\n")
        elif num in raw_frames:
            f.write(f"{num:5d}  {'-':>8}  {'-':>8}  {'-':>8}  {'-':>8}  "
                    f"centroid failed\n")
        else:
            f.write(f"{num:5d}  {'-':>8}  {'-':>8}  {'-':>8}  {'-':>8}  "
                    f"capture failed\n")

print(f"\nSummary -> {summary_path}")

# -- Done ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("CALIBRATION COMPLETE")
print(f"  Output root : {out_root}")
print(f"    01_raw_camera_profiles/   - {len(raw_frames)} files")
print(f"    02_per_pair_transforms/   - CSV + JSON")
print(f"    03_corrected_profiles/    - {len(raw_frames)} files")
print(f"    calibration_summary.txt")
print("=" * 60)
