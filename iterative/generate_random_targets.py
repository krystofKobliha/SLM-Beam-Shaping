# ──────────────────────────────────────────────────────────────────────────────
# File: generate_random_targets.py    Project: DNN-SLM Phase Mask Generation
# Purpose: Generate 30 random target intensity images for the GS/GD pipeline
# ──────────────────────────────────────────────────────────────────────────────
"""
Generates NUM_TARGETS random target masks with:
  - |cx_offset| >= 100  (shape displaced left OR right from centre)
  - cy_offset  <= -100  (shape displaced upward from centre)
  - Shapes placed fully within a single quadrant of the beam-waist aperture.
  - Random shape (circle, ring, rect, triangle) and random size.

Outputs (per target):
  random_targets/target_01/target.png   — greyscale visualisation
  random_targets/target_01/target.npy   — [H, W] float32 intensity array
  random_targets/targets_summary.txt    — human-readable parameter log

HOW TO USE:
    cd iterative
    python generate_random_targets.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import math
import random
import json

import numpy as np
import torch

from config import SLM_HEIGHT_PX, SLM_PITCH, PHYSICS_CONFIG
from target_generator import make_target
from slm_export import save_tensor_as_png


# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
NUM_TARGETS  = 30
RANDOM_SEED  = None          # set None for a different result every run
OUTPUT_ROOT  = 'random_targets'

# Resolution to generate targets at (matches the real SLM)
RESOLUTION = (SLM_HEIGHT_PX, 1280)   # (1024, 1280)

# Available shapes to sample for targets (see make_target for details)
SHAPES = ['circle', 'ring', 'rect', 'triangle', 'text']

# Characters to use for text targets
TEXT_CHARS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

# Position bounds (pixels from picture centre)
CX_MIN_ABS = 100    # |cx| is at least this
CX_MAX_ABS = 280    # |cx| is at most this  (stay well inside beam waist)
CY_MIN     = -280   # cy upper bound   (must be <= -100)
CY_MAX     = -100   # cy lower bound   (must be <= -100)
# ──────────────────────────────────────────────────────────────────────────────


# Beam-waist radius in pixels (used to keep shapes inside the aperture)
_BEAM_WAIST_PX = int(min(RESOLUTION) * PHYSICS_CONFIG.get('waist_ratio', 0.45))


def _random_position(rng: random.Random):
    """
    Sample (cx, cy) with |cx| in [CX_MIN_ABS, CX_MAX_ABS]
    and cy in [CY_MIN, CY_MAX] (both <=-100).
    """
    sign = rng.choice([-1, 1])
    cx = sign * rng.randint(CX_MIN_ABS, CX_MAX_ABS)
    cy = rng.randint(CY_MIN, CY_MAX)          # both values are negative
    return cx, cy


def _max_safe_radius(cx: int, cy: int) -> int:
    """
    Approximate radius that keeps a shape centred at (cx, cy) inside the
    beam-waist aperture with a small safety margin.
    """
    dist = math.hypot(cx, cy)
    return max(25, int(_BEAM_WAIST_PX - dist - 10))


def _random_size(shape: str, cx: int, cy: int, rng: random.Random) -> dict:
    """Return a dict of size kwargs suitable for make_target for the given shape."""
    max_r = _max_safe_radius(cx, cy)

    if shape == 'circle':
        radius = rng.randint(25, min(140, max_r))
        return {'radius': radius}

    elif shape == 'ring':
        radius = rng.randint(40, min(140, max_r))
        min_thick = 15
        max_thick = max(min_thick + 1, radius // 2)
        thickness = rng.randint(min_thick, max_thick)
        return {'radius': radius, 'thickness': thickness}

    elif shape == 'rect':
        # Width and height independently, each fits within max_r from centre
        w = rng.randint(40, min(240, max_r * 2))
        h = rng.randint(30, min(180, max_r * 2))
        return {'width': w, 'height': h}

    elif shape == 'triangle':
        radius = rng.randint(30, min(140, max_r))
        return {'radius': radius}

    elif shape == 'text':
        font_size = rng.randint(60, min(200, max_r * 2))
        character = rng.choice(TEXT_CHARS)
        return {'font_size': font_size, 'character': character}

    raise ValueError(f"Unknown shape: {shape}")


def main():
    device = torch.device('cpu')
    print(f"Device : {device}")
    print(f"Resolution : {RESOLUTION[0]}×{RESOLUTION[1]}  |  "
          f"Beam waist ≈ {_BEAM_WAIST_PX} px")
    print(f"Generating {NUM_TARGETS} random targets → {os.path.abspath(OUTPUT_ROOT)}/\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    summary_lines = [
        f"{'#':>3}  {'shape':8}  {'cx':>5}  {'cy':>5}  size",
        "-" * 60,
    ]

    for i in range(1, NUM_TARGETS + 1):
        shape   = rng.choice(SHAPES)
        cx, cy  = _random_position(rng)
        size_kw = _random_size(shape, cx, cy, rng)

        target = make_target(
            shape=shape,
            resolution=RESOLUTION,
            device=device,
            cx_offset=cx,
            cy_offset=cy,
            **size_kw,
        )

        # Skip near-blank text targets (character fell outside aperture)
        if shape == 'text' and target.max() < 0.05:
            # fall back to a circle at the same position
            shape = 'circle'
            size_kw = _random_size('circle', cx, cy, rng)
            target = make_target(
                shape='circle', resolution=RESOLUTION, device=device,
                cx_offset=cx, cy_offset=cy, **size_kw,
            )

        # ── Save outputs ────────────────────────────────────────────────────
        bmp_path  = os.path.join(OUTPUT_ROOT, f'target_{i}.bmp')
        meta_path = os.path.join(OUTPUT_ROOT, f'target_{i}.json')

        save_tensor_as_png(target, bmp_path)   # cv2.imwrite uses extension → 8-bit BMP

        params = {'index': i, 'shape': shape,
                  'cx_offset': cx, 'cy_offset': cy, **size_kw}
        with open(meta_path, 'w') as f:
            json.dump(params, f, indent=2)

        size_str = '  '.join(f'{k}={v}' for k, v in size_kw.items())
        line = f"{i:3d}  {shape:8}  {cx:+5d}  {cy:+5d}  {size_str}"
        summary_lines.append(line)
        print(line)

    # Write summary file
    summary_path = os.path.join(OUTPUT_ROOT, 'targets_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')

    print(f"\nSummary written to: {summary_path}")
    print(f"Done — {NUM_TARGETS} targets saved.")


if __name__ == '__main__':
    main()
