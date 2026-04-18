# ──────────────────────────────────────────────────────────────────────────────
# File: config.py              Project: DNN-SLM Phase Mask Generation
# Purpose: Single source of truth for all hardware, physics, and training params
# ──────────────────────────────────────────────────────────────────────────────
"""
Centralized configuration for the DNN-SLM project.
All hardware constants, physics parameters, training schedules, and paths
are defined here. Every other module imports from this file.
"""
import os

# ---------------------------------------------------------------------------
# 1. HARDWARE CONSTANTS  (Hamamatsu X15213-16L LCOS-SLM, 1280×1024)
# ---------------------------------------------------------------------------
REAL_SENSOR_WIDTH = 15.9e-3          # metres
REAL_PIXELS_W = 1280
SLM_PITCH = REAL_SENSOR_WIDTH / REAL_PIXELS_W   # ≈ 12.42 µm

SLM_ACTIVE_WIDTH_PX = 1272           # active area after 4 px crop per side
SLM_HEIGHT_PX = 1024
SLM_GRAY_2PI = 213                   # grayscale value corresponding to 2π

# Per-pixel wavefront calibration mask (added mod 2π before display)
SLM_CALIB_BMP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'CAL_LSH0804604_515nm.bmp',
)
SLM_CALIB_GRAY_2PI = 254             # calibration BMP: gray 254 = 2π

# Hamamatsu USB Control SDK DLL (64-bit cdecl)
SLM_DLL_PATH = r"C:\Users\Administrator\Desktop\Krystof\SLM_Hamamatsu_manual_and_control\USB_Control_SDK\hpkSLMdaLV_cdecl_64bit\hpkSLMdaLV.dll"

SLM_DEVICE_ID = 0
SLM_SLOT_NUMBER = 0

CAMERA_XMLRPC_URL = "http://localhost:8080/"

# ---------------------------------------------------------------------------
# Beam-stop (zero-order block) geometry — fractional coordinates
# ---------------------------------------------------------------------------
BEAM_STOP_X_CENTER     = 0.4539  # horizontal centre (fraction of W)
BEAM_STOP_X_HALF_WIDTH = 0.0711  # half-width (fraction of W)
BEAM_STOP_Y_START      = 0.3770  # top edge (fraction of H)
BEAM_STOP_Y_END        = 1.00    # bottom edge (fraction of H)

# ---------------------------------------------------------------------------
# 2. PHYSICS / OPTICS
# ---------------------------------------------------------------------------
PHYSICS_CONFIG = {
    'wavelength': 515e-9,
    'pixel_pitch': SLM_PITCH,
    'f1': 0.150,
    'f2': 0.150,
    'f3': 0.075,                     # magnification M = f3/f1 = 0.5
    'waist_ratio': 0.45,
}

# ---------------------------------------------------------------------------
# 3. PROGRESSIVE TRAINING SCHEDULE (7 stages: 256×320 → 1024×1280)
# ---------------------------------------------------------------------------
SCHEDULE = [
    {'res': (256, 320),   'batch': 16, 'lr': 3e-4,  'iters': 30_000,  'min_iters': 20_000,
     'teacher_warmup_fraction': 0.15, 'margin': 0.15, 'opt_steps': 60,  'tv': 0.001,   'accum': 2},
    {'res': (384, 480),   'batch': 12, 'lr': 2e-4,  'iters': 30_000,  'min_iters': 20_000,
     'teacher_warmup_fraction': 0.20, 'margin': 0.12, 'opt_steps': 70,  'tv': 0.0006,  'accum': 2},
    {'res': (512, 640),   'batch': 8,  'lr': 1.5e-4,'iters': 40_000,  'min_iters': 25_000,
     'teacher_warmup_fraction': 0.30, 'margin': 0.12, 'opt_steps': 80,  'tv': 0.0004,  'accum': 4},
    {'res': (640, 800),   'batch': 6,  'lr': 1.2e-4,'iters': 40_000,  'min_iters': 25_000,
     'teacher_warmup_fraction': 0.30, 'margin': 0.10, 'opt_steps': 80,  'tv': 0.0003,  'accum': 4},
    {'res': (768, 960),   'batch': 4,  'lr': 1e-4,  'iters': 40_000,  'min_iters': 25_000,
     'teacher_warmup_fraction': 0.25, 'margin': 0.10, 'opt_steps': 100, 'tv': 0.0002,  'accum': 8},
    {'res': (896, 1120),  'batch': 2,  'lr': 8e-5,  'iters': 40_000,  'min_iters': 25_000,
     'teacher_warmup_fraction': 0.25, 'margin': 0.08, 'opt_steps': 100, 'tv': 0.00015, 'accum': 8},
    {'res': (1024, 1280), 'batch': 2,  'lr': 4e-5,  'iters': 100_000, 'min_iters': 40_000,
     'teacher_warmup_fraction': 0.10, 'margin': 0.08, 'opt_steps': 120, 'tv': 0.005,   'accum': 16},
]

# ---------------------------------------------------------------------------
# 4. LOSS WEIGHTS
# ---------------------------------------------------------------------------
LOSS_CONFIG = {
    'tv_weight': 0.001,
    'pearson_weight': 1.0,
    'supervised_physics_weight': 1.0,
    'bg_suppression_weight': 1.0,
    'phase_variance_weight': 0.15,
}

# ---------------------------------------------------------------------------
# 5. TRAINING HYPER-PARAMETERS
# ---------------------------------------------------------------------------
TRAIN_CONFIG = {
    'grad_clip': 1.0,
    'scheduler_factor': 0.5,
    'scheduler_patience': 10,
    'scheduler_min_lr': 5e-7,
    'val_interval': 200,
    'vis_interval': 500,
    'accum_steps': 1,
    'ema_decay': 0.999,
    'warmup_fraction': 0.08,
    'teacher_warmup_fraction': 0.20,
    'plateau_window_iters': 80000,
    'plateau_delta_psnr': 0.05,
    'phase_noise_std': 0.0,
    'restart_period_iters': 0,
    'restart_decay': 0.85,
}

# ---------------------------------------------------------------------------
# 6. PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "training_outputs_full")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESUME_CONFIG = {
    'resume_from': None,   # path to .pth checkpoint, or None to train from scratch
    'start_stage': 1,      # 1-indexed stage to resume from
}

DEFAULT_MASK_PATH = r"C:\Users\Administrator\Desktop\Krystof\DNN_training\Phase_masks\1.bmp"