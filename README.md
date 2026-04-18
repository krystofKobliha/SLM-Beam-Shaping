# Beam Shaping for LCOS-SLM

Phase mask generation for a Hamamatsu LCOS-SLM (X15213-16L) using two complementary approaches:

1. **Gerchberg-Saxton (GS)** — iterative FFT-based phase retrieval with optional camera-in-the-loop (CITL) fine-tuning
2. **U-Net CNN** — direct target-intensity → phase-mask prediction, trained with a 7-stage progressive curriculum

Developed as part of a master thesis at CTU Prague / HiLASE Centre.

## Hardware Requirements

| Component | Model | Notes |
|-----------|-------|-------|
| SLM | Hamamatsu X15213-16L | 1024×1280 px, USB control via `hpkSLMdaLV.dll` |
| Camera | CinCam beam profiler | XML-RPC at `localhost:8080` |
| Optics | 3-lens relay | f1=150 mm, f2=150 mm, f3=75 mm (M=0.5) |
| Laser | 515 nm | Second harmonic of Yb:YAG |

> The GS algorithm and U-Net training run without hardware (pure simulation).  
> CITL fine-tuning and calibration require the SLM + camera.

## Project Structure

```
├── common/                     Shared modules
│   ├── config.py               All hardware, physics, and training parameters
│   ├── dataset.py              On-the-fly synthetic target generator
│   ├── extract_camera_data.py  CinCam frame capture (XML-RPC)
│   ├── fft_physics.py          Differentiable FFT physics model (centred FT)
│   ├── loss.py                 Composite loss (blur-Pearson, MSE, TV, BG suppression)
│   ├── measure_obstruction.py  Beam-stop obstruction measurement tool
│   ├── model.py                U-Net architecture (HoloNet)
│   ├── profile_adjustment.py   Camera↔simulation alignment & preprocessing
│   ├── run_calibration.py      Full calibration pipeline (30 random targets)
│   ├── slm_communicate.py      Hamamatsu SLM USB interface
│   ├── slm_export.py           Phase tensor → SLM-compatible BMP export
│   ├── target_generator.py     Target shape generation with DFT aspect correction
│   └── utils.py                Visualization, metrics (PSNR), tensor utilities
├── iterative/                  Iterative (GS) pipeline
│   ├── gerchberg_saxton.py     FFT Gerchberg-Saxton phase retrieval
│   ├── gs_generate.py          CLI runner: GS → optional CITL → export BMP
│   ├── citl_finetune.py        Camera-in-the-loop GS fine-tuning
│   └── generate_random_targets.py  Random targets for calibration
├── unet/                       CNN pipeline
│   ├── train.py                7-stage progressive training loop
│   ├── unet_generate.py        CLI runner: U-Net inference → export BMP
│   └── citl_unet_finetune.py   Camera-in-the-loop U-Net fine-tuning
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch ≥ 2.0, torchvision, NumPy, OpenCV, Matplotlib, Pillow.

For GPU training, install the appropriate CUDA version of PyTorch from [pytorch.org](https://pytorch.org).

## Usage

### 1. Gerchberg-Saxton Phase Retrieval

Edit the user configuration section at the top of `iterative/gs_generate.py`, then run:

```bash
python iterative/gs_generate.py
```

This will:
- Load or generate a target intensity pattern
- Run FFT-based Gerchberg-Saxton to compute a phase mask
- (Optional) Run camera-in-the-loop fine-tuning if hardware is connected
- Export the result as an SLM-compatible BMP

### 2. Camera–Simulation Calibration

Before using CITL, calibrate the camera↔simulation alignment:

```bash
python common/run_calibration.py
```

This displays 30 random patterns on the SLM, captures camera images, and fits a similarity transform (scale, rotation, translation). Results are saved to `calibration_outputs/latest.json`.

### 3. U-Net Training

```bash
python unet/train.py
```

Training uses 7 progressive resolution stages (256×320 → 1024×1280) with:
- FFT-GS teacher supervision (warm-start)
- Cosine-annealed learning rate with warmup
- EMA model tracking
- Convergence-gated stage transitions

Checkpoints are saved to `training_outputs_full/`. Configure in `common/config.py`:
- `SCHEDULE` — per-stage resolution, batch size, learning rate, iterations
- `LOSS_CONFIG` — loss component weights
- `TRAIN_CONFIG` — optimizer and scheduler settings
- `RESUME_CONFIG` — resume from a checkpoint

### 4. U-Net Inference

Edit the user configuration at the top of `unet/unet_generate.py`, then run:

```bash
python unet/unet_generate.py
```

This loads a trained checkpoint, generates a phase mask from a target in a single forward pass, and exports the SLM-ready BMP.

### 5. CITL U-Net Fine-Tuning

Fine-tune the trained U-Net using real SLM + camera feedback:

```bash
python unet/citl_unet_finetune.py
```

Requires SLM and camera hardware. Uses a camera-to-sim correction ratio so that gradients flow through the differentiable simulation while the loss reflects real hardware quality.

### 6. Using a Trained Model (Python API)

Load a checkpoint and generate phase masks:

```python
import torch
from common.model import HoloNet
from common.slm_export import phase_to_slm_bmp

model = HoloNet()
model.load_state_dict(torch.load("holonet_best_S7.pth", weights_only=True))
model.eval()

# target: [1, 1, 1024, 1280] tensor, values in [0, 1]
with torch.no_grad():
    phase = model(target)

phase_to_slm_bmp(phase, "output_phase.bmp")
```

## Key Concepts

- **FFT physics model** (`common/fft_physics.py`): models the 3-lens relay as a single centred Fourier transform with Gaussian beam illumination and circular aperture apodization. Exact forward/backward inverses ensure no noise accumulation during GS iterations.
- **Beam-stop masking**: a zero-order block in the Fourier plane is accounted for in both calibration and loss computation. Use `common/measure_obstruction.py` to measure its extent on a real setup.
- **CITL fine-tuning** (`iterative/citl_finetune.py`): iterative target pre-compensation using multiplicative ratio correction between blurred camera captures and the effective target.
- **CITL U-Net fine-tuning** (`unet/citl_unet_finetune.py`): hardware-in-the-loop gradient-based fine-tuning of the trained U-Net using a camera-to-sim correction ratio for differentiable backpropagation.

## License

This code accompanies a master thesis. See the thesis document for full details.
