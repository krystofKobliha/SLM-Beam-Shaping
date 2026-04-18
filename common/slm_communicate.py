# ──────────────────────────────────────────────────────────────────────────────
# File: slm_communicate.py      Project: DNN-SLM Phase Mask Generation
# Purpose: Control the Hamamatsu LCOS-SLM via hpkSLMdaLV.dll (USB SDK)
# ──────────────────────────────────────────────────────────────────────────────
"""
Hamamatsu LCOS-SLM USB interface using hpkSLMdaLV.dll (USB Control SDK).

The SLM connects via USB and is controlled through the manufacturer USB SDK DLL.
This is completely separate from Image_Control.dll, which only works when the
SLM is connected as an HDMI/DVI monitor — not relevant for USB connections.

SDK DLL functions used (all __cdecl, all return int32_t error code):

    Open_Dev(bIDList, bIDSize)
        Open USB connection to bIDSize devices. bIDList is populated with
        device IDs on return. Returns 0 on success.

    Close_Dev(bIDList, bIDSize)
        Close the USB connection.

    Write_FMemBMPPath(bID, path, SlotNo)
        Load a BMP file (1272x1024, 8-bit grayscale) from disk into flash
        memory slot SlotNo on the SLM. Returns 0 on success.

    Write_FMemArray(bID, array, arraySize, xPixel, yPixel, SlotNo)
        Load a raw uint8 pixel array into flash memory slot SlotNo.

    Change_DispSlot(bID, SlotNo)
        Switch the SLM display to show the pattern stored in SlotNo.

    Check_HeadSerial(bID, buf, bufSize)
        Read the SLM head serial number string.

Configuration (common/config.py):
    SLM_DLL_PATH    – path to hpkSLMdaLV.dll (cdecl 64-bit)
    SLM_DEVICE_ID   – device index (0 for the first/only SLM)
    SLM_SLOT_NUMBER – flash memory slot to use (0–15)
"""
import os
import math
import ctypes
from ctypes import cdll, c_int32, c_uint8, c_uint32, c_void_p, c_char_p, create_string_buffer
import numpy as np
from PIL import Image

from config import SLM_GRAY_2PI, SLM_DLL_PATH, SLM_DEVICE_ID, SLM_SLOT_NUMBER
from config import SLM_CALIB_BMP, SLM_CALIB_GRAY_2PI


# ---------------------------------------------------------------------------
# Calibration mask loader
# ---------------------------------------------------------------------------

_calib_phase_rad = None   # lazy-loaded 2-D float32 array [1024, 1272]

def _load_calibration_mask():
    """Load the per-pixel phase correction mask (radians, 1024×1272)."""
    global _calib_phase_rad
    if _calib_phase_rad is not None:
        return _calib_phase_rad

    if not os.path.isfile(SLM_CALIB_BMP):
        print(f"[SLM] WARNING: Calibration BMP not found at:\n  {SLM_CALIB_BMP}")
        print("  Per-pixel correction will be DISABLED.  Results may have speckle.")
        _calib_phase_rad = np.zeros((1024, 1272), dtype=np.float32)
        return _calib_phase_rad

    cal_img = np.array(Image.open(SLM_CALIB_BMP).convert('L'), dtype=np.float32)
    # Crop 1280 → 1272 if needed (same as in the Mathematica code, calibration
    # is supplied at 1024×1272 = SLM active area, but just in case).
    if cal_img.shape[1] == 1280:
        cal_img = cal_img[:, 4:-4]
    # Convert gray [0, 254] → radians [0, 2π]
    _calib_phase_rad = cal_img * (2.0 * math.pi / SLM_CALIB_GRAY_2PI)
    print(f"[SLM] Loaded calibration mask: {SLM_CALIB_BMP}")
    print(f"  shape={_calib_phase_rad.shape}  "
          f"phase range=[{_calib_phase_rad.min():.2f}, {_calib_phase_rad.max():.2f}] rad")
    return _calib_phase_rad

# ---------------------------------------------------------------------------
# DLL loader
# ---------------------------------------------------------------------------

def _load_dll():
    """Load hpkSLMdaLV.dll (cdecl 64-bit USB SDK)."""
    dll_path = os.path.abspath(SLM_DLL_PATH)
    if not os.path.isfile(dll_path):
        raise FileNotFoundError(
            f"hpkSLMdaLV.dll not found at:\n  {dll_path}\n"
            "Check SLM_DLL_PATH in config.py."
        )
    dll_dir = os.path.dirname(dll_path)
    os.add_dll_directory(dll_dir)          # Python 3.8+ requirement
    lib = cdll.LoadLibrary(dll_path)       # cdecl calling convention

    # Bind argtypes/restype from hpkSLMdaLV.h (cdecl 64-bit)
    lib.Open_Dev.argtypes  = [c_void_p, c_int32]
    lib.Open_Dev.restype   = c_int32

    lib.Close_Dev.argtypes = [c_void_p, c_int32]
    lib.Close_Dev.restype  = c_int32

    lib.Write_FMemBMPPath.argtypes = [c_uint8, c_char_p, c_uint32]
    lib.Write_FMemBMPPath.restype  = c_int32

    lib.Write_FMemArray.argtypes = [c_uint8, c_void_p, c_int32,
                                    c_uint32, c_uint32, c_uint32]
    lib.Write_FMemArray.restype  = c_int32

    lib.Change_DispSlot.argtypes = [c_uint8, c_uint32]
    lib.Change_DispSlot.restype  = c_int32

    lib.Check_HeadSerial.argtypes = [c_uint8, c_char_p, c_int32]
    lib.Check_HeadSerial.restype  = c_int32

    return lib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_mask_for_slm(image_path):
    """
    Load a BMP phase mask and return a 2D uint8 numpy array (1024 x 1272).
    Crops from 1280 px wide to 1272 px if necessary.
    """
    resX, resY = 1272, 1024
    gray = np.array(Image.open(image_path).convert('L'))

    if gray.shape == (resY, 1280):
        gray = gray[:, 4:-4]

    if gray.shape != (resY, resX):
        raise ValueError(
            f"Mask shape must be {resY}x{resX}. Got {gray.shape}."
        )
    return gray.astype(np.uint8)


# ---------------------------------------------------------------------------
# SLMController  –  USB SDK controller
# ---------------------------------------------------------------------------

class SLMController:
    """
    Controls the Hamamatsu LCOS-SLM via hpkSLMdaLV.dll (USB SDK).

    Workflow:
        1. Open_Dev  – connect to SLM over USB
        2. Write_FMemBMPPath / Write_FMemArray – upload image to flash slot
        3. Change_DispSlot – switch display to that slot
        4. Close_Dev – disconnect

    Usage:
        with SLMController() as ctrl:
            ctrl.display_bmp(r"path\\to\\mask.bmp")
            ctrl.display_grayscale(gray_2d_array)
    """

    SLM_X = 1272
    SLM_Y = 1024

    def __init__(self):
        self._lib      = _load_dll()
        self._dev_id   = SLM_DEVICE_ID
        self._slot     = SLM_SLOT_NUMBER
        self._id_list  = (c_uint8 * 1)(self._dev_id)  # single-device list
        self._opened   = False
        self._connect()

    def _connect(self):
        ret = self._lib.Open_Dev(self._id_list, c_int32(1))
        # Open_Dev returns the number of devices opened (1 = success for single device),
        # or a negative/zero value on error.
        if ret < 1:
            raise RuntimeError(f"Open_Dev failed with code {ret}. "
                               "Check USB connection and that no other app "
                               "(e.g. USB_SLMControl.exe) has the SLM open.")
        self._opened = True
        # Read serial number to confirm connection
        buf = create_string_buffer(64)
        self._lib.Check_HeadSerial(c_uint8(self._id_list[0]), buf, c_int32(64))
        serial = buf.value.decode(errors='replace')
        print(f"[SLM] Connected. Device index={self._id_list[0]}  "
              f"Serial={serial!r}  Slot={self._slot}")

    def display_bmp(self, bmp_path: str):
        """
        Upload a BMP file (1272x1024 8-bit grayscale) to flash slot and
        immediately switch the SLM display to show it.
        """
        path_bytes = os.path.abspath(bmp_path).encode('ascii') + b'\x00'
        path_buf   = create_string_buffer(path_bytes)

        ret = self._lib.Write_FMemBMPPath(
            c_uint8(self._id_list[0]),
            path_buf,
            c_uint32(self._slot),
        )
        if ret < 0:
            raise RuntimeError(f"Write_FMemBMPPath failed with code {ret} "
                               f"for {bmp_path!r}")

        ret = self._lib.Change_DispSlot(
            c_uint8(self._id_list[0]),
            c_uint32(self._slot),
        )
        if ret < 0:
            raise RuntimeError(f"Change_DispSlot failed with code {ret}")

    def display_grayscale(self, gray_2d: np.ndarray):
        """
        Send a (1024, 1272) uint8 grayscale array to the SLM.
        """
        if gray_2d.shape != (self.SLM_Y, self.SLM_X):
            raise ValueError(
                f"Expected ({self.SLM_Y}, {self.SLM_X}), got {gray_2d.shape}."
            )
        flat = np.ascontiguousarray(gray_2d.ravel(), dtype=np.uint8)
        n    = self.SLM_X * self.SLM_Y

        # Keep 'flat' alive until after the DLL call (GC safety)
        ret = self._lib.Write_FMemArray(
            c_uint8(self._id_list[0]),
            flat.ctypes.data_as(c_void_p),
            c_int32(n),
            c_uint32(self.SLM_X),
            c_uint32(self.SLM_Y),
            c_uint32(self._slot),
        )
        if ret < 0:
            raise RuntimeError(f"Write_FMemArray failed with code {ret}")

        ret = self._lib.Change_DispSlot(
            c_uint8(self._id_list[0]),
            c_uint32(self._slot),
        )
        if ret < 0:
            raise RuntimeError(f"Change_DispSlot failed with code {ret}")

        # Explicit reference to keep numpy buffer alive past DLL return
        del flat

    def display_phase_tensor(self, phase_tensor):
        """
        Convert a [1,1,H,W] float phase tensor (values in [0, 2pi]) to SLM
        grayscale format and display it.

        Applies the per-pixel calibration mask (additive mod 2π) before
        gray-level conversion — matching the Mathematica export pipeline:
            fazeSoucetRad = Mod[hologram + calibration, 2π]
        """
        phase_np = phase_tensor.detach().cpu().squeeze().numpy()

        # Crop to 1272 width (active SLM area) BEFORE adding calibration
        if phase_np.shape[1] == 1280:
            phase_np = phase_np[:, 4:-4]

        # Additive calibration correction (Mathematica: Mod[holo + cal, 2π])
        cal_rad = _load_calibration_mask()
        phase_corrected = np.mod(phase_np + cal_rad, 2.0 * math.pi)

        gray = np.clip(
            phase_corrected / (2.0 * math.pi) * SLM_GRAY_2PI, 0, SLM_GRAY_2PI
        ).astype(np.uint8)
        self.display_grayscale(gray)

    def close(self):
        if self._opened:
            self._lib.Close_Dev(self._id_list, c_int32(1))
            self._opened = False
            print("[SLM] Disconnected.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob, time
    masks = sorted(glob.glob(
        r"C:\Users\Administrator\Desktop\Krystof\Program PYTHON SLM"
        r"\Hamamatsu\Phase Masks\*.bmp"
    ))
    if not masks:
        print("No masks found.")
    else:
        with SLMController() as ctrl:
            for path in masks:
                print(f"Displaying {os.path.basename(path)} ...")
                ctrl.display_bmp(path)
                time.sleep(2.0)
        print("Done.")

