# ──────────────────────────────────────────────────────────────────────────────
# File: extract_camera_data.py  Project: DNN-SLM Phase Mask Generation
# Purpose: CinCam beam-profiler frame capture via XML-RPC
# ──────────────────────────────────────────────────────────────────────────────
import xmlrpc.client
import numpy as np
import matplotlib.pyplot as plt
from config import CAMERA_XMLRPC_URL

# Connect to the RayCi server
try:
    proxy = xmlrpc.client.ServerProxy(CAMERA_XMLRPC_URL)
    rayci = proxy.RayCi
except Exception as e:
    print(f"Failed to connect to RayCi server: {e}")

def get_single_frame():
    """
    Capture one frame from CinCam, auto-scale to 8-bit.
    Returns (uint8 HxW array, raw_max) or (None, None) on error.
    """
    try:
        width = rayci.LiveMode.Data.getSizeX(0)
        height = rayci.LiveMode.Data.getSizeY(0)
        raw_binary_data = rayci.LiveMode.Data.getFloatField(0)
        byte_data = raw_binary_data.data if isinstance(raw_binary_data, xmlrpc.client.Binary) else raw_binary_data
        intensity_array_float = np.frombuffer(byte_data, dtype=np.float32)
        
        frame_max = np.max(intensity_array_float)
        if frame_max > 0.0:
            intensity_array_scaled = (intensity_array_float / frame_max) * 255.0
        else:
            intensity_array_scaled = intensity_array_float
        
        intensity_array_8bit = np.clip(intensity_array_scaled, 0, 255).astype(np.uint8)
        intensity_matrix = intensity_array_8bit.reshape((height, width))
        
        return intensity_matrix, frame_max
        
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None, None

if __name__ == "__main__":
    print("Waiting for external trigger...")
    
    # ... Your external trigger happens here ...
    print("Trigger received! Extracting and auto-scaling frame...")
    
    beam_profile, original_max = get_single_frame()
    
    if beam_profile is not None:
        scaled_peak = np.max(beam_profile)
        print(f"Success! Captured {beam_profile.shape[1]}x{beam_profile.shape[0]} frame.")
        print(f"Original Raw Max: {original_max:.4f} -> Scaled Max: {scaled_peak}/255")
        
        # --- Visualization ---
        plt.figure(figsize=(8, 6))
        
        # Display the matrix. vmin=0 and vmax=255 strictly enforce the 8-bit grayscale mapping.
        im = plt.imshow(beam_profile, cmap='gray', vmin=0, vmax=255)
        
        cbar = plt.colorbar(im)
        cbar.set_label('Pixel Intensity (Bit Value)')
        
        plt.title('CinCam Laser Beam Profile (Auto-Scaled)')
        plt.xlabel('Sensor X-Axis (Pixels)')
        plt.ylabel('Sensor Y-Axis (Pixels)')
        
        plt.tight_layout()
        plt.show()
