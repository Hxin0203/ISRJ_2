"""Fixed factor/device assignments; no command-line override of server identity."""
import os

FACTOR = os.environ.get("ISRJ2_FACTOR")
if FACTOR not in ("dilation", "frequency"):
    raise RuntimeError("ISRJ2_FACTOR must explicitly select dilation or frequency")
GPU_INDEX = "1" if FACTOR == "dilation" else "0"
GPU_UUID = ("GPU-bb60dd3a-738e-2398-908f-7f77ac05549b" if GPU_INDEX == "1"
            else "GPU-f12e73b1-3bed-5a1e-bcb0-f9726dd1621d")
DEPENDENCY = ("unet_mf_tail_screen_20261117_v2" if FACTOR == "dilation"
              else "unet_noise_norm_screen_20261117_v1")
RUN = f"unet_{FACTOR}_screen_20261117_v1"
