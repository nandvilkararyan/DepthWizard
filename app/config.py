import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR = BASE_DIR / "frontend"

# Default Core Model Identifier
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"

# Inference parameters
DEFAULT_TILE_SIZE = 512
DEFAULT_OVERLAP_RATIO = 0.20

# Metric calibration defaults for non-georeferenced images
DEFAULT_MIN_ELEVATION_METERS = 0.0
DEFAULT_MAX_ELEVATION_METERS = 50.0

# Hardware device auto-detection
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
