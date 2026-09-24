import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR = BASE_DIR / "frontend"

# Model Identifiers and Registry
DEFAULT_MODEL_ID = str(BASE_DIR / "app" / "depthwizard_model")
DEFAULT_MODEL_KEY = "depthwizard_finetuned"

SUPPORTED_MODELS = {
    "depth_anything_v2_small": {
        "key": "depth_anything_v2_small",
        "id": "depth-anything/Depth-Anything-V2-Small-hf",
        "name": "Depth Anything V2 - Small (HF)",
        "variant": "Small",
        "params": "24.8M",
        "size_mb": 99,
        "is_local": False,
        "description": "Fast lightweight model (~25M parameters) optimized for rapid inference and quick terrain previews."
    },
    "depth_anything_v2_base": {
        "key": "depth_anything_v2_base",
        "id": "depth-anything/Depth-Anything-V2-Base-hf",
        "name": "Depth Anything V2 - Base (HF)",
        "variant": "Base",
        "params": "97.5M",
        "size_mb": 390,
        "is_local": False,
        "description": "Original 400MB base model (~97M parameters) with balanced depth fidelity across diverse landscapes."
    },
    "depth_anything_v2_large": {
        "key": "depth_anything_v2_large",
        "id": "depth-anything/Depth-Anything-V2-Large-hf",
        "name": "Depth Anything V2 - Large / High (HF)",
        "variant": "High / Large",
        "params": "335.3M",
        "size_mb": 1340,
        "is_local": False,
        "description": "High-capacity variant (~335M parameters, ~1.3GB) capturing intricate structural relief and crisp building ridges."
    },
    "depthwizard_finetuned": {
        "key": "depthwizard_finetuned",
        "id": DEFAULT_MODEL_ID,
        "name": "DepthWizard Fine-Tuned (model.safetensors)",
        "variant": "Fine-Tuned Base",
        "params": "97.5M",
        "size_mb": 390,
        "is_local": True,
        "description": "Fine-tuned Depth Anything V2 model trained on aerial/satellite imagery with intelligent water-depth suppression."
    }
}

def resolve_model(model_key_or_id: str) -> dict:
    """Resolves a model key or HF identifier into its model configuration dictionary."""
    if not model_key_or_id:
        return SUPPORTED_MODELS[DEFAULT_MODEL_KEY]
    
    # Check if exact key match
    if model_key_or_id in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_key_or_id]
    
    # Check matching id or name
    for key, cfg in SUPPORTED_MODELS.items():
        if (
            cfg["id"] == model_key_or_id or
            cfg["name"].lower() == model_key_or_id.lower() or
            key.lower() == model_key_or_id.lower()
        ):
            return cfg
            
    # Also support friendly aliases like 'small', 'base', 'large', 'high', 'finetuned'
    low = model_key_or_id.lower()
    if "small" in low:
        return SUPPORTED_MODELS["depth_anything_v2_small"]
    elif "large" in low or "high" in low:
        return SUPPORTED_MODELS["depth_anything_v2_large"]
    elif "fine" in low or "safetensor" in low:
        return SUPPORTED_MODELS["depthwizard_finetuned"]
    elif "base" in low:
        return SUPPORTED_MODELS["depth_anything_v2_base"]
        
    # Fallback to direct model string as a custom HF/local model id
    return {
        "key": "custom",
        "id": model_key_or_id,
        "name": f"Custom ({model_key_or_id})",
        "variant": "Custom",
        "params": "Unknown",
        "size_mb": 0,
        "is_local": False,
        "description": f"Custom model identifier: {model_key_or_id}"
    }

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
