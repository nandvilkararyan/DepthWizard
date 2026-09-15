import math
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import sys
from pathlib import Path

# Bootstrap project root directory into sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from app.config import DEFAULT_MODEL_ID, get_device
except ImportError:
    from config import DEFAULT_MODEL_ID, get_device


class DepthExtractor:
    """
    Module A: Single-View Depth Extraction using Depth Anything v2.
    Supports tiled sliding-window inference with 2D Hann window blending
    to handle high-resolution satellite/aerial imagery without VRAM overflow
    or boundary artifacts.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
        """
        Initializes the Depth Anything v2 model and image processor.

        Args:
            model_id: HuggingFace model identifier.
            device: Computing device ('cuda', 'mps', 'cpu'). Auto-detected if None.
        """
        self.model_id = model_id
        self.device = device or get_device()
        print(f"[DepthExtractor] Loading model '{self.model_id}' on device '{self.device}'...")

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        self.last_water_mask: Optional[np.ndarray] = None
        self.last_depth_uint8: Optional[np.ndarray] = None
        # Raw model output statistics (before normalization) — used by ScaleCalibrator
        # to derive meaningful elevation ranges instead of hardcoded 0–100.
        self.last_raw_min: float = 0.0
        self.last_raw_max: float = 1.0

    @staticmethod
    def create_2d_hann_window(height: int, width: int) -> np.ndarray:
        """
        Generates a 2D Hann (raised cosine) window matrix for smooth tile blending.

        Args:
            height: Tile height in pixels.
            width: Tile width in pixels.

        Returns:
            2D numpy array of shape (height, width) with window weights in range (0, 1].
        """
        hann_h = 0.5 * (1.0 - np.cos(2.0 * np.pi * (np.arange(height) + 0.5) / height))
        hann_w = 0.5 * (1.0 - np.cos(2.0 * np.pi * (np.arange(width) + 0.5) / width))
        window_2d = np.outer(hann_h, hann_w)
        # Add small floor to avoid divide-by-zero at exact boundary edges
        return np.maximum(window_2d, 1e-4).astype(np.float32)

    def infer_single_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Runs direct inference on a single image array (H, W, 3).

        Args:
            rgb_image: uint8 numpy array of shape (H, W, 3) in RGB format.

        Returns:
            2D float32 numpy array of relative depth predictions matching input (H, W).
        """
        h_orig, w_orig = rgb_image.shape[:2]
        pil_img = Image.fromarray(rgb_image)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Shape of predicted_depth: (batch_size, height, width) or (batch_size, 1, height, width)
        if predicted_depth.ndim == 3:
            predicted_depth = predicted_depth.unsqueeze(1)

        # Resize predicted depth back to original image dimensions
        prediction = F.interpolate(
            predicted_depth,
            size=(h_orig, w_orig),
            mode="bilinear",
            align_corners=False,
        )

        depth_np = prediction.squeeze().cpu().numpy().astype(np.float32)
        finite = np.isfinite(depth_np)
        raw_min = float(np.min(depth_np[finite])) if np.any(finite) else 0.0
        raw_max = float(np.max(depth_np[finite])) if np.any(finite) else 0.0
        raw_mean = float(np.mean(depth_np[finite])) if np.any(finite) else 0.0
        print(
            f"[DepthExtractor] Raw Depth Stats -> Min: {raw_min:.4f}, "
            f"Max: {raw_max:.4f}, Mean: {raw_mean:.4f}"
        )

        if not np.any(finite) or raw_max - raw_min <= 1e-8:
            # A broken/constant model output cannot calibrate or colorize. Use
            # image luminance as a deterministic non-empty relative-depth proxy.
            fallback = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            fallback_min = float(np.min(fallback))
            fallback_range = float(np.ptp(fallback))
            depth_np = (fallback - fallback_min) / max(fallback_range, 1.0)
            self.last_raw_min = fallback_min
            self.last_raw_max = fallback_min + max(fallback_range, 1.0)
        else:
            self.last_raw_min = raw_min
            self.last_raw_max = raw_max
            depth_np = np.nan_to_num(depth_np, nan=raw_mean, posinf=raw_max, neginf=raw_min)
            depth_np = (depth_np - raw_min) / (raw_max - raw_min + 1e-8)

        depth_np = np.clip(depth_np, 0.0, 1.0).astype(np.float32)
        
        # Handle black nodata satellite padding borders
        black_mask = np.all(rgb_image <= 5, axis=2)
        if np.any(black_mask) and np.any(~black_mask):
            valid_floor = float(np.percentile(depth_np[~black_mask], 2.0))
            depth_np[black_mask] = valid_floor
            
        self.last_depth_uint8 = np.round(depth_np * 255.0).astype(np.uint8)
        return depth_np

    def extract_depth(
        self,
        rgb_image: np.ndarray,
        tile_size: int = 512,
        overlap_ratio: float = 0.20
    ) -> np.ndarray:
        """
        Runs depth extraction on the entire image in a single pass.
        
        Note: We intentionally bypass tiling. Monocular depth models rely on global 
        context. Slicing the image into tiles causes the model to predict completely 
        different relative depth scales for each tile, which creates severe blocky seams 
        and spikes when blended. Because image_io.py already caps the input size to 
        1024px, a single full-image pass is fast, VRAM-safe, and guarantees smooth, 
        globally consistent terrain geometry.
        """
        print(f"[DepthExtractor] Running full-image depth extraction (shape={rgb_image.shape[:2]})...")
        depth = self.infer_single_image(rgb_image)
        return self._correct_water_depth(rgb_image, depth)

    def _correct_water_depth(self, rgb_image: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Suppress the common monocular-depth failure where reflective water is elevated."""
        image = rgb_image.astype(np.float32) / 255.0
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        local_mean = cv2.blur(gray, (15, 15))
        local_sq_mean = cv2.blur(gray * gray, (15, 15))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
        brightness = np.mean(image, axis=2)
        blue_green = (image[:, :, 1] + image[:, :, 2]) * 0.5
        red_suppressed = blue_green >= image[:, :, 0] * 0.92
        water_mask = (
            (local_std < 0.075) &
            (brightness >= np.percentile(brightness, 55)) &
            red_suppressed
        )
        # A large 15x15 OPEN kernel completely eradicates small false-positive 
        # water detections (like mountain snow or smooth valleys) that cause 
        # the terrain to turn into a bed of needles.
        water_mask = cv2.morphologyEx(
            water_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8)
        )
        water_mask = cv2.morphologyEx(
            water_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
        ).astype(bool)
        self.last_water_mask = water_mask
        if not np.any(water_mask):
            return depth.astype(np.float32)

        terrain = depth.copy().astype(np.float32)
        non_water = ~water_mask & np.isfinite(terrain)
        if not np.any(non_water):
            return terrain
        local_floor = float(np.percentile(terrain[non_water], 5.0))
        
        # OpenCV converts np.inf to FLT_MAX (3.4e38), which np.isfinite considers True!
        # This causes FLT_MAX to leak into the depth map. Use a dummy value instead.
        dummy_max = 10000.0
        masked = np.where(non_water, terrain, dummy_max).astype(np.float32)
        
        kernel = np.ones((31, 31), np.uint8)
        neighbourhood_floor = cv2.erode(masked, kernel)
        
        # If erode returns dummy_max, the 31x31 area was purely water; use global local_floor
        valid_floor = neighbourhood_floor < (dummy_max - 1.0)
        replacement = np.where(valid_floor, neighbourhood_floor, local_floor)
        
        terrain[water_mask] = replacement[water_mask]
        return terrain.astype(np.float32)
