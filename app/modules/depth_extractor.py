import math
from typing import Optional, Tuple
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
            mode="bicubic",
            align_corners=False,
        )

        depth_np = prediction.squeeze().cpu().numpy().astype(np.float32)
        return depth_np

    def extract_depth(
        self,
        rgb_image: np.ndarray,
        tile_size: int = 512,
        overlap_ratio: float = 0.20
    ) -> np.ndarray:
        """
        Runs depth extraction using tiled sliding-window inference with 2D Hann blending
        for large satellite tiles.

        Args:
            rgb_image: uint8 numpy array of shape (H, W, 3) in RGB format.
            tile_size: Height/width of square sliding window tiles.
            overlap_ratio: Fraction of tile size to overlap (e.g. 0.20 = 20% overlap).

        Returns:
            2D float32 numpy array of relative depth predictions of shape (H, W).
        """
        img_h, img_w = rgb_image.shape[:2]

        # If image fits within a single tile, run direct inference
        if img_h <= tile_size and img_w <= tile_size:
            return self.infer_single_image(rgb_image)

        # Calculate stride length based on tile size and overlap
        overlap_pixels = int(tile_size * overlap_ratio)
        stride = tile_size - overlap_pixels
        if stride <= 0:
            stride = tile_size // 2

        # Create tile origin coordinates
        y_starts = list(range(0, img_h - tile_size + 1, stride))
        if len(y_starts) == 0 or y_starts[-1] + tile_size < img_h:
            y_starts.append(max(0, img_h - tile_size))

        x_starts = list(range(0, img_w - tile_size + 1, stride))
        if len(x_starts) == 0 or x_starts[-1] + tile_size < img_w:
            x_starts.append(max(0, img_w - tile_size))

        # Remove duplicate starting positions if any
        y_starts = sorted(list(set(y_starts)))
        x_starts = sorted(list(set(x_starts)))

        # Initialize accumulation buffers
        depth_accum = np.zeros((img_h, img_w), dtype=np.float32)
        weight_accum = np.zeros((img_h, img_w), dtype=np.float32)

        hann_window = self.create_2d_hann_window(tile_size, tile_size)

        print(f"[DepthExtractor] Tiled processing: {len(y_starts)}x{len(x_starts)} grid, "
              f"tile_size={tile_size}, overlap={overlap_ratio:.0%}")

        for y in y_starts:
            for x in x_starts:
                y_end = min(y + tile_size, img_h)
                x_end = min(x + tile_size, img_w)
                tile_h = y_end - y
                tile_w = x_end - x

                tile_rgb = rgb_image[y:y_end, x:x_end, :]

                # Infer tile depth
                tile_depth = self.infer_single_image(tile_rgb)

                # Fetch matching window slice if tile was cropped at boundary
                if tile_h == tile_size and tile_w == tile_size:
                    win = hann_window
                else:
                    win = self.create_2d_hann_window(tile_h, tile_w)

                # Accumulate weighted predictions
                depth_accum[y:y_end, x:x_end] += tile_depth * win
                weight_accum[y:y_end, x:x_end] += win

        # Normalize by total accumulated window weights
        depth_blended = depth_accum / np.maximum(weight_accum, 1e-6)
        return depth_blended.astype(np.float32)
