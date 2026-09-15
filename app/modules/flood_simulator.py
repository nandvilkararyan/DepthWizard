"""Fast, raster-based flood inundation simulation for calibrated DSMs."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import label


@dataclass
class FloodResult:
    inundation_mask: np.ndarray
    water_depth: np.ndarray
    flooded_area_km2: float
    water_volume_m3: float
    pixel_area_m2: float
    water_level_meters: float
    overlay_png: bytes


def _connected_border_component(mask: np.ndarray) -> np.ndarray:
    """Keep only low terrain connected to the raster boundary (8-connectivity)."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    border_labels = np.unique(np.concatenate((
        components[0, :], components[-1, :], components[:, 0], components[:, -1]
    )))
    border_labels = border_labels[border_labels != 0]
    return np.isin(components, border_labels)


def _pixel_area_m2(pixel_size: Optional[Tuple[float, float]], latitude: Optional[float]) -> float:
    if not pixel_size:
        return 1.0
    x_size, y_size = abs(float(pixel_size[0])), abs(float(pixel_size[1]))
    if latitude is not None:
        # Approximate WGS84 metres per degree at the raster centre.
        lat = np.deg2rad(float(latitude))
        metres_lat = 111132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat)
        metres_lon = 111412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat)
        return max(x_size * metres_lon * y_size * metres_lat, 1e-6)
    return max(x_size * y_size, 1e-6)


def simulate_flood(
    metric_dsm: np.ndarray,
    water_level_meters: float,
    pixel_size: Optional[Tuple[float, float]] = None,
    latitude: Optional[float] = None,
) -> FloodResult:
    """Simulate border-connected inundation and encode an 8-bit depth overlay."""
    dsm = np.asarray(metric_dsm, dtype=np.float32)
    if dsm.ndim != 2:
        raise ValueError("metric_dsm must be a 2D array")
    finite = np.isfinite(dsm)
    below_level = finite & (dsm <= float(water_level_meters))
    connected = _connected_border_component(below_level)
    depth = np.where(connected, np.maximum(float(water_level_meters) - dsm, 0.0), 0.0).astype(np.float32)
    area = _pixel_area_m2(pixel_size, latitude)
    flooded_area_km2 = float(np.count_nonzero(connected) * area / 1_000_000.0)
    volume_m3 = float(np.sum(depth, dtype=np.float64) * area)

    max_depth = float(np.max(depth)) if np.any(depth) else 0.0
    overlay = np.zeros(depth.shape, dtype=np.uint8)
    if max_depth > 0:
        overlay = np.clip(depth / max_depth * 255.0, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Failed to encode flood overlay PNG")

    return FloodResult(
        inundation_mask=connected,
        water_depth=depth,
        flooded_area_km2=flooded_area_km2,
        water_volume_m3=volume_m3,
        pixel_area_m2=area,
        water_level_meters=float(water_level_meters),
        overlay_png=encoded.tobytes(),
    )
