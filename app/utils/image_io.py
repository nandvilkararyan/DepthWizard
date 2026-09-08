import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any
import numpy as np
import cv2
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


@dataclass
class GeoMetadata:
    """Dataclass holding geospatial raster metadata extracted via Rasterio."""
    is_georeferenced: bool
    crs: Optional[str] = None
    transform: Optional[Tuple[float, float, float, float, float, float]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    width: int = 0
    height: int = 0
    nodata: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_georeferenced": self.is_georeferenced,
            "crs": self.crs,
            "transform": self.transform,
            "bounds": self.bounds,
            "width": self.width,
            "height": self.height,
            "nodata": self.nodata
        }


def load_optical_image(filepath: str) -> Tuple[np.ndarray, GeoMetadata]:
    """
    Reads an optical image (PNG, JPG, or GeoTIFF) and returns an 8-bit RGB numpy array
    along with its geospatial metadata (if available).

    Args:
        filepath: Path to the input image file.

    Returns:
        Tuple containing:
            - rgb_image: np.ndarray of shape (H, W, 3) in uint8 format (RGB)
            - geo_meta: GeoMetadata object containing CRS and affine transformation if present.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input image file not found: {filepath}")

    geo_meta = GeoMetadata(is_georeferenced=False)

    # Attempt to open with Rasterio to inspect geospatial metadata
    try:
        with rasterio.open(filepath) as src:
            has_crs = src.crs is not None and bool(src.crs.to_string())
            
            geo_meta = GeoMetadata(
                is_georeferenced=has_crs,
                crs=src.crs.to_string() if has_crs else None,
                transform=tuple(src.transform)[:6] if has_crs else None,
                bounds=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top) if has_crs else None,
                width=src.width,
                height=src.height,
                nodata=src.nodata
            )

            # Read image bands
            bands = src.read()  # Shape: (C, H, W)
            count = src.count

            if count >= 3:
                # Extract top 3 channels as RGB
                img_data = bands[:3, :, :]  # (3, H, W)
                img_data = np.transpose(img_data, (1, 2, 0))  # (H, W, 3)
            elif count == 1:
                # Single band grayscale image, duplicate to 3 channels
                img_single = bands[0, :, :]
                img_data = np.stack([img_single] * 3, axis=-1)
            else:
                img_data = np.transpose(bands, (1, 2, 0))

            # Normalize data type to uint8 [0, 255]
            if img_data.dtype == np.uint8:
                rgb_image = img_data
            elif img_data.dtype == np.uint16:
                rgb_image = (img_data / 255.0).clip(0, 255).astype(np.uint8)
            elif np.issubdtype(img_data.dtype, np.floating):
                # Float normalization
                valid_mask = np.isfinite(img_data)
                if np.any(valid_mask):
                    min_val = np.nanmin(img_data[valid_mask])
                    max_val = np.nanmax(img_data[valid_mask])
                    rng = max_val - min_val if max_val > min_val else 1.0
                    rgb_image = (((img_data - min_val) / rng) * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    rgb_image = np.zeros_like(img_data, dtype=np.uint8)
            else:
                rgb_image = img_data.astype(np.uint8)

            return rgb_image, geo_meta

    except Exception:
        # Fallback to OpenCV / PIL for standard non-geospatial formats (PNG, JPG)
        bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if bgr is None:
            # Fallback to PIL
            pil_img = Image.open(filepath).convert("RGB")
            rgb_image = np.array(pil_img, dtype=np.uint8)
        else:
            rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        geo_meta = GeoMetadata(
            is_georeferenced=False,
            width=rgb_image.shape[1],
            height=rgb_image.shape[0]
        )

        return rgb_image, geo_meta


def load_reference_dem(dem_filepath: str) -> Tuple[np.ndarray, GeoMetadata]:
    """
    Loads a reference digital elevation model (DEM) GeoTIFF file.

    Args:
        dem_filepath: Path to reference DEM GeoTIFF file.

    Returns:
        Tuple containing:
            - dem_data: np.ndarray (H, W) float32 elevation values
            - dem_meta: GeoMetadata object containing CRS and affine transformation.
    """
    if not os.path.exists(dem_filepath):
        raise FileNotFoundError(f"Reference DEM file not found: {dem_filepath}")

    with rasterio.open(dem_filepath) as src:
        has_crs = src.crs is not None and bool(src.crs.to_string())
        dem_data = src.read(1).astype(np.float32)

        dem_meta = GeoMetadata(
            is_georeferenced=has_crs,
            crs=src.crs.to_string() if has_crs else None,
            transform=tuple(src.transform)[:6] if has_crs else None,
            bounds=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top) if has_crs else None,
            width=src.width,
            height=src.height,
            nodata=src.nodata
        )

        return dem_data, dem_meta
