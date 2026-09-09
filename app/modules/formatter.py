import json
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

import sys
from pathlib import Path

# Bootstrap project root directory into sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from app.utils.image_io import GeoMetadata
    from app.modules.scale_calibrator import CalibrationResult
except ImportError:
    from utils.image_io import GeoMetadata
    from modules.scale_calibrator import CalibrationResult


class OutputFormatter:
    """
    Module C: Output Formatting for Unity 3D & GIS Workflows.
    Handles export of:
      1. 16-bit single-channel PNG for Unity heightmap displacement.
      2. 32-bit floating-point GeoTIFF with original CRS geospatial metadata.
      3. Scene-level calibration and geometry metadata JSON.
      4. Optical texture PNG visualization.
    """

    @staticmethod
    def export_unity_16bit_png(
        metric_dsm: np.ndarray,
        output_filepath: str
    ) -> Tuple[str, float, float]:
        """
        Quantizes metric DSM float values to a 16-bit uint PNG (0 to 65535)
        specifically formatted for Unity terrain heightmap displacement.

        Args:
            metric_dsm: 2D float32 array of metric elevation values (meters).
            output_filepath: Output path for the PNG file.

        Returns:
            Tuple of (output_filepath, min_elev, max_elev).
        """
        min_elev = float(np.nanmin(metric_dsm))
        max_elev = float(np.nanmax(metric_dsm))
        elev_range = max_elev - min_elev if max_elev > min_elev else 1.0

        # Normalize to [0, 1] then scale to uint16 [0, 65535]
        norm_dsm = (metric_dsm - min_elev) / elev_range
        uint16_dsm = np.clip(norm_dsm * 65535.0, 0, 65535).astype(np.uint16)

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        cv2.imwrite(output_filepath, uint16_dsm)

        return output_filepath, min_elev, max_elev

    @staticmethod
    def export_geotiff(
        metric_dsm: np.ndarray,
        geo_meta: GeoMetadata,
        output_filepath: str
    ) -> str:
        """
        Exports a 32-bit floating-point GeoTIFF preserving CRS and Affine transform.

        Args:
            metric_dsm: 2D float32 array of metric elevation values (meters).
            geo_meta: GeoMetadata object containing CRS and affine transformation.
            output_filepath: Output file path for the .tif file.

        Returns:
            Output file path string.
        """
        height, width = metric_dsm.shape[:2]
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        if geo_meta.is_georeferenced and geo_meta.crs and geo_meta.transform:
            crs_obj = CRS.from_string(geo_meta.crs)
            transform_obj = Affine(*geo_meta.transform)
        else:
            # Default pixel identity coordinate reference
            crs_obj = None
            transform_obj = Affine.identity()

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": crs_obj,
            "transform": transform_obj,
            "nodata": -9999.0,
            "compress": "lzw"
        }

        dsm_clean = np.where(np.isfinite(metric_dsm), metric_dsm, -9999.0).astype(np.float32)

        with rasterio.open(output_filepath, "w", **profile) as dst:
            dst.write(dsm_clean, 1)

        return output_filepath

    @staticmethod
    def export_optical_texture(
        rgb_image: np.ndarray,
        output_filepath: str
    ) -> str:
        """
        Exports original optical image as standard 8-bit RGB PNG for Unity material texturing.
        """
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_filepath, bgr_image)
        return output_filepath

    @staticmethod
    def export_preview_8bit_png(
        metric_dsm: np.ndarray,
        output_filepath: str
    ) -> str:
        """
        Exports an 8-bit grayscale heightmap PNG (0-255) for browser-based 3D terrain viewers
        (e.g. Three.js DisplacementMap). Normalized linearly from the DSM's min-max range.

        Args:
            metric_dsm: 2D float32 array of metric elevation values (meters).
            output_filepath: Output path for the PNG file.

        Returns:
            Output file path string.
        """
        min_elev = float(np.nanmin(metric_dsm))
        max_elev = float(np.nanmax(metric_dsm))
        elev_range = max_elev - min_elev if max_elev > min_elev else 1.0

        norm_dsm = (metric_dsm - min_elev) / elev_range
        uint8_dsm = np.clip(norm_dsm * 255.0, 0, 255).astype(np.uint8)

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        cv2.imwrite(output_filepath, uint8_dsm)
        return output_filepath

    @staticmethod
    def export_colorized_depth_preview(
        metric_dsm: np.ndarray,
        output_filepath: str
    ) -> str:
        """
        Exports a false-color (COLORMAP_TURBO) 8-bit RGB PNG of the DSM for 2D preview
        in the web frontend results panel.

        Args:
            metric_dsm: 2D float32 array of metric elevation values.
            output_filepath: Output path for the PNG file.

        Returns:
            Output file path string.
        """
        min_elev = float(np.nanmin(metric_dsm))
        max_elev = float(np.nanmax(metric_dsm))
        elev_range = max_elev - min_elev if max_elev > min_elev else 1.0

        norm_dsm = (metric_dsm - min_elev) / elev_range
        uint8_dsm = np.clip(norm_dsm * 255.0, 0, 255).astype(np.uint8)
        colorized = cv2.applyColorMap(uint8_dsm, cv2.COLORMAP_TURBO)

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        cv2.imwrite(output_filepath, colorized)
        return output_filepath

    @staticmethod
    def generate_metadata_json(
        calib_result: CalibrationResult,
        geo_meta: GeoMetadata,
        img_shape: Tuple[int, int],
        output_filepath: str
    ) -> Dict[str, Any]:
        """
        Calculates and exports scene-level metadata JSON.

        Args:
            calib_result: CalibrationResult object from Module B.
            geo_meta: GeoMetadata object.
            img_shape: Tuple of (height, width).
            output_filepath: Target output filepath.

        Returns:
            Dictionary containing the full JSON payload.
        """
        height, width = img_shape
        aspect_ratio = float(width) / float(height) if height > 0 else 1.0

        suggested_disp_scale = float(np.clip(0.4 + (calib_result.elevation_range / 300.0) * 0.8, 0.4, 2.2))

        metadata = {
            "scene_geometry": {
                "width_pixels": width,
                "height_pixels": height,
                "aspect_ratio": round(aspect_ratio, 4)
            },
            "elevation_metrics": {
                "min_elevation_meters": round(calib_result.min_elevation, 4),
                "max_elevation_meters": round(calib_result.max_elevation, 4),
                "elevation_range_meters": round(calib_result.elevation_range, 4),
                "suggested_disp_scale": round(suggested_disp_scale, 4)
            },
            "calibration_parameters": {
                "scale_alpha": float(calib_result.alpha),
                "shift_beta_meters": float(calib_result.beta),
                "calibration_type": calib_result.calibration_type,
                "rmse_meters": round(calib_result.rmse, 4) if calib_result.rmse is not None else None,
                "mae_meters": round(calib_result.mae, 4) if calib_result.mae is not None else None
            },
            "geospatial_metadata": {
                "is_georeferenced": geo_meta.is_georeferenced,
                "crs": geo_meta.crs,
                "bounds": geo_meta.bounds,
                "transform": geo_meta.transform
            }
        }

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
