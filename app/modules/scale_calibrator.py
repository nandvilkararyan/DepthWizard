from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
from scipy.ndimage import zoom

import sys
from pathlib import Path

# Bootstrap project root directory into sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from app.utils.image_io import GeoMetadata
except ImportError:
    from utils.image_io import GeoMetadata


@dataclass
class CalibrationResult:
    """Dataclass storing the output of metric scale & shift calibration."""
    metric_dsm: np.ndarray          # 2D float32 array of metric DSM elevation values in meters
    alpha: float                     # Linear scale factor alpha
    beta: float                      # Linear shift factor beta (meters)
    min_elevation: float             # Minimum elevation in scene (meters)
    max_elevation: float             # Maximum elevation in scene (meters)
    elevation_range: float           # Elevation spread (meters)
    rmse: Optional[float] = None     # Root Mean Squared Error against GT reference (if provided)
    mae: Optional[float] = None      # Mean Absolute Error against GT reference (if provided)
    calibration_type: str = "relative_fallback"  # 'reference_dem', 'reference_points', or 'relative_fallback'


class ScaleCalibrator:
    """
    Module B: Scale & Shift Calibration for Metric Elevation Mapping.
    Aligns single-view relative depth predictions to absolute metric height values
    using linear regression against low-resolution reference DEMs (e.g., SRTM 30m)
    or LiDAR ground truth points.
    """

    def __init__(self, default_min_alt: float = 0.0, default_max_alt: float = 50.0):
        """
        Args:
            default_min_alt: Fallback minimum elevation in meters for uncalibrated images.
            default_max_alt: Fallback maximum elevation in meters for uncalibrated images.
        """
        self.default_min_alt = default_min_alt
        self.default_max_alt = default_max_alt

    def calibrate_depth(
        self,
        predicted_depth: np.ndarray,
        geo_meta: GeoMetadata,
        reference_dem: Optional[np.ndarray] = None,
        reference_dem_meta: Optional[GeoMetadata] = None,
        reference_points: Optional[List[Tuple[float, float, float]]] = None,  # List of (row, col, z_meters)
        min_alt_override: Optional[float] = None,
        max_alt_override: Optional[float] = None
    ) -> CalibrationResult:
        """
        Calibrates predicted relative depth map into metric elevation values (meters).

        Args:
            predicted_depth: 2D float32 relative depth map from Module A (H, W).
            geo_meta: Geospatial metadata of the optical image.
            reference_dem: Optional 2D float32 reference DEM raster (e.g. SRTM 30m).
            reference_dem_meta: Optional GeoMetadata of the reference DEM raster.
            reference_points: Optional list of (pixel_row, pixel_col, z_elevation_meters).
            min_alt_override: Custom minimum fallback altitude if non-georeferenced.
            max_alt_override: Custom maximum fallback altitude if non-georeferenced.

        Returns:
            CalibrationResult object containing metric DSM array, alpha, beta, and metrics.
        """
        h_pred, w_pred = predicted_depth.shape[:2]

        # Case 1: Georeferenced image with a reference raster DEM
        if reference_dem is not None and reference_dem.ndim == 2:
            return self._calibrate_with_reference_dem(
                predicted_depth=predicted_depth,
                reference_dem=reference_dem,
                reference_dem_meta=reference_dem_meta
            )

        # Case 2: Reference LiDAR point set provided (pixel_row, pixel_col, elevation)
        if reference_points is not None and len(reference_points) >= 2:
            return self._calibrate_with_reference_points(
                predicted_depth=predicted_depth,
                reference_points=reference_points
            )

        # Case 3: Non-Georeferenced or Georeferenced without reference DEM ground truth
        # Fallback to linear mapping over target elevation range [min_alt, max_alt]
        min_alt = min_alt_override if min_alt_override is not None else self.default_min_alt
        max_alt = max_alt_override if max_alt_override is not None else self.default_max_alt

        return self._calibrate_relative_fallback(
            predicted_depth=predicted_depth,
            min_alt=min_alt,
            max_alt=max_alt
        )

    def _calibrate_with_reference_dem(
        self,
        predicted_depth: np.ndarray,
        reference_dem: np.ndarray,
        reference_dem_meta: Optional[GeoMetadata]
    ) -> CalibrationResult:
        """
        Solves for linear scale alpha and shift beta using least squares optimization
        against a co-registered low-resolution reference DEM.

        SCALE-FITTING LOGIC:
        -------------------
        Monocular depth estimation models predict relative depth values D_pred up to an
        unknown scale and shift factor. The true metric elevation H_metric relates to D_pred as:
            H_metric = alpha * D_pred + beta

        Given a reference DEM H_ref (e.g. SRTM 30m):
        1. Downsample/resample H_ref to match the exact spatial grid (H, W) of D_pred.
        2. Filter out invalid/nodata pixels in H_ref (e.g., values <= -9999 or NaNs).
        3. Formulate the overdetermined linear matrix equation:
               [ D_pred_valid   1 ] * [ alpha ] = [ H_ref_valid ]
                                      [ beta  ]
        4. Compute the least-squares solution using numpy.linalg.lstsq.
        5. Evaluate RMSE and MAE on the valid sample points.
        6. Apply (alpha, beta) across the full-resolution D_pred map.
        """
        h_pred, w_pred = predicted_depth.shape[:2]

        # Resample reference DEM to match predicted depth dimensions using bilinear interpolation
        h_ref, w_ref = reference_dem.shape[:2]
        if (h_ref, w_ref) != (h_pred, w_pred):
            dem_resampled = cv2.resize(
                reference_dem.astype(np.float32),
                (w_pred, h_pred),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            dem_resampled = reference_dem.astype(np.float32)

        # Create valid mask filtering out NoData values (-9999, NaNs, infs)
        nodata_val = reference_dem_meta.nodata if reference_dem_meta and reference_dem_meta.nodata is not None else -9999.0
        valid_mask = (
            np.isfinite(dem_resampled) &
            (dem_resampled > nodata_val + 1.0) &
            np.isfinite(predicted_depth)
        )

        num_valid = np.count_nonzero(valid_mask)
        if num_valid < 10:
            # Insufficient valid pixels in reference DEM, fallback to relative calibration
            return self._calibrate_relative_fallback(
                predicted_depth,
                self.default_min_alt,
                self.default_max_alt
            )

        # Extract valid pairs
        d_valid = predicted_depth[valid_mask].astype(np.float64)
        h_valid = dem_resampled[valid_mask].astype(np.float64)

        # Build design matrix A = [d_valid, 1]
        A = np.vstack([d_valid, np.ones_like(d_valid)]).T

        # Solve for [alpha, beta] using Ordinary Least Squares (OLS)
        solution, residuals, rank, s = np.linalg.lstsq(A, h_valid, rcond=None)
        alpha, beta = float(solution[0]), float(solution[1])

        # Compute metric DSM for full map
        metric_dsm = (alpha * predicted_depth + beta).astype(np.float32)

        # Calculate error metrics against reference DEM
        h_fitted = alpha * d_valid + beta
        errors = h_valid - h_fitted
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        min_elev = float(np.nanmin(metric_dsm))
        max_elev = float(np.nanmax(metric_dsm))

        return CalibrationResult(
            metric_dsm=metric_dsm,
            alpha=alpha,
            beta=beta,
            min_elevation=min_elev,
            max_elevation=max_elev,
            elevation_range=max_elev - min_elev,
            rmse=rmse,
            mae=mae,
            calibration_type="reference_dem"
        )

    def _calibrate_with_reference_points(
        self,
        predicted_depth: np.ndarray,
        reference_points: List[Tuple[float, float, float]]
    ) -> CalibrationResult:
        """
        Fits alpha and beta using discrete reference ground control points (row, col, elevation).
        """
        h_pred, w_pred = predicted_depth.shape[:2]
        d_samples = []
        h_samples = []

        for r, c, z in reference_points:
            r_idx = int(clip_val(round(r), 0, h_pred - 1))
            c_idx = int(clip_val(round(c), 0, w_pred - 1))
            d_val = float(predicted_depth[r_idx, c_idx])
            if np.isfinite(d_val) and np.isfinite(z):
                d_samples.append(d_val)
                h_samples.append(float(z))

        if len(d_samples) < 2:
            return self._calibrate_relative_fallback(
                predicted_depth,
                self.default_min_alt,
                self.default_max_alt
            )

        d_arr = np.array(d_samples, dtype=np.float64)
        h_arr = np.array(h_samples, dtype=np.float64)

        A = np.vstack([d_arr, np.ones_like(d_arr)]).T
        solution, _, _, _ = np.linalg.lstsq(A, h_arr, rcond=None)
        alpha, beta = float(solution[0]), float(solution[1])

        metric_dsm = (alpha * predicted_depth + beta).astype(np.float32)

        h_fitted = alpha * d_arr + beta
        errors = h_arr - h_fitted
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        min_elev = float(np.nanmin(metric_dsm))
        max_elev = float(np.nanmax(metric_dsm))

        return CalibrationResult(
            metric_dsm=metric_dsm,
            alpha=alpha,
            beta=beta,
            min_elevation=min_elev,
            max_elevation=max_elev,
            elevation_range=max_elev - min_elev,
            rmse=rmse,
            mae=mae,
            calibration_type="reference_points"
        )

    def _calibrate_relative_fallback(
        self,
        predicted_depth: np.ndarray,
        min_alt: float,
        max_alt: float
    ) -> CalibrationResult:
        """
        Normalizes relative depth to [0, 1] and maps linearly to [min_alt, max_alt].
        """
        d_min = float(np.nanmin(predicted_depth))
        d_max = float(np.nanmax(predicted_depth))
        d_range = d_max - d_min if d_max > d_min else 1.0

        # Derivation:
        # H_metric = min_alt + ((D_pred - d_min) / d_range) * (max_alt - min_alt)
        # H_metric = alpha * D_pred + beta
        # alpha = (max_alt - min_alt) / d_range
        # beta = min_alt - alpha * d_min

        alpha = (max_alt - min_alt) / d_range
        beta = min_alt - alpha * d_min

        norm_depth = (predicted_depth - d_min) / d_range
        metric_dsm = (min_alt + norm_depth * (max_alt - min_alt)).astype(np.float32)

        min_elev = float(np.nanmin(metric_dsm))
        max_elev = float(np.nanmax(metric_dsm))

        return CalibrationResult(
            metric_dsm=metric_dsm,
            alpha=alpha,
            beta=beta,
            min_elevation=min_elev,
            max_elevation=max_elev,
            elevation_range=max_elev - min_elev,
            rmse=None,
            mae=None,
            calibration_type="relative_fallback"
        )


def clip_val(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, val))
