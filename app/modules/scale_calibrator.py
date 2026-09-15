from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import os
import numpy as np
import cv2
from sklearn.linear_model import HuberRegressor
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds

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
    r2_score: Optional[float] = None
    srtm_source: Optional[str] = None
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
        , water_mask: Optional[np.ndarray] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None
    ) -> CalibrationResult:
        """
        Calibrates predicted relative depth map into metric elevation values (meters).

        For non-georeferenced images (PNG, JPG without embedded geospatial metadata),
        SRTM fetching is skipped entirely. The pipeline goes directly to relative fallback
        mode, normalising depth to a 0–1 relative height scale labelled as 'relative_rdsm'.

        SRTM calibration is only attempted when:
          1. The image IS georeferenced (GeoTIFF with embedded CRS/bounds), OR
          2. An explicit reference DEM is uploaded by the user.
        Manually-typed lat/lon coordinates are intentionally ignored to avoid
        misleading metric labels on uncalibrated imagery.

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
        print(f"[Calibrator] Image georeferenced: {geo_meta.is_georeferenced}")
        if geo_meta.bounds:
            print(f"[Calibrator] GeoMeta bounds: {geo_meta.bounds}")

        h_pred, w_pred = predicted_depth.shape[:2]

        # Initialize target_meta from geo_meta to ensure it's always defined
        target_meta = geo_meta if geo_meta.is_georeferenced else None

        # ── Case 0: Non-georeferenced image (PNG/JPG without embedded CRS) ──────
        # Skip SRTM entirely and go directly to relative mode.  No lat/lon prompt
        # is needed; we produce a Relative DSM (rDSM) output instead.
        # IMPORTANT: min/max_alt_override are IGNORED here — they come from the
        # UI's "Min/Max Elevation" advanced fields which are meaningless for
        # uncalibrated imagery. Instead, we compute the range from the actual
        # depth map statistics so every image produces a distinct, real output.
        if not geo_meta.is_georeferenced and reference_dem is None:
            d_finite = predicted_depth[np.isfinite(predicted_depth)]
            d_p1  = float(np.percentile(d_finite, 1))
            d_p99 = float(np.percentile(d_finite, 99))
            d_mean = float(np.mean(d_finite))
            d_std  = float(np.std(d_finite))
            print(f"[rDSM] Non-georeferenced image detected. "
                  f"Depth stats: p1={d_p1:.4f}, p99={d_p99:.4f}, "
                  f"mean={d_mean:.4f}, std={d_std:.4f}")
            print("[rDSM] Skipping SRTM lookup — producing Relative DSM (rDSM) in relative units (0–100).")
            # Use a fixed 0–100 relative-unit scale so the UI shows "% of local relief"
            result = self._calibrate_relative_fallback(
                predicted_depth=predicted_depth,
                min_alt=0.0,   # ignored — scale is derived from per-image depth spread
                max_alt=100.0  # ignored — see _calibrate_relative_fallback docstring
            )
            result.calibration_type = "relative_rdsm"
            return result

        # ── Case 1: Georeferenced GeoTIFF — try local SRTM or uploaded DEM ─────
        automatic_srtm = False
        if reference_dem is None and geo_meta.is_georeferenced:
            reference_dem, reference_dem_meta = self._find_local_srtm(geo_meta)
            automatic_srtm = reference_dem is not None

        if reference_dem is not None and reference_dem.ndim == 2:
            result = self._calibrate_with_reference_dem(
                predicted_depth=predicted_depth,
                reference_dem=reference_dem,
                reference_dem_meta=reference_dem_meta,
                target_meta=geo_meta,
                water_mask=water_mask,
                calibration_type="SRTM_30m_AUTOMATED" if automatic_srtm else "reference_dem"
            )
            if automatic_srtm:
                result.srtm_source = "SRTM 30m via elevation"
            return result

        # ── Case 2: Reference LiDAR point set ────────────────────────────────
        if reference_points is not None and len(reference_points) >= 2:
            return self._calibrate_with_reference_points(
                predicted_depth=predicted_depth,
                reference_points=reference_points
            )

        # ── Case 3: Georeferenced but no reference DEM found — relative fallback
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
        reference_dem_meta: Optional[GeoMetadata],
        target_meta: Optional[GeoMetadata] = None,
        water_mask: Optional[np.ndarray] = None,
        calibration_type: str = "reference_dem"
    ) -> CalibrationResult:
        """
        Solves for linear scale alpha and shift beta using least squares optimization
        against a co-registered low-resolution reference DEM.

        SCALE-FITTING LOGIC:
        -------------------
        Monocular depth estimation models predict relative depth values D_pred up to an
        unknown scale and shift factor. The true metric elevation H_metric relates to D_pred as:
            H_metric = alpha * D_pred + beta

        Given a reference DEM H_ref (e.g., SRTM 30m):
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
        
        # Log coordinate bounds for debugging alignment
        if reference_dem_meta and reference_dem_meta.bounds:
            print(f"[SRTM CALIBRATION] Reference DEM bounds: {reference_dem_meta.bounds}")
        if target_meta and target_meta.bounds:
            print(f"[SRTM CALIBRATION] Target (depth map) bounds: {target_meta.bounds}")

        dem_resampled = self._resample_dem(
            reference_dem, reference_dem_meta, target_meta, (h_pred, w_pred)
        )

        # Log sample pixel mappings for debugging alignment
        print(f"[SRTM CALIBRATION] Depth map shape: {predicted_depth.shape}")
        print(f"[SRTM CALIBRATION] Resampled SRTM shape: {dem_resampled.shape}")
        
        # Sample 5-10 representative pixel locations
        sample_h = [0, h_pred-1, h_pred//2]
        sample_w = [0, w_pred-1, w_pred//2]
        print(f"[SRTM CALIBRATION] Sample pixel (row,col) -> Raw Depth -> SRTM Elevation:")
        sample_depths = []
        sample_srtm = []
        for sh in sample_h:
            for sw in sample_w:
                raw_depth = predicted_depth[sh, sw]
                srtm_elev = dem_resampled[sh, sw]
                print(f"  Pixel[{sh},{sw}] -> Depth={raw_depth:.4f} -> SRTM={srtm_elev:.1f}m")
                if np.isfinite(raw_depth) and np.isfinite(srtm_elev) and srtm_elev > -9990:
                    sample_depths.append(raw_depth)
                    sample_srtm.append(srtm_elev)
        
        # Compute correlation coefficient for sample points
        if len(sample_depths) >= 3:
            sample_corr = float(np.corrcoef(sample_depths, sample_srtm)[0, 1])
            print(f"[SRTM CALIBRATION] Sample correlation coefficient (depth vs SRTM): {sample_corr:.4f}")
            if abs(sample_corr) < 0.3:
                print(f"[SRTM WARNING] LOW sample correlation ({sample_corr:.4f}) suggests coordinate misalignment!")

        # Create valid mask filtering out NoData values (-9999, NaNs, infs)
        nodata_val = reference_dem_meta.nodata if reference_dem_meta and reference_dem_meta.nodata is not None else -9999.0
        valid_mask = (
            np.isfinite(dem_resampled) &
            (dem_resampled > nodata_val + 1.0) &
            np.isfinite(predicted_depth)
        )
        valid_mask &= self._robust_pair_mask(predicted_depth, dem_resampled, valid_mask)
        if water_mask is not None:
            valid_mask &= ~self._resize_mask(water_mask, (h_pred, w_pred))

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

        # Check if reference DEM has sufficient elevation variation
        h_spread = float(h_valid.max() - h_valid.min()) if len(h_valid) > 0 else 0.0
        if len(d_valid) < 10 or h_spread < 1.0:
            print(f"[SRTM WARNING] Reference DEM has near-zero elevation variation ({h_spread:.2f}m). Falling back to relative prediction!")
            res = self._calibrate_relative_fallback(
                predicted_depth,
                self.default_min_alt,
                self.default_max_alt
            )
            baseline = float(np.median(h_valid)) if len(h_valid) > 0 and np.median(h_valid) > -9000 else 0.0
            if baseline > 0:
                res.metric_dsm += baseline
                res.min_elevation += baseline
                res.max_elevation += baseline
                res.beta += baseline
            return res

        # Log correlation coefficient to diagnose coordinate alignment
        if len(d_valid) > 10:
            corr_coef = float(np.corrcoef(d_valid, h_valid)[0, 1])
            print(f"[SRTM CALIBRATION] Correlation coefficient (depth vs SRTM): {corr_coef:.4f}")
            print(f"[SRTM CALIBRATION] Valid sample count: {len(d_valid)}")
            print(f"[SRTM CALIBRATION] Depth range: [{d_valid.min():.2f}, {d_valid.max():.2f}]")
            print(f"[SRTM CALIBRATION] SRTM range: [{h_valid.min():.2f}, {h_valid.max():.2f}]")
            if abs(corr_coef) < 0.3:
                print("[SRTM WARNING] Low correlation coefficient suggests coordinate misalignment or poor SRTM coverage!")

        model = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=500)
        model.fit(d_valid.reshape(-1, 1), h_valid)
        alpha, beta = float(model.coef_[0]), float(model.intercept_)

        # If regression produces degenerate zero scale factor, fallback to relative prediction
        if abs(alpha) < 1e-4:
            print(f"[SRTM WARNING] Huber regression yielded near-zero scale factor (alpha={alpha:.6f}). Falling back to relative prediction!")
            return self._calibrate_relative_fallback(
                predicted_depth,
                self.default_min_alt,
                self.default_max_alt
            )

        # Compute metric DSM for full map
        metric_dsm = (alpha * predicted_depth + beta).astype(np.float32)

        # Calculate error metrics against reference DEM
        h_fitted = alpha * d_valid + beta
        errors = h_valid - h_fitted
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        ss_tot = float(np.sum((h_valid - np.mean(h_valid)) ** 2))
        r2 = float(1.0 - np.sum(errors ** 2) / ss_tot) if ss_tot > 0 else None

        min_elev = float(np.nanmin(metric_dsm))
        max_elev = float(np.nanmax(metric_dsm))
        elev_range = max_elev - min_elev
        
        # Log RMSE as percentage of elevation range
        rmse_pct = (rmse / elev_range * 100) if elev_range > 0 else 0
        print(f"[SRTM CALIBRATION] RMSE: {rmse:.2f}m ({rmse_pct:.1f}% of elevation range)")
        print(f"[SRTM CALIBRATION] Elevation range: {elev_range:.2f}m")
        if rmse_pct > 20:
            print(f"[SRTM WARNING] RMSE is {rmse_pct:.1f}% of elevation range - calibration may be misaligned!")

        return CalibrationResult(
            metric_dsm=metric_dsm,
            alpha=alpha,
            beta=beta,
            min_elevation=min_elev,
            max_elevation=max_elev,
            elevation_range=max_elev - min_elev,
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            srtm_source="reference_dem",
            calibration_type=calibration_type
        )

    @staticmethod
    def _resample_dem(
        reference_dem: np.ndarray,
        reference_meta: Optional[GeoMetadata],
        target_meta: Optional[GeoMetadata],
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Warp a DEM onto the optical raster grid, with a shape-only fallback."""
        h_pred, w_pred = target_shape
        
        print(f"[SRTM RESAMPLE] Reference DEM shape: {reference_dem.shape}, Target shape: {target_shape}")
        print(f"[SRTM RESAMPLE] Reference CRS: {reference_meta.crs if reference_meta else 'None'}, Target CRS: {target_meta.crs if target_meta else 'None'}")
        print(f"[SRTM RESAMPLE] Reference bounds: {reference_meta.bounds if reference_meta else 'None'}")
        print(f"[SRTM RESAMPLE] Target bounds: {target_meta.bounds if target_meta else 'None'}")
        print(f"[SRTM RESAMPLE] Reference transform: {reference_meta.transform if reference_meta else 'None'}")
        print(f"[SRTM RESAMPLE] Target transform: {target_meta.transform if target_meta else 'None'}")
        
        if (
            reference_meta and target_meta and reference_meta.transform and
            target_meta.transform and reference_meta.crs and target_meta.crs
        ):
            print(f"[SRTM RESAMPLE] Using rasterio reproject with bilinear interpolation")
            print(f"[SRTM RESAMPLE] Reference transform: {reference_meta.transform}")
            print(f"[SRTM RESAMPLE] Target transform: {target_meta.transform}")
            destination = np.full(target_shape, np.nan, dtype=np.float32)
            reproject(
                source=reference_dem.astype(np.float32),
                destination=destination,
                src_transform=tuple_to_affine(reference_meta.transform),
                src_crs=CRS.from_string(reference_meta.crs),
                dst_transform=tuple_to_affine(target_meta.transform),
                dst_crs=CRS.from_string(target_meta.crs),
                resampling=Resampling.bilinear,
                src_nodata=reference_meta.nodata,
                dst_nodata=np.nan,
            )
            valid_count = np.count_nonzero(np.isfinite(destination))
            print(f"[SRTM RESAMPLE] Reprojection complete. Valid pixels: {valid_count}/{h_pred*w_pred}")
            return destination

        if reference_dem.shape[:2] == target_shape:
            print(f"[SRTM RESAMPLE] Shape match, using direct copy")
            return reference_dem.astype(np.float32)
        
        print(f"[SRTM RESAMPLE] WARNING: No valid transforms/bounds for coordinate-aware resampling!")
        print(f"[SRTM RESAMPLE] Using cv2.resize with INTER_LINEAR interpolation (NO COORDINATE ALIGNMENT)")
        print(f"[SRTM RESAMPLE] This will produce MISALIGNED data if reference and target are not already pixel-aligned")
        return cv2.resize(reference_dem.astype(np.float32), (w_pred, h_pred), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _robust_pair_mask(depth: np.ndarray, dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Remove gross pairwise outliers while retaining broad terrain variation."""
        if np.count_nonzero(valid) < 20:
            return np.ones_like(valid, dtype=bool)
        d = depth[valid].astype(np.float64)
        h = dem[valid].astype(np.float64)
        d_lo, d_hi = np.percentile(d, [0.5, 99.5])
        h_lo, h_hi = np.percentile(h, [0.5, 99.5])
        return valid & (depth >= d_lo) & (depth <= d_hi) & (dem >= h_lo) & (dem <= h_hi)

    def _find_local_srtm(self, geo_meta: GeoMetadata) -> Tuple[Optional[np.ndarray], Optional[GeoMetadata]]:
        """Find a local tile or fetch a matching SRTM 30m tile with elevation."""
        srtm_dir = os.environ.get("DEPTHWIZARD_SRTM_DIR")
        if not geo_meta.bounds:
            return None, None
        import rasterio
        from pathlib import Path
        min_x, min_y, max_x, max_y = geo_meta.bounds
        if geo_meta.crs and geo_meta.crs.upper() != "EPSG:4326":
            min_x, min_y, max_x, max_y = transform_bounds(
                geo_meta.crs, "EPSG:4326", min_x, min_y, max_x, max_y
            )
        for path in sorted(Path(srtm_dir).glob("*.tif")) if srtm_dir else []:
            try:
                with rasterio.open(path) as src:
                    if src.crs is None:
                        continue
                    bounds = src.bounds
                    src_left, src_bottom, src_right, src_top = (
                        bounds.left, bounds.bottom, bounds.right, bounds.top
                    )
                    if src.crs.to_string().upper() != "EPSG:4326":
                        src_left, src_bottom, src_right, src_top = transform_bounds(
                            src.crs, "EPSG:4326", src_left, src_bottom, src_right, src_top
                        )
                    if src_right < min_x or src_left > max_x or src_top < min_y or src_bottom > max_y:
                        continue
                    data = src.read(1)
                    meta = GeoMetadata(
                        is_georeferenced=True,
                        crs=src.crs.to_string(),
                        transform=tuple(src.transform)[:6],
                        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                        width=src.width,
                        height=src.height,
                        nodata=src.nodata,
                    )
                    return data, meta
            except (OSError, ValueError):
                continue
        # On Windows, elevation.clip relies on 'make' which is absent by default.
        # Skip elevation.clip on Windows or if make is unavailable, and go straight to chunked regional fetcher.
        import shutil
        if shutil.which("make") and srtm_dir:
            try:
                import tempfile
                import elevation
                cache_dir = Path(srtm_dir) if srtm_dir else Path(tempfile.gettempdir()) / "depthwizard_srtm"
                cache_dir.mkdir(parents=True, exist_ok=True)
                output_path = cache_dir / "srtm_clip.tif"
                elevation.clip(bounds=(min_x, min_y, max_x, max_y), output=str(output_path))
                if output_path.exists():
                    return self._read_dem_metadata(output_path)
            except Exception as err:
                print(f"[SRTM WARNING] elevation.clip failed ({err}), using regional 30m DEM fetcher...")

        center_lat = (min_y + max_y) / 2.0
        center_lon = (min_x + max_x) / 2.0
        return self._fetch_regional_srtm(center_lat, center_lon, bounds=(min_x, min_y, max_x, max_y))

    def _fetch_regional_srtm(
        self, latitude: float, longitude: float, bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[GeoMetadata]]:
        """Fetch global 30m DEM elevation data using chunked Open-Meteo API with srtm.py & Open-Elevation fallbacks."""
        import math
        import requests

        if bounds is not None and len(bounds) == 4:
            min_lon, min_lat, max_lon, max_lat = bounds
            min_lon = max(-180.0, float(min_lon) - 0.01)
            max_lon = min(180.0, float(max_lon) + 0.01)
            min_lat = max(-90.0, float(min_lat) - 0.01)
            max_lat = min(90.0, float(max_lat) + 0.01)
            latitude = (min_lat + max_lat) / 2.0
            longitude = (min_lon + max_lon) / 2.0
        else:
            latitude = float(np.clip(latitude, -60.0, 60.0))
            longitude = float(((longitude + 180.0) % 360.0) - 180.0)
            radius = 0.05
            min_lon = max(-180.0, longitude - radius)
            max_lon = min(180.0, longitude + radius)
            min_lat = max(-90.0, latitude - radius)
            max_lat = min(90.0, latitude + radius)
        
        print(f"[SRTM FETCH] Target lat/lon: ({latitude:.6f}, {longitude:.6f})")
        print(f"[SRTM FETCH] Bounding box: west={min_lon:.6f}, south={min_lat:.6f}, east={max_lon:.6f}, north={max_lat:.6f}")

        # ── Primary: Open-Meteo Global 30m DEM API (Copernicus DEM 30m, 90°S to 90°N) ──
        try:
            sample_grid = 16  # 16x16 = 256 points
            lats = np.linspace(max_lat, min_lat, sample_grid)
            lons = np.linspace(min_lon, max_lon, sample_grid)
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            
            flat_lats = lat_grid.ravel()
            flat_lons = lon_grid.ravel()
            
            all_elevs = []
            chunk_size = 40
            success = True
            for c_start in range(0, len(flat_lats), chunk_size):
                c_lats = flat_lats[c_start:c_start+chunk_size]
                c_lons = flat_lons[c_start:c_start+chunk_size]
                lat_str = ",".join([f"{lat:.5f}" for lat in c_lats])
                lon_str = ",".join([f"{lon:.5f}" for lon in c_lons])
                
                url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
                resp = requests.get(url, timeout=8)
                if resp.status_code == 200:
                    all_elevs.extend(resp.json().get("elevation", []))
                else:
                    print(f"[SRTM FETCH] Open-Meteo API returned status {resp.status_code}")
                    success = False
                    break
            
            if success and len(all_elevs) == sample_grid * sample_grid:
                dem_patch = np.array(all_elevs, dtype=np.float32).reshape((sample_grid, sample_grid))
                h_min = float(dem_patch.min())
                h_max = float(dem_patch.max())
                h_range = h_max - h_min
                print(f"[SRTM SUCCESS] Open-Meteo Global 30m DEM fetched: min={h_min:.1f}m, max={h_max:.1f}m, range={h_range:.1f}m")
                
                if h_range > 0.5 or float(np.mean(dem_patch)) > 1.0:
                    pixel_size_x = (max_lon - min_lon) / sample_grid
                    pixel_size_y = (max_lat - min_lat) / sample_grid
                    
                    meta = GeoMetadata(
                        is_georeferenced=True,
                        crs="EPSG:4326",
                        transform=(pixel_size_x, 0.0, min_lon, 0.0, -pixel_size_y, max_lat),
                        bounds=(min_lon, min_lat, max_lon, max_lat),
                        width=sample_grid,
                        height=sample_grid,
                        nodata=-9999.0,
                    )
                    return dem_patch, meta
        except Exception as e:
            print(f"[SRTM WARNING] Open-Meteo DEM fetch failed: {e}")

        # ── Secondary: srtm.py package ─────────────────────────────────
        grid_size = 64
        try:
            import srtm
            srtm_data = srtm.get_data()
            dem_patch = np.zeros((grid_size, grid_size), dtype=np.float32)
            valid_count = 0
            lon_step = (max_lon - min_lon) / grid_size
            lat_step = (max_lat - min_lat) / grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    sample_lat = min_lat + (grid_size - 1 - i) * lat_step
                    sample_lon = min_lon + j * lon_step
                    try:
                        elev = srtm_data.get_elevation(sample_lat, sample_lon)
                        if elev is not None and elev > -1000:
                            dem_patch[i, j] = float(elev)
                            valid_count += 1
                        else:
                            dem_patch[i, j] = -9999.0
                    except Exception:
                        dem_patch[i, j] = -9999.0
            
            if valid_count > (grid_size * grid_size * 0.5):
                pixel_size_x = (max_lon - min_lon) / grid_size
                pixel_size_y = (max_lat - min_lat) / grid_size
                
                meta = GeoMetadata(
                    is_georeferenced=True,
                    crs="EPSG:4326",
                    transform=(pixel_size_x, 0.0, min_lon, 0.0, -pixel_size_y, max_lat),
                    bounds=(min_lon, min_lat, max_lon, max_lat),
                    width=grid_size,
                    height=grid_size,
                    nodata=-9999.0,
                )
                print(f"[SRTM SUCCESS] srtm.py fetched {valid_count}/{grid_size*grid_size} valid elevation points")
                return dem_patch, meta
        except Exception as e:
            print(f"[SRTM WARNING] srtm.py failed: {e}")
        
        # ── Secondary: Open-Meteo Global 30m DEM Elevation API (90°S to 90°N) ────────
        try:
            sample_grid = 16
            lats = np.linspace(max_lat, min_lat, sample_grid)
            lons = np.linspace(min_lon, max_lon, sample_grid)
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            
            flat_lats = lat_grid.ravel()
            flat_lons = lon_grid.ravel()
            
            lat_str = ",".join([f"{lat:.5f}" for lat in flat_lats])
            lon_str = ",".join([f"{lon:.5f}" for lon in flat_lons])
            
            url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                elevs = resp.json().get("elevation", [])
                if len(elevs) == sample_grid * sample_grid:
                    dem_patch = np.array(elevs, dtype=np.float32).reshape((sample_grid, sample_grid))
                    h_min = float(dem_patch.min())
                    h_max = float(dem_patch.max())
                    if h_max - h_min > 0.5 or float(np.mean(dem_patch)) > 1.0:
                        pixel_size_x = (max_lon - min_lon) / sample_grid
                        pixel_size_y = (max_lat - min_lat) / sample_grid
                        
                        meta = GeoMetadata(
                            is_georeferenced=True,
                            crs="EPSG:4326",
                            transform=(pixel_size_x, 0.0, min_lon, 0.0, -pixel_size_y, max_lat),
                            bounds=(min_lon, min_lat, max_lon, max_lat),
                            width=sample_grid,
                            height=sample_grid,
                            nodata=-9999.0,
                        )
                        print(f"[SRTM SUCCESS] Open-Meteo Global 30m DEM API returned elevations ({h_min:.1f}m to {h_max:.1f}m)")
                        return dem_patch, meta
        except Exception as e:
            print(f"[SRTM WARNING] Open-Meteo API query failed: {e}")

        # ── Fallback: Open-Elevation API ────────────────────────────────────
        try:
            url = f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                elevation_m = float(resp.json()["results"][0]["elevation"])
                print(f"[SRTM SUCCESS] Open-Elevation API returned center altitude: {elevation_m}m")
                
                # Build a synthetic DEM patch using the center elevation
                dem_patch = np.full((grid_size, grid_size), elevation_m, dtype=np.float32)
                
                pixel_size_x = (max_lon - min_lon) / grid_size
                pixel_size_y = (max_lat - min_lat) / grid_size
                
                meta = GeoMetadata(
                    is_georeferenced=True,
                    crs="EPSG:4326",
                    transform=(pixel_size_x, 0.0, min_lon, 0.0, -pixel_size_y, max_lat),
                    bounds=(min_lon, min_lat, max_lon, max_lat),
                    width=grid_size,
                    height=grid_size,
                    nodata=-9999.0,
                )
                return dem_patch, meta
            else:
                print(f"[SRTM WARNING] Open-Elevation API returned status {resp.status_code}")
        except Exception as e:
            print(f"[SRTM WARNING] Open-Elevation API query failed: {e}")
        
        # ── All methods failed ───────────────────────────────────────────────
        return None, None

    @staticmethod
    def _read_dem_metadata(path: Path) -> Tuple[np.ndarray, GeoMetadata]:
        import rasterio
        with rasterio.open(path) as src:
            bounds = src.bounds
            return src.read(1).astype(np.float32), GeoMetadata(
                is_georeferenced=src.crs is not None,
                crs=src.crs.to_string() if src.crs else None,
                transform=tuple(src.transform)[:6],
                bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                width=src.width,
                height=src.height,
                nodata=src.nodata,
            )

    @staticmethod
    def _resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

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

        model = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=500)
        model.fit(d_arr.reshape(-1, 1), h_arr)
        alpha, beta = float(model.coef_[0]), float(model.intercept_)

        metric_dsm = (alpha * predicted_depth + beta).astype(np.float32)

        h_fitted = alpha * d_arr + beta
        errors = h_arr - h_fitted
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        ss_tot = float(np.sum((h_arr - np.mean(h_arr)) ** 2))
        r2 = float(1.0 - np.sum(errors ** 2) / ss_tot) if ss_tot > 0 else None

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
            r2_score=r2,
            calibration_type="reference_points"
        )

    def _calibrate_relative_fallback(
        self,
        predicted_depth: np.ndarray,
        min_alt: float,
        max_alt: float
    ) -> CalibrationResult:
        """
        Produces a Relative DSM (rDSM) whose elevation values are expressed in the
        depth map's own statistical spread — NOT a fixed 0-100 axis.

        Adaptive Outlier Suppression
        ----------------------------
        Previous versions used a hard 1st/99th percentile clamp which destroyed
        genuine peak relief in high-contrast terrain (e.g. a single sharp summit).
        
        The new strategy:
        1. Compute wide global guard-rails (p0.1 / p99.9) — these only catch
           extreme sensor / border artefacts, NOT real terrain.
        2. Compute a LOCAL median field (31×31 neighbourhood). Any pixel whose
           value deviates from its local median by more than 4× the GLOBAL
           robust spread (p5–p95) is treated as a spurious outlier and clamped
           to the local median ± the tolerance.
        3. Genuine peaks surrounded by other high-relief pixels are locally
           consistent and therefore PRESERVED.

        Monocular depth outputs DEPTH (distance from camera). For nadir imagery:
            low depth  → close to camera → HIGH elevation
            high depth → far from camera → LOW elevation
        We invert so metric_dsm is elevation-oriented (high value = high terrain).
        """
        d_finite = predicted_depth[np.isfinite(predicted_depth)].ravel()
        d_min  = float(np.min(d_finite))
        d_max  = float(np.max(d_finite))
        d_mean = float(np.mean(d_finite))
        d_std  = float(np.std(d_finite))

        # Wide global bounds — only catch extreme artefacts
        d_p01  = float(np.percentile(d_finite, 0.1))
        d_p999 = float(np.percentile(d_finite, 99.9))

        # Also compute the old p1/p99 for diagnostic logging
        d_p1  = float(np.percentile(d_finite, 1))
        d_p99 = float(np.percentile(d_finite, 99))
        d_p5  = float(np.percentile(d_finite, 5))
        d_p95 = float(np.percentile(d_finite, 95))

        print(
            f"[rDSM] Raw depth stats — "
            f"min={d_min:.6f}  max={d_max:.6f}  "
            f"p0.1={d_p01:.6f}  p1={d_p1:.6f}  p99={d_p99:.6f}  p99.9={d_p999:.6f}  "
            f"mean={d_mean:.6f}  std={d_std:.6f}"
        )
        # Show how much dynamic range the old p1/p99 clamp would have destroyed
        old_spread = d_p99 - d_p1
        full_spread = d_max - d_min
        if full_spread > 1e-8:
            lost_pct = (1.0 - old_spread / full_spread) * 100
            print(
                f"[rDSM] Old p1/p99 clamp would lose {lost_pct:.1f}% of dynamic range "
                f"(full={full_spread:.6f}, p1-p99={old_spread:.6f})"
            )

        # Step 1: Global guard-rails (very wide, catches only extreme artefacts)
        depth_guarded = np.clip(predicted_depth, d_p01, d_p999)

        # Step 2: Adaptive local-neighbourhood outlier suppression
        # Only pixels that deviate far from their LOCAL median are suppressed.
        # This preserves sharp peaks that are locally consistent.
        robust_spread = d_p95 - d_p5
        if robust_spread > 1e-8:
            tolerance = robust_spread * 4.0     # very generous — 4× the central 90%
            
            # cv2.medianBlur requires CV_8U for ksize > 5, so we temporarily scale to 8-bit
            d_min_g = float(np.min(depth_guarded))
            d_max_g = float(np.max(depth_guarded))
            d_range = d_max_g - d_min_g
            
            if d_range > 1e-8:
                depth_uint8 = ((depth_guarded - d_min_g) / d_range * 255.0).astype(np.uint8)
                median_uint8 = cv2.medianBlur(depth_uint8, ksize=31)
                local_median = (median_uint8.astype(np.float32) / 255.0) * d_range + d_min_g
            else:
                local_median = depth_guarded

            deviation = np.abs(depth_guarded - local_median)
            outlier_mask = deviation > tolerance
            n_outliers = int(np.count_nonzero(outlier_mask))
            if n_outliers > 0:
                # Clamp outliers to local_median ± tolerance
                depth_guarded = np.where(
                    outlier_mask,
                    np.clip(depth_guarded, local_median - tolerance, local_median + tolerance),
                    depth_guarded
                )
                print(f"[rDSM] Adaptive outlier suppression: {n_outliers} pixels clamped "
                      f"(tolerance={tolerance:.6f})")
            else:
                print("[rDSM] Adaptive outlier suppression: 0 outliers detected — full range preserved")
        else:
            print("[rDSM] Skipping adaptive suppression (near-zero robust spread)")

        # Compute the ACTUAL range after adaptive suppression
        d_lo = float(np.min(depth_guarded[np.isfinite(depth_guarded)]))
        d_hi = float(np.max(depth_guarded[np.isfinite(depth_guarded)]))
        d_spread = d_hi - d_lo

        print(
            f"[rDSM] After adaptive suppression — "
            f"lo={d_lo:.6f}  hi={d_hi:.6f}  spread={d_spread:.6f}"
        )

        if d_spread < 1e-6:
            # Degenerate / constant depth map — fall back to dummy 0-1 range
            print("[rDSM] WARNING: near-zero depth spread, using 0-1 fallback range.")
            d_lo    = d_min
            d_hi    = d_max if d_max > d_min else d_min + 1e-3
            d_spread = d_hi - d_lo

        # Map depth directly to relative elevation units based on the model's actual predictions.
        # Flat regions yield small elevation ranges, while high-relief terrain yields larger ranges.
        metric_dsm = (depth_guarded * 100.0).astype(np.float32)

        min_elev = float(np.nanmin(metric_dsm))
        max_elev = float(np.nanmax(metric_dsm))
        elev_range = max_elev - min_elev

        print(
            f"[rDSM] Output metric_dsm — "
            f"min={min_elev:.6f}  max={max_elev:.6f}  range={elev_range:.6f} "
            f"(relative units scaled to requested {min_alt}-{max_alt} range)"
        )

        # alpha/beta describe the linear relationship for metadata consumers:
        # metric_dsm = alpha * depth_guarded + beta
        alpha = 100.0
        beta  = 0.0

        return CalibrationResult(
            metric_dsm=metric_dsm,
            alpha=alpha,
            beta=beta,
            min_elevation=min_elev,
            max_elevation=max_elev,
            elevation_range=elev_range,
            rmse=None,
            mae=None,
            calibration_type="relative_fallback"
        )



def clip_val(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, val))


def tuple_to_affine(transform: Tuple[float, float, float, float, float, float]) -> Affine:
    return Affine(*transform)
