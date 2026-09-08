import os
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from app.config import OUTPUT_DIR, DEFAULT_MODEL_ID, DEFAULT_MIN_ELEVATION_METERS, DEFAULT_MAX_ELEVATION_METERS, get_device, FRONTEND_DIR
from app.utils.image_io import load_optical_image, load_reference_dem
from app.modules.depth_extractor import DepthExtractor
from app.modules.scale_calibrator import ScaleCalibrator
from app.modules.formatter import OutputFormatter


# Global pipeline components
depth_extractor: Optional[DepthExtractor] = None
scale_calibrator: Optional[ScaleCalibrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    Pre-initializes the Depth Anything v2 model and scale calibrator on app startup.
    """
    global depth_extractor, scale_calibrator
    print("[FastAPI Startup] Initializing DSM extraction pipeline models...")
    try:
        depth_extractor = DepthExtractor(model_id=DEFAULT_MODEL_ID)
        scale_calibrator = ScaleCalibrator(
            default_min_alt=DEFAULT_MIN_ELEVATION_METERS,
            default_max_alt=DEFAULT_MAX_ELEVATION_METERS
        )
        print("[FastAPI Startup] Models loaded successfully!")
    except Exception as e:
        print(f"[FastAPI Startup WARNING] Model initialization deferred to first request: {e}")
    
    yield
    
    print("[FastAPI Shutdown] Cleaning up resources...")


app = FastAPI(
    title="ISRO DepthWizard - Automated Digital Surface Model (DSM) Pipeline",
    description="Production-ready FastAPI backend for single-view depth extraction and metric scale-calibration.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output subdirectories
UPLOADS_DIR = OUTPUT_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static outputs
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

# Mount frontend (served at /app/*)
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to the frontend."""
    return RedirectResponse(url="/app/index.html")


def get_depth_extractor() -> DepthExtractor:
    global depth_extractor
    if depth_extractor is None:
        depth_extractor = DepthExtractor(model_id=DEFAULT_MODEL_ID)
    return depth_extractor


def get_scale_calibrator() -> ScaleCalibrator:
    global scale_calibrator
    if scale_calibrator is None:
        scale_calibrator = ScaleCalibrator(
            default_min_alt=DEFAULT_MIN_ELEVATION_METERS,
            default_max_alt=DEFAULT_MAX_ELEVATION_METERS
        )
    return scale_calibrator


@app.get("/health", tags=["Diagnostic"])
async def health_check() -> Dict[str, Any]:
    """Diagnostic health check endpoint."""
    device = get_device()
    return {
        "status": "online",
        "device": device,
        "model_id": DEFAULT_MODEL_ID,
        "output_directory": str(OUTPUT_DIR)
    }


@app.post("/process", tags=["DSM Extraction Pipeline"])
async def process_image(
    file: UploadFile = File(..., description="Optical input image (PNG, JPG, or GeoTIFF)"),
    ref_dem: Optional[UploadFile] = File(None, description="Optional low-resolution reference DEM GeoTIFF (e.g. SRTM 30m)"),
    min_alt: float = Form(DEFAULT_MIN_ELEVATION_METERS, description="Fallback minimum elevation in meters if uncalibrated"),
    max_alt: float = Form(DEFAULT_MAX_ELEVATION_METERS, description="Fallback maximum elevation in meters if uncalibrated"),
    tile_size: int = Form(512, description="Sliding window tile size in pixels"),
    overlap_ratio: float = Form(0.20, description="Tiled sliding window overlap ratio (0.0 - 0.5)")
) -> Dict[str, Any]:
    """
    POST /process: Runs the automated Digital Surface Model (DSM) extraction pipeline.
    
    Processing Steps:
    1. Module A: Single-View Depth Extraction with 2D Hann window blending.
    2. Module B: Scale & Shift Calibration (Linear OLS against reference DEM or fallback).
    3. Module C: Export Unity 16-bit Heightmap PNG, 32-bit float GeoTIFF, and Metadata JSON.
    """
    task_id = f"dsm_{uuid.uuid4().hex[:10]}"

    # Save uploaded input file
    file_ext = Path(file.filename).suffix if file.filename else ".tif"
    input_filepath = UPLOADS_DIR / f"{task_id}_input{file_ext}"
    
    try:
        contents = await file.read()
        with open(input_filepath, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded optical image: {str(e)}"
        )

    # Save optional reference DEM
    ref_dem_filepath: Optional[Path] = None
    if ref_dem is not None and ref_dem.filename:
        ref_dem_ext = Path(ref_dem.filename).suffix
        ref_dem_filepath = UPLOADS_DIR / f"{task_id}_ref_dem{ref_dem_ext}"
        try:
            ref_contents = await ref_dem.read()
            with open(ref_dem_filepath, "wb") as f:
                f.write(ref_contents)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save uploaded reference DEM: {str(e)}"
            )

    try:
        # Load optical image and geospatial metadata
        rgb_image, geo_meta = load_optical_image(str(input_filepath))

        # Load reference DEM if provided
        reference_dem_arr: Optional[np.ndarray] = None
        reference_dem_meta = None
        if ref_dem_filepath is not None:
            reference_dem_arr, reference_dem_meta = load_reference_dem(str(ref_dem_filepath))

        # Module A: Single-View Depth Extraction with Tiled Hann Window Blending
        extractor = get_depth_extractor()
        predicted_depth = extractor.extract_depth(
            rgb_image=rgb_image,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio
        )

        # Module B: Scale & Shift Calibration
        calibrator = get_scale_calibrator()
        calib_result = calibrator.calibrate_depth(
            predicted_depth=predicted_depth,
            geo_meta=geo_meta,
            reference_dem=reference_dem_arr,
            reference_dem_meta=reference_dem_meta,
            min_alt_override=min_alt,
            max_alt_override=max_alt
        )

        # Module C: Output Formatting
        unity_png_name = f"{task_id}_heightmap_unity16.png"
        preview_png_name = f"{task_id}_heightmap_preview8.png"
        colorized_depth_name = f"{task_id}_depth_colorized.png"
        geotiff_name = f"{task_id}_dsm_metric.tif"
        optical_png_name = f"{task_id}_optical_texture.png"
        metadata_json_name = f"{task_id}_metadata.json"

        unity_png_path = str(OUTPUT_DIR / unity_png_name)
        preview_png_path = str(OUTPUT_DIR / preview_png_name)
        colorized_depth_path = str(OUTPUT_DIR / colorized_depth_name)
        geotiff_path = str(OUTPUT_DIR / geotiff_name)
        optical_png_path = str(OUTPUT_DIR / optical_png_name)
        metadata_json_path = str(OUTPUT_DIR / metadata_json_name)

        # 1. Export 16-bit Unity PNG
        OutputFormatter.export_unity_16bit_png(
            metric_dsm=calib_result.metric_dsm,
            output_filepath=unity_png_path
        )

        # 2. Export 8-bit browser preview heightmap (for Three.js DisplacementMap)
        OutputFormatter.export_preview_8bit_png(
            metric_dsm=calib_result.metric_dsm,
            output_filepath=preview_png_path
        )

        # 3. Export colorized depth preview (Turbo colormap, for 2D results panel)
        OutputFormatter.export_colorized_depth_preview(
            metric_dsm=calib_result.metric_dsm,
            output_filepath=colorized_depth_path
        )

        # 4. Export 32-bit Float GeoTIFF
        OutputFormatter.export_geotiff(
            metric_dsm=calib_result.metric_dsm,
            geo_meta=geo_meta,
            output_filepath=geotiff_path
        )

        # 5. Export optical texture PNG
        OutputFormatter.export_optical_texture(
            rgb_image=rgb_image,
            output_filepath=optical_png_path
        )

        # 6. Export Metadata JSON
        meta_payload = OutputFormatter.generate_metadata_json(
            calib_result=calib_result,
            geo_meta=geo_meta,
            img_shape=rgb_image.shape[:2],
            output_filepath=metadata_json_path
        )

        # Construct download URLs
        base_url = "/files"
        download_urls = {
            "heightmap_16bit_png": f"{base_url}/{unity_png_name}",
            "heightmap_8bit_preview_png": f"{base_url}/{preview_png_name}",
            "depth_colorized_png": f"{base_url}/{colorized_depth_name}",
            "optical_texture_png": f"{base_url}/{optical_png_name}",
            "geotiff_dsm_32bit": f"{base_url}/{geotiff_name}",
            "calibration_metadata_json": f"{base_url}/{metadata_json_name}"
        }

        return {
            "task_id": task_id,
            "status": "success",
            "metadata": meta_payload,
            "download_urls": download_urls
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DSM Processing failed: {str(e)}"
        )


@app.get("/download/{filename}", tags=["DSM Extraction Pipeline"])
async def download_file(filename: str):
    """File download endpoint for generated assets."""
    target_path = OUTPUT_DIR / filename
    if not target_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Requested file '{filename}' not found."
        )
    return FileResponse(path=str(target_path), filename=filename)
