import os
import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager

# Bootstrap project root directory into sys.path to prevent ModuleNotFoundError
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from app.config import (
    OUTPUT_DIR,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_KEY,
    SUPPORTED_MODELS,
    resolve_model,
    DEFAULT_MIN_ELEVATION_METERS,
    DEFAULT_MAX_ELEVATION_METERS,
    get_device,
    FRONTEND_DIR
)

from app.utils.image_io import load_optical_image, load_reference_dem
from app.modules.depth_extractor import DepthExtractor
from app.modules.scale_calibrator import ScaleCalibrator
from app.modules.formatter import OutputFormatter
from app.modules.point_cloud_generator import PointCloudGenerator
from app.modules.flood_simulator import simulate_flood


class ModelManager:
    """
    Manages loading, active caching, and switching of Depth Anything V2 model variants.
    Releases inactive model memory before loading a newly requested model to guarantee
    safe RAM usage on CPU / limited hardware.
    """
    def __init__(self):
        self.current_key: Optional[str] = None
        self.extractor: Optional[DepthExtractor] = None

    def get_extractor(self, model_key_or_id: Optional[str] = None) -> Tuple[DepthExtractor, Dict[str, Any]]:
        cfg = resolve_model(model_key_or_id or DEFAULT_MODEL_KEY)
        req_key = cfg["key"]

        # If local model requested but weights file is missing, fallback to Base HF model
        if cfg.get("is_local"):
            weights_file = Path(cfg["id"]) / "model.safetensors"
            if not weights_file.exists():
                print(
                    f"[ModelManager WARNING] Local fine-tuned weights '{weights_file}' not found. "
                    "Falling back to Depth Anything V2 Base from HuggingFace Hub."
                )
                cfg = SUPPORTED_MODELS["depth_anything_v2_base"]
                req_key = cfg["key"]

        # Check if requested model is already active
        if self.extractor is not None and self.current_key == req_key:
            return self.extractor, cfg

        print(f"[ModelManager] Switching model from '{self.current_key}' to '{cfg['name']}' ({cfg['id']})...")
        if self.extractor is not None:
            self.extractor.release()
            self.extractor = None

        self.extractor = DepthExtractor(model_id=cfg["id"])
        self.current_key = req_key
        return self.extractor, cfg

    def get_current_info(self) -> Dict[str, Any]:
        cfg = resolve_model(self.current_key or DEFAULT_MODEL_KEY)
        return {
            "current_key": self.current_key,
            "model_info": cfg,
            "loaded": self.extractor is not None
        }


# Global pipeline components
model_manager = ModelManager()
scale_calibrator: Optional[ScaleCalibrator] = None
task_store: Dict[str, Dict[str, Any]] = {}


class FloodRequest(BaseModel):
    task_id: str = Field(min_length=1)
    water_level: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    Pre-initializes the Depth Anything v2 model and scale calibrator on app startup.
    """
    global scale_calibrator
    print("[FastAPI Startup] Initializing DSM extraction pipeline models...")
    try:
        model_manager.get_extractor(DEFAULT_MODEL_KEY)
        scale_calibrator = ScaleCalibrator(
            default_min_alt=DEFAULT_MIN_ELEVATION_METERS,
            default_max_alt=DEFAULT_MAX_ELEVATION_METERS
        )
        print("[FastAPI Startup] Default model and scale calibrator loaded successfully!")
    except Exception as e:
        print(f"[FastAPI Startup WARNING] Model initialization deferred to first request: {e}")
    
    yield
    
    print("[FastAPI Shutdown] Cleaning up resources...")
    if model_manager.extractor is not None:
        model_manager.extractor.release()


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

# Mount Unity WebGL build (served at /unity-build/*) — optional, present only after a Unity WebGL build
UNITY_BUILD_DIR = Path(__file__).parent.parent / "unity-build"
if UNITY_BUILD_DIR.exists():
    app.mount("/unity-build", StaticFiles(directory=str(UNITY_BUILD_DIR), html=True), name="unity_build")


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to the frontend."""
    return RedirectResponse(url="/app/index.html")


@app.get("/simulator", include_in_schema=False)
async def simulator_redirect():
    """Redirect /simulator to the 3D simulator page."""
    return RedirectResponse(url="/app/simulator.html")


def get_depth_extractor(model_id: Optional[str] = None) -> DepthExtractor:
    extractor, _ = model_manager.get_extractor(model_id)
    return extractor


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
    curr = model_manager.get_current_info()
    return {
        "status": "online",
        "device": device,
        "current_model": curr["model_info"]["name"],
        "current_model_key": curr["current_key"],
        "output_directory": str(OUTPUT_DIR)
    }


@app.get("/api/models", tags=["Model Registry"])
async def list_models() -> Dict[str, Any]:
    """Returns available Depth Anything V2 model variants and active model key."""
    return {
        "status": "success",
        "current_key": model_manager.current_key or DEFAULT_MODEL_KEY,
        "default_key": DEFAULT_MODEL_KEY,
        "models": list(SUPPORTED_MODELS.values())
    }


@app.post("/process", tags=["DSM Extraction Pipeline"])
async def process_image(
    file: UploadFile = File(..., description="Optical input image (PNG, JPG, or GeoTIFF)"),
    ref_dem: Optional[UploadFile] = File(None, description="Optional low-resolution reference DEM GeoTIFF (e.g. SRTM 30m)"),
    model_id: str = Form(DEFAULT_MODEL_KEY, description="Selected model key or HuggingFace ID"),
    min_alt: float = Form(DEFAULT_MIN_ELEVATION_METERS, description="Fallback minimum elevation in meters if uncalibrated"),
    max_alt: float = Form(DEFAULT_MAX_ELEVATION_METERS, description="Fallback maximum elevation in meters if uncalibrated"),
    tile_size: int = Form(512, description="Sliding window tile size in pixels"),
    overlap_ratio: float = Form(0.20, description="Tiled sliding window overlap ratio (0.0 - 0.5)")
) -> Dict[str, Any]:
    """
    POST /process: Runs the automated Digital Surface Model (DSM) extraction pipeline.
    
    Processing Steps:
    1. Module A: Single-View Depth Extraction with 2D Hann window blending.
    2. Module B: Scale & Shift Calibration.
       - GeoTIFF with embedded CRS: attempts SRTM 30m automatic calibration.
       - PNG/JPG (no geospatial metadata): produces Relative DSM (rDSM) output
         directly — no lat/lon input required or used.
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

        # Module A: Single-View Depth Extraction with user-selected model
        extractor, model_cfg = model_manager.get_extractor(model_id)
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
            water_mask=getattr(extractor, "last_water_mask", None),
            min_alt_override=min_alt,
            max_alt_override=max_alt
        )

        print(f"[Calibration] Type: {calib_result.calibration_type}, "
              f"Range: {calib_result.min_elevation:.2f} — {calib_result.max_elevation:.2f} "
              f"({'relative units' if calib_result.calibration_type == 'relative_rdsm' else 'm'})")

        task_store[task_id] = {
            "metric_dsm": calib_result.metric_dsm,
            "geo_meta": geo_meta,
            "metadata_path": None,
        }

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
        meta_payload["model_info"] = {
            "key": model_cfg["key"],
            "name": model_cfg["name"],
            "id": model_cfg["id"],
            "variant": model_cfg.get("variant", "Unknown"),
            "params": model_cfg.get("params", "Unknown"),
            "size_mb": model_cfg.get("size_mb", 0)
        }
        import json
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        task_store[task_id]["metadata_path"] = metadata_json_path

        # Module D: Point Cloud & Surface Mesh (requires open3d + trimesh)
        mesh_glb_name:   str | None = None
        ply_name:        str | None = None

        if PointCloudGenerator.is_available():
            try:
                pc_gen = PointCloudGenerator()
                mesh_glb_name  = f"{task_id}_mesh.glb"
                ply_name       = f"{task_id}_pointcloud.ply"
                mesh_glb_path  = str(OUTPUT_DIR / mesh_glb_name)
                ply_path       = str(OUTPUT_DIR / ply_name)

                pcd, mesh = pc_gen.generate(
                    rgb_image=rgb_image,
                    metric_dsm=calib_result.metric_dsm,
                )
                PointCloudGenerator.export_glb(mesh, mesh_glb_path)
                PointCloudGenerator.export_ply(pcd, ply_path)
            except Exception as mesh_err:
                import traceback
                print(f"[Module D WARNING] Mesh generation failed (pipeline continues): {mesh_err}")
                traceback.print_exc()
                mesh_glb_name = None
                ply_name      = None

        # Construct download URLs
        base_url = "/files"
        download_urls = {
            "heightmap_16bit_png":        f"{base_url}/{unity_png_name}",
            "heightmap_8bit_preview_png": f"{base_url}/{preview_png_name}",
            "depth_colorized_png":        f"{base_url}/{colorized_depth_name}",
            "optical_texture_png":        f"{base_url}/{optical_png_name}",
            "geotiff_dsm_32bit":          f"{base_url}/{geotiff_name}",
            "calibration_metadata_json":  f"{base_url}/{metadata_json_name}",
            # Module D outputs (null when open3d/trimesh not installed)
            "mesh_glb":       f"{base_url}/{mesh_glb_name}" if mesh_glb_name else None,
            "pointcloud_ply": f"{base_url}/{ply_name}"      if ply_name      else None,
        }

        return {
            "task_id": task_id,
            "status": "success",
            "model_used": model_cfg["name"],
            "model_key": model_cfg["key"],
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


@app.post("/simulate-flood", tags=["Flood Simulation"])
async def simulate_flood_endpoint(request: FloodRequest) -> Dict[str, Any]:
    """Run border-connected inundation for a previously processed DSM."""
    task = task_store.get(request.task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown task_id")
    if not np.isfinite(request.water_level):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="water_level must be finite")

    geo_meta = task["geo_meta"]
    transform = geo_meta.transform
    pixel_size = (transform[0], transform[4]) if transform else None
    latitude = None
    if geo_meta.bounds:
        latitude = (geo_meta.bounds[1] + geo_meta.bounds[3]) / 2.0 if geo_meta.crs == "EPSG:4326" else None
    result = simulate_flood(task["metric_dsm"], request.water_level, pixel_size, latitude)
    overlay_name = f"{request.task_id}_flood_overlay.png"
    overlay_path = OUTPUT_DIR / overlay_name
    overlay_path.write_bytes(result.overlay_png)

    flood_metrics = {
        "water_level_meters": result.water_level_meters,
        "flooded_area_km2": result.flooded_area_km2,
        "water_volume_m3": result.water_volume_m3,
        "pixel_area_m2": result.pixel_area_m2,
    }
    metadata_path = task.get("metadata_path")
    if metadata_path:
        import json
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        metadata["flood_simulation"] = flood_metrics
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    return {
        "task_id": request.task_id,
        "status": "success",
        **flood_metrics,
        "overlay_png_url": f"/files/{overlay_name}",
    }


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


if __name__ == "__main__":
    import uvicorn
    print("[Launcher] Starting DepthWizard FastAPI server on http://localhost:8000...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

