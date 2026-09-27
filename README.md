<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Three.js-r165-black?style=for-the-badge&logo=three.js&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

# 🌍 DepthWizard
### AI-Powered Monocular Satellite Imagery → Calibrated 3D Terrain & Digital Surface Models

**DepthWizard** transforms single-view satellite and aerial photographs into fully calibrated Digital Surface Models (DSMs), 3D point clouds, watertight surface meshes, and interactive 3D simulations directly inside your browser. No LiDAR, stereo pairs, or manual Ground Control Point (GCP) surveys required.

[Quick Start](#-quick-start) · [System Architecture](#-system-architecture) · [Features & UI](#-features--interactive-interfaces) · [API Reference](#-api-reference) · [Algorithms](#-algorithm-deep-dive) · [Tech Stack](#-tech-stack)

</div>

---

## 🎯 Problem Statement

High-resolution 3D terrain and Digital Surface Models (DSMs) are critical for disaster management, urban planning, defense, and environmental monitoring. Traditionally, generating accurate elevation models requires:

- Expensive stereo satellite constellations or airborne LiDAR missions
- Tedious physical GCP surveys for metric datum alignment
- Heavy proprietary photogrammetry suites (e.g., Pix4D, Agisoft Metashape)
- Hours to days of multi-view matching computation

**DepthWizard** breaks this bottleneck by pairing state-of-the-art monocular foundation models (**Depth Anything V2**) with automated open DEM datum calibration (Copernicus 30m / SRTM via Open-Meteo & Open-Elevation). A single optical image yields metric elevation rasters, 3D point clouds, surface meshes, and a real-time 3D simulation environment in seconds.

---

## ✨ Key Capabilities

| Capability | Technical Implementation |
|---|---|
| 🤖 **Multi-Model Inference Engine** | Depth Anything V2 (Small / Base / Large) + local fine-tuned `model.safetensors` with water-depth suppression |
| 🧩 **Tiled Sliding-Window Inference** | 512×512 sliding windows with 2D Hann (raised cosine) blending to eliminate seam artifacts on arbitrary raster sizes |
| 📐 **Metric Datum Calibration** | Automatic regional 30m DEM fetching (Copernicus/SRTM) with OLS & Huber robust regression; adaptive rDSM fallback |
| 🌐 **Dedicated 3D Simulator** | Real-time Three.js WebGL terrain with satellite texture draping, GLB Poisson mesh overlay, and dynamic solar lighting |
| 🎥 **4 Adaptive Camera Rigs** | **Orbit Target** (interactive inspection), **Drone FPV** (6-DOF flight), **Nadir Orthographic** (survey view), and **Ground Walk FPS** (first-person pedestrian walkthrough with pointer lock and terrain collision) |
| 🌊 **Hydrological Simulation** | Connected border inundation analysis with real-time flood area ($km^2$) and water volume ($m^3$) computation |
| 📦 **Production GIS & 3D Export** | 16-bit uint16 heightmaps (PNG), 32-bit float GeoTIFFs (with embedded CRS & affine transform), GLB surface meshes, PLY point clouds, and metadata JSON |

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DEPTHWIZARD PIPELINE                           │
└────────────────────────────────────────────────────────────────────────┘

  INPUT: Optical satellite / aerial raster (GeoTIFF, PNG, JPG)
         └── Optional: Reference DEM GeoTIFF (or automated regional 30m DEM lookup)

  ┌─────────────────────────────────────────────────────────────────────┐
  │ MODULE A — Single-View Depth Extraction                             │
  │                                                                     │
  │ ┌──────────────┐    Tiled Sliding Window (512×512, 20% overlap)    │
  │ │ RGB Image    │──► Depth Anything V2 ViT inference                │
  │ └──────────────┘    2D Hann-window cosine blending → seam-free depth│
  │                     Intelligent water-body suppression              │
  │                     Output: Relative depth map D ∈ ℝ^(H×W)          │
  └─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ MODULE B — Scale & Shift Metric Calibration                         │
  │                                                                     │
  │ Georeferenced Mode:                                                 │
  │   - Automated regional 30m DEM fetching (Copernicus DEM via API)    │
  │   - Coordinate-aware reprojection and bilinear resampling           │
  │   - OLS & Huber Robust Linear Fit: H_metric = α · D_relative + β   │
  │   - Validation metrics: RMSE, MAE, and Pearson correlation          │
  │                                                                     │
  │ Non-Georeferenced Mode:                                             │
  │   - Adaptive outlier suppression & dynamic relative rDSM scaling    │
  └─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ MODULE C — Output Formatting & Geospatial Packaging                │
  │                                                                     │
  │ ├── 16-bit Heightmap PNG  (uint16, 0–65535, for 3D game engines)    │
  │ ├── 8-bit Preview PNG     (uint8, optimized for Three.js WebGL)    │
  │ ├── Turbo Colorized PNG   (scientific hypsometric preview)          │
  │ ├── 32-bit Float GeoTIFF  (GeoTIFF with EPSG CRS & affine matrix)   │
  │ └── Metadata JSON         (elevation stats, GSD, RMSE, MAE)         │
  └─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ MODULE D — 3D Reconstruction, Point Cloud & WebGL Simulation        │
  │                                                                     │
  │ ├── 3D Point Cloud (.PLY) (unprojected 3D coordinates + RGB colors) │
  │ ├── Watertight Mesh (.GLB)(Poisson surface reconstruction / Delaunay│
  │ └── Interactive Three.js WebGL Simulator:                           │
  │     ├── Displaced PlaneGeometry with satellite texture draping      │
  │     ├── Co-registered GLB Poisson mesh & cyan wireframe overlay     │
  │     ├── 4 Camera Rigs (Orbit, Drone FPV, Nadir Ortho, Ground Walk)  │
  │     ├── Real-time Solar Angle & dynamic shadows                     │
  │     └── Hydrological flood simulation & depth profiling             │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Interactive Interfaces

DepthWizard features two browser-based web applications served directly by the FastAPI backend:

### 1. DepthWizard Studio (`/app/index.html`)
- **Dual Raster Comparison**: Side-by-side interactive split comparison between the original optical raster and the inferred depth / DSM.
- **Model Engine Selector**: Toggle seamlessly between *Depth Anything V2 Small*, *Base*, *Large*, or *DepthWizard Fine-Tuned*.
- **Datum Tie-In & Calibration Inspector**: Real-time readouts of min/max elevation, elevation range, scale factor ($\alpha$), shift ($\beta$), RMSE, MAE, and sample correlation.
- **Single-Click Generation**: Computes depth maps, DSM GeoTIFFs, 16-bit heightmaps, 3D point clouds, and GLB surface meshes in a unified pipeline.

### 2. DepthWizard 3D Simulator (`/app/simulator.html`)
- **4 Camera Rigs**:
  - 🔄 **Orbit Target**: Standard orbit, pan, and zoom controls around the terrain center.
  - 🚁 **Drone FPV**: 6-DOF flight camera using `W/A/S/D` for directional thrust and `Q/E` for altitude elevation.
  - 📐 **Orthographic / Nadir**: 90° top-down survey view for planar measurements and aerial inspection.
  - 🚶 **Ground Walk (FPS)**: First-person pedestrian walkthrough at realistic human eye height (~1.7 m) and walking speed (~1.4 m/s). Features pointer-lock mouse look, `W/A/S/D` movement, `Shift` sprint, head bobbing, and $O(1)$ bilinear terrain collision raycasting.
- **Mesh Overlay & Wireframe Mode**: Co-registered Poisson surface mesh overlay with adjustable opacity and high-visibility wireframe modes.
- **Dynamic Solar Simulation**: Time-of-day slider adjusting the virtual sun angle (azimuth and elevation) with dynamic shadow casting.
- **Hydrological Inundation**: Interactive water-level slider modeling border-connected flooding with live area ($km^2$) and water volume ($m^3$) readouts.
- **Live Telemetry HUD**: Displays real-time altitude, GSD, eye height, heading, pitch, and stable 60 FPS performance monitoring.

---

## 📁 Repository Structure

```
depthwizard/
│
├── app/                                # FastAPI application & pipeline backend
│   ├── __init__.py
│   ├── config.py                       # Model registry, directory paths, device detection
│   ├── main.py                         # REST API endpoints, static file mounts, lifecycle
│   │
│   ├── depthwizard_model/              # Fine-tuned model directory
│   │   ├── README.md                   # Instructions & GitHub Release asset link
│   │   ├── config.json                 # Model architecture configuration
│   │   └── preprocessor_config.json    # Image preprocessor parameters
│   │   # model.safetensors             # Optional local fine-tuned weights (~390 MB)
│   │
│   ├── modules/
│   │   ├── depth_extractor.py          # Module A: Depth Anything V2 inference & Hann blending
│   │   ├── scale_calibrator.py         # Module B: 30m DEM fetching, OLS/Huber calibration, rDSM
│   │   ├── formatter.py                # Module C: 16-bit PNG, float32 GeoTIFF, metadata JSON
│   │   ├── point_cloud_generator.py    # Module D: PLY point cloud & GLB mesh reconstruction
│   │   └── flood_simulator.py          # Border-connected hydrological flood simulation
│   │
│   └── utils/
│       ├── __init__.py
│       └── image_io.py                 # Rasterio geospatial I/O, GeoMetadata parsing
│
├── frontend/                           # Client-side WebGL application (Three.js)
│   ├── index.html                      # DepthWizard Studio interface
│   ├── studio.js                       # Studio UI controller & API orchestration
│   ├── simulator.html                  # Standalone 3D Terrain Simulator
│   ├── simulator.js                    # Simulator UI controller, camera rigs, telemetry HUD
│   ├── viewer.js                       # Three.js 3D engine (terrain, mesh, walk mode, lighting)
│   ├── cesium_viewer.js                # Optional CesiumJS geospatial viewer
│   └── style.css                       # Styling & glassmorphism theme
│
├── output/                             # Generated pipeline assets (git-ignored)
│   └── .gitkeep
│
├── test_pipeline.py                    # Comprehensive automated test suite
├── cli_test.py                         # Command-line utility for offline batch processing
├── test_mt_st_helens.py                # Quick sample image test script
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Git ignore rules for virtualenvs, caches & outputs
├── LICENSE                             # MIT License
└── README.md                           # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (Python 3.10 to 3.12 recommended)
- **CUDA-capable GPU** (optional, recommended for fast inference; CPU inference fully supported)
- **Modern Browser** (Chrome 90+, Edge 90+, Firefox 100+ with WebGL enabled)

### 1. Clone the Repository
```bash
git clone https://github.com/nandvilkararyan/DepthWizard.git
cd DepthWizard
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note on Model Weights:**
> - By default, the application will automatically download the standard `depth-anything/Depth-Anything-V2-Base-hf` weights (~400 MB) from Hugging Face Hub upon first run.
> - To use the optional **DepthWizard Fine-Tuned** model (`model.safetensors`, ~390 MB), download it from the [GitHub Release](https://github.com/AdeshSrivastava-06/DepthWizard/releases/download/v1.0.0/model.safetensors) and place it inside `app/depthwizard_model/model.safetensors`.
> - If `model.safetensors` is not present, the server automatically falls back to Depth Anything V2 Base with zero interruption.

### 4. Launch the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open DepthWizard
Open your browser and navigate to:
- **DepthWizard Studio**: [http://localhost:8000/app/index.html](http://localhost:8000/app/index.html)
- **DepthWizard 3D Simulator**: [http://localhost:8000/app/simulator.html](http://localhost:8000/app/simulator.html)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Reference

### `POST /process`
Executes the end-to-end depth extraction, calibration, formatting, and 3D mesh reconstruction pipeline.

- **Request**: `multipart/form-data`
  - `file`: Input satellite or aerial image (GeoTIFF, PNG, or JPG) — **Required**
  - `ref_dem`: Reference DEM GeoTIFF for OLS calibration — *Optional*
  - `model_id`: Model key (`depthwizard_finetuned`, `depth_anything_v2_small`, `depth_anything_v2_base`, `depth_anything_v2_large`) — *Default: `depthwizard_finetuned`*
  - `min_alt`: Minimum elevation override in meters — *Optional, default: `0.0`*
  - `max_alt`: Maximum elevation override in meters — *Optional, default: `100.0`*
  - `tile_size`: Sliding window tile size in pixels — *Default: `512`*
  - `overlap_ratio`: Tile overlap fraction ($0.0 - 0.5$) — *Default: `0.20`*

- **Response**: `application/json`
```json
{
  "task_id": "dsm_3b7d939682",
  "status": "success",
  "metadata": {
    "scene_geometry": {
      "width_pixels": 1024,
      "height_pixels": 1024,
      "aspect_ratio": 1.0
    },
    "elevation_metrics": {
      "min_elevation_meters": 0.0,
      "max_elevation_meters": 270.5,
      "elevation_range_meters": 270.5,
      "suggested_disp_scale": 0.6
    },
    "calibration": {
      "scale_alpha": 3.84,
      "shift_beta_meters": 12.5,
      "calibration_type": "SRTM_30m_AUTOMATED",
      "rmse_meters": 2.14,
      "mae_meters": 1.68
    },
    "geospatial_metadata": {
      "is_georeferenced": true,
      "crs": "EPSG:4326",
      "bounds": [77.0, 28.0, 77.1, 28.1]
    }
  },
  "download_urls": {
    "heightmap_16bit_png": "/files/dsm_3b7d939682_heightmap_unity16.png",
    "heightmap_8bit_preview_png": "/files/dsm_3b7d939682_heightmap_preview8.png",
    "depth_colorized_png": "/files/dsm_3b7d939682_depth_colorized.png",
    "optical_texture_png": "/files/dsm_3b7d939682_optical_texture.png",
    "geotiff_dsm_32bit": "/files/dsm_3b7d939682_dsm_metric.tif",
    "calibration_metadata_json": "/files/dsm_3b7d939682_metadata.json",
    "mesh_glb": "/files/dsm_3b7d939682_mesh.glb",
    "pointcloud_ply": "/files/dsm_3b7d939682_pointcloud.ply"
  }
}
```

### `POST /simulate-flood`
Executes border-connected hydrological flood simulation for a processed DSM.

- **Request**: `application/json`
```json
{
  "task_id": "dsm_3b7d939682",
  "water_level": 45.0
}
```
- **Response**: `application/json`
```json
{
  "task_id": "dsm_3b7d939682",
  "status": "success",
  "water_level_meters": 45.0,
  "flooded_area_km2": 3.42,
  "water_volume_m3": 15420000.0,
  "overlay_png_url": "/files/dsm_3b7d939682_flood_overlay.png"
}
```

### `GET /health`
Returns system status, active compute device (`cuda`, `mps`, or `cpu`), and loaded model configuration.

### `GET /download/{filename}`
Streams generated artifacts from the `output/` directory for download.

---

## 🧮 Algorithm Deep Dive

### 1. Sliding-Window 2D Hann Cosine Blending
To prevent GPU Out-Of-Memory (OOM) errors and eliminate seam artifacts across large satellite rasters, images are tiled into overlapping patches and blended using a 2D Hann window:

$$w(i, j) = 0.5 \left(1 - \cos\frac{2\pi (i + 0.5)}{H}\right) \times 0.5 \left(1 - \cos\frac{2\pi (j + 0.5)}{W}\right)$$

$$D_{\text{blended}}(x, y) = \frac{\sum_k D_k(x, y) \cdot w_k(x, y)}{\sum_k w_k(x, y)}$$

### 2. Multi-Strategy Metric Scale Calibration
Monocular depth estimation yields scale-ambiguous relative depth $D_{\text{rel}}$. DepthWizard recovers real metric elevation $H_{\text{metric}}$ via:

- **Automated Regional SRTM/Copernicus Fetching**: Extracts bounding coordinates from georeferenced rasters and retrieves regional 30m DEM elevation patches via Open-Meteo or Open-Elevation APIs.
- **Ordinary Least Squares (OLS) & Huber Regression**:
  $$\min_{\alpha, \beta} \sum_i \rho\left(H_{\text{ref}, i} - (\alpha \cdot D_{\text{rel}, i} + \beta)\right)$$
- **Relative rDSM Fallback**: For unreferenced optical imagery, dynamic percentile clamping ($P_{0.1}$ to $P_{99.9}$) with adaptive outlier suppression normalizes topography smoothly across relative metric units.

### 3. Surface Reconstruction (Open3D / SciPy)
- **Point Cloud Generation**: Dense unprojection of $(X, Y, Z)$ coordinates with RGB pixel color binding into binary `.PLY`.
- **Poisson Surface Reconstruction**: Screened Poisson reconstruction calculates surface normal vectors to generate continuous, watertight triangular surface meshes exported as standard `.GLB`.

---

## 🛠️ Tech Stack

### Backend
- **FastAPI**: Asynchronous high-performance REST API
- **PyTorch**: Deep learning inference and CUDA GPU acceleration
- **Transformers**: Hugging Face model loading and image processing
- **Rasterio & GDAL**: Geospatial raster I/O, CRS reprojection, affine transforms
- **Trimesh & Open3D**: 3D mesh generation, coordinate transforms, GLB export
- **OpenCV & NumPy & SciPy**: Computer vision, numerical linear algebra, OLS calibration
- **Requests**: Resilient regional DEM API communication with offline fallbacks

### Frontend
- **Three.js (r165)**: WebGL 3D rendering engine, custom shaders, and lighting
- **OrbitControls**: Mouse-driven orbit and inspection controls
- **Native ES Modules**: Fast, zero-build client-side modular architecture
- **HTML5 Canvas & Pointer Lock API**: True first-person mouse-look and keyboard flight

---

## 🔬 Running Tests

The repository includes a comprehensive automated test suite verifying all pipeline modules:

```bash
# Run the full regression test suite (6 tests)
python -m unittest test_pipeline.py

# Run standalone CLI test on any image
python cli_test.py --input path/to/image.png
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">
Built for the <strong>ISRO DepthWizard Project</strong>
</div>
