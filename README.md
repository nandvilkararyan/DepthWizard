<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Three.js-r165-black?style=for-the-badge&logo=three.js&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

# 🌍 DepthWizard
### AI-Powered 2D Satellite Image → Interactive 3D Terrain

**DepthWizard** converts any single-view satellite or aerial photograph into a fully navigable 3D terrain model with metric elevation data — directly in your browser. No LiDAR, no stereo pairs, no GCP surveys required.

[Live Demo](#running-locally) · [API Docs](http://localhost:8000/docs) · [Architecture](#system-architecture) · [Tech Stack](#tech-stack)

</div>

---

## 🎯 Problem Statement

High-resolution 3D terrain models are critical for disaster management, urban planning, environmental monitoring, and defence applications. However, generating accurate Digital Surface Models (DSMs) traditionally requires:

- Expensive stereo satellite constellations or LiDAR sensors
- Ground control point (GCP) surveys for metric calibration
- Proprietary photogrammetry software (e.g. Pix4D, Agisoft Metashape)
- Days of processing time

**DepthWizard solves this** by applying state-of-the-art monocular depth estimation (Depth Anything V2) to derive metric elevation from a single optical image, calibrating it against freely available reference DEMs, and delivering an interactive 3D flythrough in seconds.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **AI Depth Estimation** | Depth Anything V2 (ViT-Base) extracts relative depth from a single image |
| 📐 **Metric Calibration** | OLS regression against SRTM/ASTER reference DEMs for real-world elevation values |
| 🗺️ **GeoTIFF Support** | Reads CRS, affine transforms, and bounding boxes from georeferenced rasters |
| 🧩 **Tiled Inference** | Sliding-window processing with 2D Hann-window blending eliminates seam artifacts |
| 🌐 **Browser 3D Viewer** | Three.js terrain with satellite texture draping, flythrough camera, orbit controls |
| ⬇️ **Multi-Format Export** | 16-bit Unity heightmap PNG, 32-bit float GeoTIFF, calibration metadata JSON |
| 🎮 **Unity Integration** | Complete C# scripts for Unity URP terrain visualisation and GIS fly camera |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DEPTHWIZARD PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

  INPUT: Optical satellite image (PNG / JPG / GeoTIFF)
         └── Optional: Low-resolution reference DEM (SRTM 30m GeoTIFF)

  ┌──────────────────────────────────────────────────────────────────┐
  │  MODULE A — Single-View Depth Extraction                        │
  │                                                                  │
  │  ┌──────────────┐    Tiled Sliding Window (512×512, 20% overlap)│
  │  │ RGB Image    │──► Depth Anything V2 ViT-Base inference       │
  │  └──────────────┘    2D Hann-window blending → seam-free depth  │
  │                      Output: relative depth map D ∈ ℝ^(H×W)    │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  MODULE B — Scale & Shift Calibration                           │
  │                                                                  │
  │  Georeferenced mode (with reference DEM):                        │
  │    H_metric = α · D_relative + β                                │
  │    Solved by OLS: [α, β] = argmin Σ(H_ref - H_pred)²           │
  │    Validation: RMSE and MAE computed in meters                  │
  │                                                                  │
  │  Non-georeferenced fallback:                                     │
  │    H_metric = normalize(D) → [min_alt, max_alt] meters          │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  MODULE C — Output Formatting                                   │
  │                                                                  │
  │  ├── Unity 16-bit Heightmap PNG  (uint16, 0–65535)              │
  │  ├── Browser 8-bit Preview PNG   (uint8, for Three.js)          │
  │  ├── Turbo Colorized Depth PNG   (false-color 2D preview)       │
  │  ├── 32-bit Float GeoTIFF        (CRS + affine transform)       │
  │  └── Calibration Metadata JSON   (elevation metrics, RMSE, MAE) │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  MODULE D — FastAPI REST Service + Browser Frontend             │
  │                                                                  │
  │  POST /process  → Full pipeline, returns JSON + download URLs  │
  │  GET  /files/*  → Static file server for generated assets       │
  │  GET  /app/*    → Serves browser frontend (Three.js viewer)     │
  │                                                                  │
  │  Frontend:                                                       │
  │  ├── Upload panel (drag-and-drop, advanced options)             │
  │  ├── Processing spinner (step-by-step status)                   │
  │  ├── Results panel (2D previews + metadata cards)               │
  │  └── Three.js 3D Viewer                                         │
  │       ├── PlaneGeometry displaced by heightmap                  │
  │       ├── Optical texture draped on terrain                     │
  │       ├── Auto flythrough camera (orbiting spline)              │
  │       └── Manual OrbitControls on mouse grab                    │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend
| Component | Technology | Version | Purpose |
|---|---|---|---|
| Web Framework | **FastAPI** | ≥ 0.100 | REST API, static file serving, CORS |
| ASGI Server | **Uvicorn** | ≥ 0.20 | Production-grade async server |
| Depth Model | **Depth Anything V2** | Base (ViT-B) | Monocular depth estimation |
| ML Framework | **PyTorch** | ≥ 2.0 | Model inference, GPU acceleration |
| Model Hub | **HuggingFace Transformers** | ≥ 4.40 | Model loading and image processor |
| GIS Library | **Rasterio** | ≥ 1.3 | GeoTIFF I/O, CRS handling, affine transforms |
| Image Processing | **OpenCV** | ≥ 4.8 | Image I/O, colormap, PNG encoding |
| Linear Algebra | **NumPy + SciPy** | ≥ 1.24, ≥ 1.10 | OLS regression, array ops |
| Data Validation | **Pydantic** | ≥ 2.0 | Request/response schemas |

### Frontend
| Component | Technology | Version | Purpose |
|---|---|---|---|
| 3D Rendering | **Three.js** | r165 | WebGL terrain mesh, lighting, camera |
| Camera Controls | **OrbitControls** | Three.js addon | Mouse-driven terrain orbit |
| Module System | **ES Modules + importmap** | Native browser | Zero-build CDN module resolution |
| Styling | **Vanilla CSS** | — | Glassmorphism dark-mode UI |
| Typography | **Google Fonts** (Inter, Outfit) | — | Premium sans-serif typography |

### Unity Integration (Optional — WebGL builds run inside the browser)
| Component | Technology | Purpose |
|---|---|---|
| Render Pipeline | **Unity URP** | Physically-based rendering |
| Terrain Mesh | `TerrainMeshGenerator.cs` | Procedural mesh from 16-bit PNG |
| GIS Camera | `FlyCameraController.cs` | WASD + RMB free-fly with terrain clearance |
| Shader | `HeightDisplacement.hlsl` | Custom vertex displacement shader |
| One-Click Setup | `SetupDepthWizardScene.cs` | Editor utility to scaffold full scene |
| **WebGL Bridge** | `WebGLBridge.cs` + `DepthWizardBridge.jslib` | **postMessage bridge — lets the browser page send terrain data into a Unity WebGL iframe** |

---

## 📁 Repository Structure

```
depthwizard/
│
├── app/                            # FastAPI application
│   ├── __init__.py
│   ├── config.py                   # Global constants, device auto-detection
│   ├── main.py                     # FastAPI routes, lifespan, static mounts
│   ├── modules/
│   │   ├── depth_extractor.py      # Module A: Depth Anything V2 + Hann blending
│   │   ├── scale_calibrator.py     # Module B: OLS scale & shift calibration
│   │   └── formatter.py            # Module C: PNG/GeoTIFF/JSON export
│   └── utils/
│       └── image_io.py             # Rasterio + OpenCV I/O helpers, GeoMetadata
│
├── frontend/                       # Browser-based 3D viewer (served by FastAPI)
│   ├── index.html                  # Single-page app (Unity iframe primary / Three.js fallback)
│   ├── style.css                   # Dark glassmorphism UI
│   └── viewer.js                   # Three.js terrain viewer ES module
│
├── unity/                          # Unity URP integration (compile to unity-build/ for WebGL)
│   ├── Scripts/
│   │   ├── AppManager.cs           # HTTP pipeline integration + LoadFromUrls() for WebGL
│   │   ├── TerrainMeshGenerator.cs # 16-bit heightmap → Unity Mesh
│   │   ├── FlyCameraController.cs  # GIS free-fly camera (WASD + RMB)
│   │   ├── TerrainInspector.cs     # On-screen elevation stats overlay
│   │   └── WebGLBridge.cs          # postMessage ↔ C# bridge for WebGL iframe
│   ├── Plugins/
│   │   └── WebGL/
│   │       └── DepthWizardBridge.jslib  # JS side of the WebGL bridge
│   ├── Editor/
│   │   └── SetupDepthWizardScene.cs# One-click Unity scene scaffolding
│   └── Shaders/
│       └── HeightDisplacement.shader # URP vertex displacement HLSL
│
├── unity-build/                    # ⬅ Unity WebGL output (NOT committed, built locally)
│   └── index.html                  #   Auto-detected by FastAPI → served at /unity-build/*
│
├── output/                         # Generated assets (git-ignored)
├── test_pipeline.py                # Full automated test suite
├── cli_test.py                     # CLI tool for quick pipeline tests
├── requirements.txt                # Python dependency specifications
└── README.md
```

---

## 🎮 Unity WebGL Build (Optional — Activates Premium 3D Viewer)

The browser frontend automatically detects whether a Unity WebGL build is present.
- **Build present** → Unity renders the terrain inside an iframe (CPU-baked mesh, URP shading, full FPS camera)
- **Build absent** → Three.js fallback activates automatically — no action needed

### One-Time Build Steps

> **Requirements:** Unity 2022 LTS or newer with the **WebGL Build Support** module installed.

```
1. Open Unity Hub → Add → select depthwizard/unity/ as the project folder
2. Wait for Unity to import assets and compile shaders

3. Menu → DepthWizard → Setup 3D Elevation Scene
   (This scaffolds the scene hierarchy, wires up AppManager, attaches WebGLBridge)

4. Ensure the "WebGLBridge" GameObject is in the scene with:
   - WebGLBridge.cs attached
   - AppManager reference assigned

5. File → Build Settings → switch Platform to WebGL → click "Switch Platform"

6. Player Settings → Publishing Settings:
   - Compression Format: Disabled  (avoids .br/.gz serving issues on local dev)
   - Strip Engine Code: Off

7. Click Build → choose output folder:  depthwizard/unity-build/
   (The folder name must be exactly "unity-build" at the project root)

8. Start the FastAPI server:
   uvicorn app.main:app --reload

9. Open http://localhost:8000 in your browser
   → The engine badge in the viewer corner will show "🎮 Unity"
   → Upload any image to generate terrain

To switch back to Three.js: rename or delete the unity-build/ folder, then refresh.
```

### Unity Controls (in WebGL mode)
| Key / Input | Action |
|---|---|
| `W / A / S / D` | Fly forward / left / back / right |
| `E` or `Space` | Ascend |
| `Q` or `Shift` | Descend |
| `Right Mouse + Drag` | Look / rotate view |
| `Scroll Wheel` | Adjust fly speed |

---


## 🚀 Running Locally

### Prerequisites

- Python 3.10 or later
- CUDA-capable GPU (recommended) — CPU inference supported but significantly slower
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/nandvilkararyan/DepthWizard.git
cd DepthWizard
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will automatically download the `depth-anything/Depth-Anything-V2-Base-hf` model weights (~400 MB) from HuggingFace Hub. Ensure you have an internet connection.

### 4. Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the App

Navigate to **http://localhost:8000** in any modern browser (Chrome 89+, Firefox 108+, Edge 89+).

---

## 📡 API Reference

### `POST /process`

Runs the full DSM extraction pipeline.

**Request:** `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | ✅ | — | Input optical image (PNG, JPG, GeoTIFF) |
| `ref_dem` | File | ❌ | — | Reference DEM GeoTIFF (e.g. SRTM 30m) for OLS calibration |
| `min_alt` | float | ❌ | `0.0` | Minimum elevation fallback (meters) |
| `max_alt` | float | ❌ | `500.0` | Maximum elevation fallback (meters) |
| `tile_size` | int | ❌ | `512` | Sliding window tile size in pixels |
| `overlap_ratio` | float | ❌ | `0.20` | Tile overlap fraction (0.0 – 0.5) |

**Response:** `application/json`

```json
{
  "task_id": "dsm_8f29ab4c12",
  "status": "success",
  "metadata": {
    "scene_geometry": {
      "width_pixels": 1024,
      "height_pixels": 1024,
      "aspect_ratio": 1.0
    },
    "elevation_metrics": {
      "min_elevation_meters": 120.45,
      "max_elevation_meters": 485.80,
      "elevation_range_meters": 365.35
    },
    "calibration_parameters": {
      "scale_alpha": 4.125,
      "shift_beta_meters": 115.20,
      "calibration_type": "reference_dem",
      "rmse_meters": 1.84,
      "mae_meters": 1.42
    },
    "geospatial_metadata": {
      "is_georeferenced": true,
      "crs": "EPSG:4326",
      "bounds": [72.8, 18.9, 73.0, 19.1],
      "transform": [0.0002, 0, 72.8, 0, -0.0002, 19.1]
    }
  },
  "download_urls": {
    "heightmap_16bit_png":      "/files/dsm_8f29ab4c12_heightmap_unity16.png",
    "heightmap_8bit_preview_png": "/files/dsm_8f29ab4c12_heightmap_preview8.png",
    "depth_colorized_png":      "/files/dsm_8f29ab4c12_depth_colorized.png",
    "optical_texture_png":      "/files/dsm_8f29ab4c12_optical_texture.png",
    "geotiff_dsm_32bit":        "/files/dsm_8f29ab4c12_dsm_metric.tif",
    "calibration_metadata_json": "/files/dsm_8f29ab4c12_metadata.json"
  }
}
```

### `GET /health`

Returns system diagnostics.

```json
{ "status": "online", "device": "cuda", "model_id": "depth-anything/Depth-Anything-V2-Base-hf", "output_directory": "/app/output" }
```

### `GET /download/{filename}`

Direct file download for any asset in the `output/` directory.

---

## 🧮 Algorithm Deep Dive

### Module A — Tiled Hann-Window Inference

Large satellite images (> 512×512 px) are processed via a sliding-window approach:

1. The image is divided into overlapping `tile_size × tile_size` patches with `overlap_ratio` overlap.
2. Each patch is independently inferred by Depth Anything V2.
3. A **2D Hann (raised-cosine) window** weights each tile's contribution:
   ```
   w(i, j) = 0.5 × (1 − cos(2π(i+0.5)/H)) × 0.5 × (1 − cos(2π(j+0.5)/W))
   ```
4. Weighted predictions are accumulated and normalised:
   ```
   D_blended(x, y) = Σ D_tile(x, y) × w(x, y) / Σ w(x, y)
   ```

This eliminates the blocky seam artifacts that arise from naive tile stitching.

### Module B — OLS Metric Calibration

The relative depth output of Depth Anything V2 has no absolute scale. Calibration fits a linear mapping:

```
H_metric = α × D_relative + β
```

**Georeferenced mode** (reference DEM provided): The reference DEM is reprojected and resampled to match the optical image's spatial extent. OLS regression is solved over all co-registered valid pixels:

```
[α, β] = (XᵀX)⁻¹ Xᵀ y    where X = [D_relative | 1],  y = H_reference
```

**Validation metrics** are computed against held-out pixels:
- RMSE = √(mean((H_pred − H_ref)²))  
- MAE  = mean(|H_pred − H_ref|)

**Fallback mode** (no reference DEM): Min-max normalisation maps relative depth to a user-specified `[min_alt, max_alt]` elevation range.

---

## 🌐 Frontend — 3D Terrain Viewer

The browser frontend is a zero-build-step single-page application served directly by FastAPI at `/app/index.html`.

### Key interactions

| Action | Behaviour |
|---|---|
| Upload image | Drag-and-drop or click; calls `POST /process` |
| Load Demo | Loads pre-computed example outputs instantly |
| Auto Flythrough | Camera orbits the terrain on a parametric ellipse |
| Click & Drag | Pauses flythrough; switches to free OrbitControls |
| Elevation Scale slider | Live-adjusts Three.js `displacementScale` |
| Sun Angle slider | Rotates the directional light in real-time |
| Fly Speed slider | Controls camera angular velocity |

### Three.js terrain construction

```javascript
// 256×256 segment plane displaced by 8-bit heightmap
const geo = new THREE.PlaneGeometry(8, 8, 256, 256);
const mat = new THREE.MeshStandardMaterial({
  map: opticalTexture,           // satellite photo
  displacementMap: heightTexture, // 8-bit grayscale PNG
  displacementScale: dispScale,   // scaled from elevation_range_meters
  roughness: 0.88,
  metalness: 0.04,
});
```

---

## 🎮 Unity Integration

For teams requiring a native desktop experience, the `unity/` directory contains a complete Unity URP integration:

| Script | Purpose |
|---|---|
| `AppManager.cs` | Calls the FastAPI `/process` endpoint from within Unity; downloads and applies heightmap and texture at runtime |
| `TerrainMeshGenerator.cs` | Decodes a 16-bit PNG into vertex positions for a procedural Unity Mesh |
| `FlyCameraController.cs` | WASD + RMB free-fly camera with terrain-clearance raycasting |
| `TerrainInspector.cs` | On-screen IMGUI overlay showing elevation at cursor position |
| `HeightDisplacement.shader` | Custom URP HLSL shader for GPU-side vertex displacement |
| `SetupDepthWizardScene.cs` | Editor one-click setup — builds full scene hierarchy automatically |

**One-click setup:** In the Unity Editor, go to `DepthWizard → Setup 3D Elevation Scene`.

---

## 🔬 Running Tests

```bash
# Full automated pipeline test suite
python test_pipeline.py

# CLI quick test (processes a local image)
python cli_test.py --input path/to/satellite.tif
```

---

## 📋 Requirements

```
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.40.0
rasterio>=1.3.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
scipy>=1.10.0
pillow>=9.5.0
pydantic>=2.0.0
python-multipart>=0.0.6
```

---

## 🗺️ Roadmap

- [ ] WebSocket / SSE streaming for real-time model progress updates
- [ ] Integration with Copernicus Open Access Hub for automatic satellite tile fetching
- [ ] Support for Depth Anything V3 / metric depth models (ZoeDepth, UniDepth)
- [ ] Batch processing API endpoint for multi-tile mosaic generation
- [ ] Contour line and slope/aspect raster export
- [ ] GCP (ground control points) upload for precision OLS calibration
- [ ] Point cloud (.LAS / .PLY) export from the calibrated DSM

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Depth Anything V2](https://depth-anything.github.io/) — Lihe Yang et al., 2024
- [HuggingFace Transformers](https://huggingface.co/transformers) — model hosting and inference
- [Three.js](https://threejs.org/) — WebGL 3D rendering engine
- [Rasterio](https://rasterio.readthedocs.io/) — geospatial raster I/O
- [FastAPI](https://fastapi.tiangolo.com/) — modern Python web framework

---

<div align="center">
Built for the <strong>ISRO DepthWizard Hackathon</strong> · 2026
</div>
