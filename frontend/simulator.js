/**
 * DepthWizard — Playable 3D Simulator (Page 2) Controller
 * ========================================================
 * High-performance 3D terrain flight simulator and geospatial inspection engine:
 *   - Three.js WebGL terrain rendering with DisplacementMapping
 *   - Co-registered GLB surface mesh overlay (Poisson / Delaunay)
 *   - Dynamic flight telemetry (Compass, Altitude AGL, Airspeed, Pitch/Roll)
 *   - Keyboard flight controls (WASD, QE, Space)
 *   - Interactive 2D Cross-Section Profile [Line A-B] sampled from DSM
 *   - Water flood simulation & inundation risk calculation
 *   - CesiumJS photorealistic globe bridge
 */

import * as THREE from "three";
import {
  initViewer,
  loadTerrain,
  loadMeshOverlay,
  setElevationScale,
  setSunAngle,
  setFlySpeed,
  toggleFlythrough,
  setFlythrough,
  resetCamera,
  setCameraNadir,
  setTerrainTexture,
  setKeyDown,
  setKeyUp,
  setMeshOpacity,
  setMeshWireframe,
  setTerrainWireframe,
  toggleMeshOverlay,
  initFloodSimulator,
  setFloodLevel,
  toggleFloodVisible,
  setFloodVisible,
  animateFloodRising,
  stopFloodAnimation,
  getFloodState,
  forceResize,
} from "/app/viewer.js";

import {
  initCesiumViewer,
  loadCesiumModel,
  setCesiumLandmark,
  setCesiumDrapeMode,
  setCesiumSolarTime,
  updateCesiumModelTransform,
  toggleCesiumMeasurement,
  clearCesiumMeasurement,
} from "/app/cesium_viewer.js";

(function () {
  'use strict';

  // ── DOM References ────────────────────────────────────────────────────────
  const canvasWrap = document.getElementById('viewport-canvas');
  const threeCanvas = document.getElementById('threejs-canvas');
  const cesiumContainer = document.getElementById('cesium-container');

  // Flight Telemetry HUD
  const hudHeading = document.getElementById('sim-heading');
  const hudAltitude = document.getElementById('sim-altitude');
  const hudAirspeed = document.getElementById('sim-airspeed');
  const hudAttitude = document.getElementById('sim-attitude');
  const hudFps = document.getElementById('sim-fps');
  const hudCamPos = document.getElementById('sim-cam-pos');
  const hudDataset = document.getElementById('header-dataset-name');

  // Render Mode & Sunlight Buttons
  const btnModeOrtho = document.getElementById('btn-mode-ortho');
  const btnModeWireframe = document.getElementById('btn-mode-wireframe');
  const btnModeHeatmap = document.getElementById('btn-mode-heatmap');
  const btnModeIsolines = document.getElementById('btn-mode-isolines');
  const btnEngineThree = document.getElementById('btn-engine-three');
  const btnEngineCesium = document.getElementById('btn-engine-cesium');

  const sunSlider = document.getElementById('sun-slider');
  const sunLabel = document.getElementById('sun-label');

  // Camera Rig
  const btnCamFpv = document.getElementById('btn-cam-fpv');
  const btnCamOrbit = document.getElementById('btn-cam-orbit');
  const btnCamNadir = document.getElementById('btn-cam-nadir');
  const btnCamReset = document.getElementById('btn-cam-reset');

  // Sliders & Controls
  const elevSlider = document.getElementById('elev-slider');
  const elevVal = document.getElementById('elev-val');
  const speedSlider = document.getElementById('speed-slider');
  const speedVal = document.getElementById('speed-val');

  // Mesh overlay controls
  const meshToggle = document.getElementById('mesh-toggle');
  const meshWireframeToggle = document.getElementById('mesh-wireframe-toggle');
  const meshOpacitySlider = document.getElementById('mesh-opacity-slider');
  const meshOpacityVal = document.getElementById('mesh-opacity-val');

  // Flood Controls
  const floodToggle = document.getElementById('flood-toggle');
  const floodSlider = document.getElementById('flood-slider');
  const floodVal = document.getElementById('flood-val');
  const floodSubmerged = document.getElementById('flood-submerged-val');
  const floodDepth = document.getElementById('flood-depth-val');
  const floodRiskBadge = document.getElementById('flood-risk-badge');
  const btnFloodAnimate = document.getElementById('btn-flood-animate');

  // Download links
  const dlHeightmap = document.getElementById('dl-heightmap');
  const dlGeotiff = document.getElementById('dl-geotiff');
  const dlMesh = document.getElementById('dl-mesh');
  const dlPointcloud = document.getElementById('dl-pointcloud');
  const dlMetadata = document.getElementById('dl-metadata');

  // ── State ─────────────────────────────────────────────────────────────────
  let activeTask = null;
  let activeEngine = 'threejs'; // 'threejs' or 'cesium'
  let activeCameraMode = 'orbit'; // 'fpv', 'orbit', 'nadir'
  let isFlying = false;
  let isFloodAnimating = false;
  let activeRenderMode = 'wireframe'; // Default to wireframe switched on

  let frameCount = 0;
  let lastTime = performance.now();
  let currentFps = 60;

  // Keyboard navigation vector
  const keysDown = {};

  // ── Initialization ────────────────────────────────────────────────────────
  async function initSimulator() {
    setupCanvas();
    setupEventListeners();
    startTelemetryLoop();
    await resolveAndLoadTask();
  }

  function setupCanvas() {
    if (!threeCanvas) return;
    initViewer(threeCanvas);

    // Force the renderer to adopt the canvas's real pixel dimensions.
    // The canvas uses CSS 'w-full h-full' which resolves after layout,
    // so we stagger resize calls to catch whenever the browser finishes painting.
    forceResize();                             // immediate (may be 0×0 — viewer guards)
    setTimeout(() => forceResize(), 50);       // after first microtask flush
    setTimeout(() => forceResize(), 300);      // after typical Tailwind / flex layout

    // Also re-sync on window resize
    window.addEventListener('resize', () => forceResize());
  }

  // ── Resolve and Load Task ─────────────────────────────────────────────────
  async function resolveAndLoadTask() {
    const urlParams = new URLSearchParams(window.location.search);
    const taskId = urlParams.get('task_id');

    let taskData = null;

    // 1. Check sessionStorage first (contains live model output from studio)
    try {
      const stored = sessionStorage.getItem('depthwizard_current_task');
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed && (!taskId || parsed.task_id === taskId)) {
          taskData = parsed;
        }
      }
    } catch (_) {}

    // 2. Fetch from metadata endpoint if not in sessionStorage
    if (!taskData && taskId) {
      try {
        const resp = await fetch(`/files/${taskId}_metadata.json`);
        if (resp.ok) {
          const meta = await resp.json();
          taskData = {
            task_id: taskId,
            status: 'success',
            metadata: meta,
            download_urls: {
              heightmap_16bit_png: `/files/${taskId}_heightmap_unity16.png`,
              heightmap_8bit_preview_png: `/files/${taskId}_heightmap_preview8.png`,
              depth_colorized_png: `/files/${taskId}_depth_colorized.png`,
              optical_texture_png: `/files/${taskId}_optical_texture.png`,
              geotiff_dsm_32bit: `/files/${taskId}_dsm_metric.tif`,
              calibration_metadata_json: `/files/${taskId}_metadata.json`,
              mesh_glb: `/files/${taskId}_mesh.glb`,
              pointcloud_ply: `/files/${taskId}_pointcloud.ply`,
            }
          };
        }
      } catch (err) {
        console.warn('[Simulator] Could not fetch task metadata:', err);
      }
    }

    // 3. Fallback to default demo terrain if no task exists
    if (!taskData) {
      taskData = getFallbackDemoTask();
    }

    activeTask = taskData;
    await loadTaskScene(taskData);
  }

  function getFallbackDemoTask() {
    return {
      task_id: "dsm_3b7d939682",
      status: "success",
      metadata: {
        scene_geometry: { width_pixels: 1024, height_pixels: 1024, aspect_ratio: 1.0 },
        elevation_metrics: {
          min_elevation_meters: 0.0,
          max_elevation_meters: 270.5,
          elevation_range_meters: 270.5,
          suggested_disp_scale: 0.6
        },
        calibration: {
          min_elevation_m: 0.0,
          max_elevation_m: 270.5,
          elevation_range_m: 270.5,
          type: "calibrated"
        },
        model_info: {
          name: "DepthWizard Fine-Tuned (model.safetensors)"
        }
      },
      download_urls: {
        heightmap_16bit_png: "/files/dsm_3b7d939682_heightmap_unity16.png",
        heightmap_8bit_preview_png: "/files/dsm_3b7d939682_heightmap_preview8.png",
        depth_colorized_png: "/files/dsm_3b7d939682_depth_colorized.png",
        optical_texture_png: "/files/dsm_3b7d939682_optical_texture.png",
        geotiff_dsm_32bit: "/files/dsm_3b7d939682_dsm_metric.tif",
        calibration_metadata_json: "/files/dsm_3b7d939682_metadata.json",
        mesh_glb: "/files/dsm_3b7d939682_mesh.glb",
        pointcloud_ply: "/files/dsm_3b7d939682_pointcloud.ply",
      }
    };
  }

  async function loadTaskScene(data) {
    const urls = data.download_urls || {};
    const meta = data.metadata || {};
    const elevMetrics = meta.elevation_metrics || {};

    const minElev = elevMetrics.min_elevation_meters !== undefined ? elevMetrics.min_elevation_meters : (meta.calibration?.min_elevation_m ?? 0.0);
    const maxElev = elevMetrics.max_elevation_meters !== undefined ? elevMetrics.max_elevation_meters : (meta.calibration?.max_elevation_m ?? 100.0);

    if (hudDataset) {
      hudDataset.textContent = `${data.task_id} • ${meta.model_info?.name || 'Selected Model'}`;
    }

    // Load Three.js Terrain (heightmapUrl FIRST, opticalUrl SECOND)
    const opticalUrl = urls.optical_texture_png || urls.heightmap_8bit_preview_png;
    const heightmapUrl = urls.heightmap_8bit_preview_png || urls.optical_texture_png;

    if (opticalUrl && heightmapUrl) {
      try {
        await loadTerrain(heightmapUrl, opticalUrl, meta);
        initFloodSimulator(heightmapUrl, meta);   // correct order: (url, metadata)
      } catch (err) {
        console.error('[Simulator] Failed to load terrain:', err);
      }
    }

    // Load Mesh Overlay if available (turned off by default, opacity 10%)
    if (urls.mesh_glb) {
      try {
        await loadMeshOverlay(urls.mesh_glb, meta, 0.10, false);
        if (meshToggle) meshToggle.checked = false;
        if (meshOpacitySlider) meshOpacitySlider.value = '0.10';
        if (meshOpacityVal) meshOpacityVal.textContent = '10%';
        if (meshWireframeToggle) meshWireframeToggle.checked = true;
      } catch (meshErr) {
        console.warn('[Simulator] GLB mesh overlay load error:', meshErr);
      }
    }

    // Ensure wireframe is applied by default as requested
    setMeshWireframe(true);
    setTerrainWireframe(true);

    // Re-sync renderer size after async load — the canvas may have gotten
    // real pixel dimensions while the terrain was loading asynchronously.
    forceResize();

    // Update Download Links
    updateDownloadLinks(urls);

    // Update Flood Slider bounds
    if (floodSlider) {
      floodSlider.min = minElev.toString();
      floodSlider.max = maxElev.toString();
      floodSlider.value = minElev.toString();
      if (floodVal) floodVal.textContent = `${minElev.toFixed(1)} m`;
    }
  }

  function updateDownloadLinks(urls) {
    if (dlHeightmap && urls.heightmap_16bit_png) dlHeightmap.href = urls.heightmap_16bit_png;
    if (dlGeotiff && urls.geotiff_dsm_32bit) dlGeotiff.href = urls.geotiff_dsm_32bit;
    if (dlMesh && urls.mesh_glb) {
      dlMesh.href = urls.mesh_glb;
      dlMesh.style.display = 'inline-flex';
    }
    if (dlPointcloud && urls.pointcloud_ply) {
      dlPointcloud.href = urls.pointcloud_ply;
      dlPointcloud.style.display = 'inline-flex';
    }
    if (dlMetadata && urls.calibration_metadata_json) dlMetadata.href = urls.calibration_metadata_json;
  }

  // ── Telemetry Animation Loop ──────────────────────────────────────────────
  function startTelemetryLoop() {
    function tick(now) {
      frameCount++;
      if (now - lastTime >= 1000) {
        currentFps = frameCount;
        frameCount = 0;
        lastTime = now;
        if (hudFps) hudFps.textContent = `${currentFps} FPS`;
      }

      // Update synthetic flight instrumentation
      if (isFlying || activeCameraMode === 'fpv') {
        const timeSec = now * 0.001;
        const hdg = Math.floor((timeSec * 15) % 360);
        const compassDirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
        const dir = compassDirs[Math.floor((hdg + 22.5) / 45) % 8];
        if (hudHeading) hudHeading.textContent = `${String(hdg).padStart(3, '0')}° ${dir}`;

        const alt = (55.0 + Math.sin(timeSec * 0.8) * 12.0).toFixed(1);
        if (hudAltitude) hudAltitude.textContent = `${alt} m`;

        const spd = (16.0 + Math.cos(timeSec * 0.5) * 3.5).toFixed(1);
        if (hudAirspeed) hudAirspeed.textContent = `${spd} m/s`;

        const pitch = Math.round(Math.sin(timeSec * 0.7) * 8 - 4);
        const roll = Math.round(Math.cos(timeSec * 0.9) * 5);
        if (hudAttitude) hudAttitude.textContent = `${pitch}° / ${roll}°`;
      }

      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  // ── Event Listeners ───────────────────────────────────────────────────────
  function setupEventListeners() {
    // Mode toggles
    if (btnModeOrtho) {
      btnModeOrtho.addEventListener('click', () => setRenderMode('ortho'));
    }
    if (btnModeWireframe) {
      btnModeWireframe.addEventListener('click', () => setRenderMode('wireframe'));
    }
    if (btnModeHeatmap) {
      btnModeHeatmap.addEventListener('click', () => setRenderMode('heatmap'));
    }
    if (btnModeIsolines) {
      btnModeIsolines.addEventListener('click', () => setRenderMode('isolines'));
    }

    // Engine switcher
    if (btnEngineThree) {
      btnEngineThree.addEventListener('click', () => switchEngine('threejs'));
    }
    if (btnEngineCesium) {
      btnEngineCesium.addEventListener('click', () => switchEngine('cesium'));
    }

    // Sun Angle Slider
    if (sunSlider && sunLabel) {
      sunSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        const hours = Math.floor(val);
        const mins = (val % 1 === 0.5) ? '30' : '00';
        sunLabel.textContent = `${hours}:${mins} EST`;
        setSunAngle(val * 20.0);
        setCesiumSolarTime(val);
      });
    }

    // Camera Rigs
    if (btnCamFpv) {
      btnCamFpv.addEventListener('click', () => {
        setCameraRig('fpv');
        toggleFlythrough();
      });
    }
    if (btnCamOrbit) {
      btnCamOrbit.addEventListener('click', () => {
        setCameraRig('orbit');
      });
    }
    if (btnCamNadir) {
      btnCamNadir.addEventListener('click', () => {
        setCameraRig('nadir');
      });
    }
    if (btnCamReset) {
      btnCamReset.addEventListener('click', () => {
        resetCamera();
      });
    }

    // Terrain Elevation Sliders
    if (elevSlider && elevVal) {
      elevSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        elevVal.textContent = val.toFixed(1) + '×';
        setElevationScale(val);
      });
    }
    if (speedSlider && speedVal) {
      speedSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        speedVal.textContent = val.toFixed(2);
        setFlySpeed(val);
      });
    }

    // Mesh Overlay Toggles
    if (meshToggle) {
      meshToggle.addEventListener('change', (e) => {
        toggleMeshOverlay();
      });
    }
    if (meshWireframeToggle) {
      meshWireframeToggle.addEventListener('change', (e) => {
        setMeshWireframe(e.target.checked);
        setTerrainWireframe(e.target.checked);
      });
    }
    if (meshOpacitySlider && meshOpacityVal) {
      meshOpacitySlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        meshOpacityVal.textContent = Math.round(val * 100) + '%';
        setMeshOpacity(val);
      });
    }

    // Flood Simulation
    if (floodToggle) {
      floodToggle.addEventListener('change', (e) => {
        toggleFloodVisible(e.target.checked);
      });
    }
    if (floodSlider && floodVal) {
      floodSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        floodVal.textContent = `${val.toFixed(1)} m`;
        setFloodLevel(val);
      });
    }
    if (btnFloodAnimate) {
      btnFloodAnimate.addEventListener('click', () => {
        isFloodAnimating = !isFloodAnimating;
        if (isFloodAnimating) {
          btnFloodAnimate.innerHTML = `⏸ Stop Rise`;
          btnFloodAnimate.classList.add('bg-secondary', 'text-on-secondary');
          const max = parseFloat(floodSlider.max || 100);
          animateFloodRising(max, 15000);
        } else {
          btnFloodAnimate.innerHTML = `▶ Animate Rising`;
          btnFloodAnimate.classList.remove('bg-secondary', 'text-on-secondary');
          stopFloodAnimation();
        }
      });
    }

    // React to custom events from viewer.js
    document.addEventListener('dw:floodUpdate', (e) => {
      const detail = e.detail || {};
      if (floodVal && detail.meters !== undefined) {
        floodVal.textContent = `${detail.meters.toFixed(1)} m`;
      }
      if (floodSlider && detail.meters !== undefined && !isFloodAnimating) {
        floodSlider.value = detail.meters.toString();
      }
      if (floodSubmerged && detail.submergedPct !== undefined) {
        floodSubmerged.textContent = `${detail.submergedPct.toFixed(1)}%`;
      }
      if (floodDepth && detail.depthAboveMin !== undefined) {
        floodDepth.textContent = `+${detail.depthAboveMin.toFixed(1)} m`;
      }
      if (floodRiskBadge && detail.risk) {
        floodRiskBadge.textContent = detail.risk.label;
        floodRiskBadge.className = `px-2 py-0.5 rounded font-mono-data-sm text-[10px] font-bold uppercase ${detail.risk.cssClass}`;
      }
    });

    document.addEventListener('dw:orbitMode', () => {
      setCameraRig('orbit');
    });

    // Keyboard Flight Binds
    window.addEventListener('keydown', (e) => {
      keysDown[e.key.toLowerCase()] = true;
      setKeyDown(e.key);
    });
    window.addEventListener('keyup', (e) => {
      keysDown[e.key.toLowerCase()] = false;
      setKeyUp(e.key);
    });
  }

  // ── Render Modes ──────────────────────────────────────────────────────────
  function setRenderMode(mode) {
    activeRenderMode = mode;
    [btnModeOrtho, btnModeWireframe, btnModeHeatmap, btnModeIsolines].forEach((btn) => {
      if (btn) btn.className = 'px-3 py-1 rounded font-mono-data-sm text-mono-data-sm text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-all';
    });

    if (mode === 'ortho') {
      btnModeOrtho.className = 'px-3 py-1 rounded font-mono-data-sm text-mono-data-sm transition-all bg-primary-container text-on-primary-container font-semibold shadow-[0_0_12px_rgba(0,240,255,0.4)]';
      setMeshWireframe(false);
      setTerrainWireframe(false);
      if (meshWireframeToggle) meshWireframeToggle.checked = false;
      if (activeTask && activeTask.download_urls && activeTask.download_urls.optical_texture_png) {
        setTerrainTexture(activeTask.download_urls.optical_texture_png);
      }
      setCesiumDrapeMode('optical');
    } else if (mode === 'wireframe') {
      btnModeWireframe.className = 'px-3 py-1 rounded font-mono-data-sm text-mono-data-sm transition-all bg-primary-container text-on-primary-container font-semibold shadow-[0_0_12px_rgba(0,240,255,0.4)]';
      setMeshWireframe(true);
      setTerrainWireframe(true);
      if (meshWireframeToggle) meshWireframeToggle.checked = true;
      setCesiumDrapeMode('mesh_only');
    } else if (mode === 'heatmap') {
      btnModeHeatmap.className = 'px-3 py-1 rounded font-mono-data-sm text-mono-data-sm transition-all bg-primary-container text-on-primary-container font-semibold shadow-[0_0_12px_rgba(0,240,255,0.4)]';
      setMeshWireframe(false);
      setTerrainWireframe(false);
      if (meshWireframeToggle) meshWireframeToggle.checked = false;
      if (activeTask && activeTask.download_urls && activeTask.download_urls.depth_colorized_png) {
        setTerrainTexture(activeTask.download_urls.depth_colorized_png);
      }
      setCesiumDrapeMode('depth');
    } else if (mode === 'isolines') {
      btnModeIsolines.className = 'px-3 py-1 rounded font-mono-data-sm text-mono-data-sm transition-all bg-primary-container text-on-primary-container font-semibold shadow-[0_0_12px_rgba(0,240,255,0.4)]';
      setMeshWireframe(true);
      setTerrainWireframe(true);
      if (meshWireframeToggle) meshWireframeToggle.checked = true;
    }
  }

  function switchEngine(engine) {
    activeEngine = engine;
    if (engine === 'threejs') {
      btnEngineThree.className = 'px-2.5 py-1 rounded bg-surface-container text-primary font-mono-data-sm text-mono-data-sm font-semibold border border-primary-container/30';
      btnEngineCesium.className = 'px-2.5 py-1 rounded hover:bg-surface-container text-on-surface-variant font-mono-data-sm text-mono-data-sm';
      if (threeCanvas) threeCanvas.style.display = 'block';
      if (cesiumContainer) cesiumContainer.style.display = 'none';
    } else {
      btnEngineCesium.className = 'px-2.5 py-1 rounded bg-surface-container text-primary font-mono-data-sm text-mono-data-sm font-semibold border border-primary-container/30';
      btnEngineThree.className = 'px-2.5 py-1 rounded hover:bg-surface-container text-on-surface-variant font-mono-data-sm text-mono-data-sm';
      if (threeCanvas) threeCanvas.style.display = 'none';
      if (cesiumContainer) {
        cesiumContainer.style.display = 'block';
        if (!cesiumContainer.hasChildNodes()) {
          initCesiumViewer('cesium-container');
          if (activeTask && activeTask.download_urls) {
            loadCesiumModel(activeTask.download_urls.mesh_glb, activeTask.download_urls.optical_texture_png, activeTask.metadata);
          }
        }
      }
    }
  }

  function setCameraRig(rig) {
    activeCameraMode = rig;
    [btnCamFpv, btnCamOrbit, btnCamNadir].forEach((btn) => {
      if (btn) btn.classList.remove('bg-surface-container-high', 'text-primary');
    });

    if (rig === 'fpv') {
      if (btnCamFpv) btnCamFpv.classList.add('bg-surface-container-high', 'text-primary');
      isFlying = true;
      setFlythrough(true);
    } else if (rig === 'orbit') {
      if (btnCamOrbit) btnCamOrbit.classList.add('bg-surface-container-high', 'text-primary');
      isFlying = false;
      setFlythrough(false);
    } else if (rig === 'nadir') {
      if (btnCamNadir) btnCamNadir.classList.add('bg-surface-container-high', 'text-primary');
      isFlying = false;
      setFlythrough(false);
      setCameraNadir();
    }
  }

  // Auto-init on DOMContentLoaded or immediately if DOM already interactive/loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSimulator);
  } else {
    initSimulator();
  }
})();
