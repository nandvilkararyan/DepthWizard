/**
 * DepthWizard — Three.js 3D Terrain Viewer
 * ==========================================
 * Primary terrain renderer with:
 *   - Displacement-mapped PlaneGeometry (base terrain layer)
 *   - GLB mesh overlay from Open3D point-cloud/Poisson pipeline (Module D)
 *   - Auto-flythrough camera animation
 *   - Manual OrbitControls
 *
 * Mesh overlay API:
 *   loadMeshOverlay(glbUrl, metadata)  – loads GLB, co-registers with terrain
 *   setMeshOpacity(0..1)               – live opacity control
 *   toggleMeshOverlay()                – show/hide, returns boolean
 *   setMeshWireframe(bool)             – toggle wireframe debug view
 *   disposeMeshOverlay()               – cleanup on new image
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

/* ------------------------------------------------------------------ */
/*  Scene globals                                                       */
/* ------------------------------------------------------------------ */
let renderer, scene, camera, controls;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let terrainMesh = null;
let terrainHeightData = null;
let terrainHeightWidth = 0;
let terrainHeightHeight = 0;
let terrainHeightImage = null;
let inspectionCard = null;
let animFrameId = null;
let _lastMetadata = null; // saved for dispScale re-calc by setElevationScale

// Flythrough state
let isFlying = false; // disabled by default for better manual 3D control
let flyClock = new THREE.Clock();
let flyPathRadius = 6.0;
let flyPathHeight = 4.0;
let flySpeed = 0.1; // radians per second

// Terrain parameters (updated from metadata)
let elevationScale = 1.0;
let sunAngle = 45;

/* ── Mesh overlay state ───────────────────────────────────────────── */
let meshOverlayGroup = null; // THREE.Group holding the loaded GLB
let meshOverlayVisible = true;

/* ── Flood simulator state ─────────────────────────────────────────── */
const _flood = {
  mesh: null, // THREE.Mesh — the water plane
  visible: false,
  minElev: 0, // meters — from metadata
  maxElev: 100, // meters — from metadata
  dispScale: 1.5, // mirror of terrain dispScale
  dispBias: -0.6, // mirror of terrain dispBias
  currentLevel: 0, // current water elevation in meters
  elevPixels: null, // Float32Array of per-pixel elevations (from 8-bit PNG)
  animRaf: null, // animation frame id for rising animation
  rippleTime: 0, // cumulative time for UV ripple
  terrainDepth: 8,
};

// Dispatch event so index.html can update the flood readout UI
function _floodEvent(detail) {
  document.dispatchEvent(new CustomEvent("dw:floodUpdate", { detail }));
}

/* ------------------------------------------------------------------ */
/*  Init                                                                */
/* ------------------------------------------------------------------ */
export function initViewer(canvasEl) {
  renderer = new THREE.WebGLRenderer({
    canvas: canvasEl,
    antialias: true,
    alpha: false,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.3;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070b17);
  scene.fog = new THREE.FogExp2(0x070b17, 0.12);

  // Camera
  camera = new THREE.PerspectiveCamera(
    55,
    canvasEl.clientWidth / canvasEl.clientHeight,
    0.01,
    100,
  );
  // Set initial camera position for proper 3D viewing angle (not top-down)
  camera.position.set(5, 4, 5);
  camera.lookAt(0, 0, 0);

  // OrbitControls
  controls = new OrbitControls(camera, canvasEl);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI / 2.05;
  controls.minDistance = 1;
  controls.maxDistance = 12;
  controls.enabled = true; // enabled by default for proper 3D interaction
  controls.autoRotate = false; // disable auto-rotate for better manual control

  // Lights
  const ambient = new THREE.AmbientLight(0xffffff, 1.0);
  ambient.name = "ambient";
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xffffff, 2.0);
  sun.name = "sun";
  sun.position.set(5, 10, 7);
  sun.target.position.set(0, 0, 0);
  scene.add(sun.target);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 0.1;
  sun.shadow.camera.far = 30;
  sun.shadow.camera.left = -8;
  sun.shadow.camera.right = 8;
  sun.shadow.camera.top = 8;
  sun.shadow.camera.bottom = -8;
  scene.add(sun);

  // Subtle hemisphere fill
  scene.add(new THREE.HemisphereLight(0x1a3a5c, 0x0a0e1a, 0.5));

  // Stars particle field
  _addStars();

  // Resize handling
  const ro = new ResizeObserver(() => _onResize());
  ro.observe(canvasEl.parentElement);
  _onResize();

  // Mouse interaction → pause flythrough, enable orbit
  canvasEl.addEventListener("mousedown", _onUserInteract);
  canvasEl.addEventListener("touchstart", _onUserInteract, { passive: true });
  canvasEl.addEventListener("click", _inspectTerrain);

  inspectionCard = document.getElementById("terrain-inspection-card");
  if (!inspectionCard) {
    inspectionCard = document.createElement("div");
    inspectionCard.id = "terrain-inspection-card";
    inspectionCard.style.cssText =
      "display:none;position:absolute;z-index:20;pointer-events:none;background:rgba(7,11,23,.94);color:#fff;padding:10px 12px;border:1px solid rgba(126,200,227,.55);border-radius:8px;font:12px/1.5 sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.35)";
    canvasEl.parentElement.appendChild(inspectionCard);
  }
  inspectionCard.style.cssText =
    "display:none;position:absolute;z-index:20;pointer-events:none;background:rgba(7,11,23,.94);color:#fff;padding:10px 12px;border:1px solid rgba(126,200,227,.55);border-radius:8px;font:12px/1.5 sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.35)";

  _startRenderLoop();
}

/* ------------------------------------------------------------------ */
/*  Load terrain from heightmap + texture URLs + metadata              */
/* ------------------------------------------------------------------ */
export async function loadTerrain(heightmapUrl, textureUrl, metadata) {
  _lastMetadata = metadata;

  // Dispose old terrain
  if (terrainMesh) {
    scene.remove(terrainMesh);
    terrainMesh.geometry.dispose();
    terrainMesh.material.dispose();
    terrainMesh = null;
  }
  // Also dispose any stale mesh overlay and flood sim (new image loaded)
  disposeMeshOverlay();
  disposeFloodSimulator();

  const { dispScale } = _computeDispScale(metadata);

  const loader = new THREE.TextureLoader();

  const [heightTex, colorTex] = await Promise.all([
    loader.loadAsync(heightmapUrl),
    loader.loadAsync(textureUrl),
  ]);

  heightTex.wrapS = heightTex.wrapT = THREE.ClampToEdgeWrapping;
  colorTex.wrapS = colorTex.wrapT = THREE.ClampToEdgeWrapping;
  colorTex.colorSpace = THREE.SRGBColorSpace;

  const { width, depth, segmentsX, segmentsY } = _terrainGeometry(
    metadata,
    heightTex.image,
  );
  const geo = new THREE.PlaneGeometry(width, depth, segmentsX, segmentsY);
  geo.rotateX(-Math.PI / 2);
  geo.center();
  _prepareHeightSampler(heightTex.image);
  _applyHeightDisplacement(geo, heightTex.image, dispScale, -dispScale * 0.4);

  const mat = new THREE.MeshStandardMaterial({
    map: colorTex,
    displacementMap: null,
    displacementScale: 0,
    displacementBias: 0,
    normalMap: _heightNormalMap(heightTex.image),
    normalScale: new THREE.Vector2(0.8, 0.8),
    roughness: 0.88,
    metalness: 0.04,
  });

  terrainMesh = new THREE.Mesh(geo, mat);
  terrainMesh.receiveShadow = true;
  terrainMesh.castShadow = false;
  scene.add(terrainMesh);
  _frameTerrain(geo);

  // Reset camera to flythrough start
  flyClock = new THREE.Clock();
  isFlying = true;
  controls.enabled = false;

  // Adjust fog to terrain size
  scene.fog.density = 0.08;

  return terrainMesh;
}

/* ------------------------------------------------------------------ */
/*  Load terrain with hypsometric (elevation-based) color fallback     */
/* ------------------------------------------------------------------ */
export async function loadTerrainHeightOnly(heightmapUrl, metadata) {
  _lastMetadata = metadata;

  if (terrainMesh) {
    scene.remove(terrainMesh);
    terrainMesh.geometry.dispose();
    terrainMesh.material.dispose();
    terrainMesh = null;
  }
  disposeMeshOverlay();
  disposeFloodSimulator();

  const { dispScale } = _computeDispScale(metadata);
  const loader = new THREE.TextureLoader();
  const heightTex = await loader.loadAsync(heightmapUrl);
  heightTex.wrapS = heightTex.wrapT = THREE.ClampToEdgeWrapping;

  const { width, depth, segmentsX, segmentsY } = _terrainGeometry(
    metadata,
    heightTex.image,
  );
  const geo = new THREE.PlaneGeometry(width, depth, segmentsX, segmentsY);
  geo.rotateX(-Math.PI / 2);
  geo.center();
  _prepareHeightSampler(heightTex.image);
  _applyHeightDisplacement(geo, heightTex.image, dispScale, -dispScale * 0.4);

  // Hypsometric gradient: green → brown → snow
  const mat = new THREE.MeshStandardMaterial({
    color: 0x4a7c59,
    displacementMap: null,
    displacementScale: 0,
    displacementBias: 0,
    normalMap: _heightNormalMap(heightTex.image),
    normalScale: new THREE.Vector2(0.8, 0.8),
    roughness: 0.9,
    metalness: 0.02,
  });

  terrainMesh = new THREE.Mesh(geo, mat);
  terrainMesh.receiveShadow = true;
  scene.add(terrainMesh);
  _frameTerrain(geo);

  flyClock = new THREE.Clock();
  isFlying = true;
  controls.enabled = false;
}

/* ------------------------------------------------------------------ */
/*  Mesh Overlay API (Module D — GLB from Poisson reconstruction)      */
/* ------------------------------------------------------------------ */

/**
 * Loads a GLB mesh produced by the Python Open3D pipeline and places it
 * as a semi-transparent overlay, co-registered with the terrain plane.
 *
 * The GLB vertices are in a normalized coordinate space:
 *   X ∈ [-1,  1]  (image columns)
 *   Y ∈ [ 0,  1]  (normalized elevation: 0=ground, 1=max)
 *   Z ∈ [-1,  1]  (image rows)
 *
 * We scale to match the terrain plane: scale(4, dispScale, 4) so the
 * mesh spans ±4 units in X/Z (terrain is 8×8) and the elevation
 * axis matches the displacement-map height.
 *
 * @param {string}  glbUrl   – URL served by FastAPI /files/
 * @param {object}  metadata – From the /process response
 * @param {number}  opacity  – Initial mesh opacity (default 0.88)
 * @returns {Promise<THREE.Group>}
 */
export function loadMeshOverlay(glbUrl, metadata, opacity = 0.88) {
  // Clean up previous overlay if any
  disposeMeshOverlay();

  const { dispScale } = _computeDispScale(metadata);

  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();

    loader.load(
      glbUrl,
      (gltf) => {
        meshOverlayGroup = gltf.scene;

        // ── Co-registration with terrain plane ────────────────────────
        // Mesh spans X ∈ [-1,1], Z ∈ [-1,1], Y ∈ [0,1] (normalized).
        // Terrain plane is 8×8 units with Y displaced by ±dispScale.
        meshOverlayGroup.scale.set(4.0, dispScale, 4.0);
        // Match terrain displacement bias (the plane is shifted down by 40% of dispScale)
        meshOverlayGroup.position.set(0, -dispScale * 0.4, 0);

        // ── Materials ─────────────────────────────────────────────────
        meshOverlayGroup.traverse((child) => {
          if (!child.isMesh) return;

          // Replace with a MeshStandardMaterial that shows vertex colors
          const newMat = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.65,
            metalness: 0.05,
            transparent: opacity < 1.0,
            opacity: opacity,
            depthWrite: opacity >= 0.99, // disable depth write when transparent
          });

          // Keep original material side-setting if available
          if (child.material && child.material.side !== undefined) {
            newMat.side = THREE.DoubleSide;
          }

          child.material.dispose();
          child.material = newMat;
          child.castShadow = true;
          child.receiveShadow = true;
        });

        scene.add(meshOverlayGroup);
        meshOverlayVisible = true;

        console.log(
          `[Viewer] Mesh overlay loaded: ${glbUrl}  ` +
            `(dispScale=${dispScale.toFixed(3)}, opacity=${opacity})`,
        );
        resolve(meshOverlayGroup);
      },
      undefined, // progress callback
      (err) => {
        console.error("[Viewer] GLB load failed:", err);
        reject(err);
      },
    );
  });
}

/**
 * Adjust the opacity of all meshes in the overlay group.
 * @param {number} val – 0 (invisible) to 1 (fully opaque)
 */
export function setMeshOpacity(val) {
  userMeshOpacity = val;
  if (!meshOverlayGroup) return;
  meshOverlayGroup.traverse((child) => {
    if (!child.isMesh) return;
    child.material.opacity = val;
    child.material.transparent = val < 1.0;
    child.material.depthWrite = val >= 0.99;
    child.material.needsUpdate = true;
  });
}

/**
 * Toggle the mesh overlay visibility.
 * @returns {boolean} New visible state
 */
export function toggleMeshOverlay() {
  if (!meshOverlayGroup) return false;
  meshOverlayVisible = !meshOverlayVisible;
  meshOverlayGroup.visible = meshOverlayVisible;
  return meshOverlayVisible;
}

/**
 * Switch the mesh overlay to wireframe mode (useful for inspecting geometry).
 * @param {boolean} on
 */
export function setMeshWireframe(on) {
  if (!meshOverlayGroup) return;
  meshOverlayGroup.traverse((child) => {
    if (!child.isMesh) return;
    child.material.wireframe = on;
    child.material.transparent = on ? false : child.material.opacity < 1.0;
    child.material.needsUpdate = true;
  });
}

/**
 * Dispose the mesh overlay and remove it from the scene.
 */
export function disposeMeshOverlay() {
  if (!meshOverlayGroup) return;
  scene.remove(meshOverlayGroup);
  meshOverlayGroup.traverse((child) => {
    if (child.isMesh) {
      child.geometry.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
  meshOverlayGroup = null;
  meshOverlayVisible = true;
}

/* ------------------------------------------------------------------ */
/*  Camera Controls API                                                 */
/* ------------------------------------------------------------------ */
export function toggleFlythrough() {
  isFlying = !isFlying;
  controls.enabled = !isFlying;
  if (isFlying) flyClock = new THREE.Clock();
  return isFlying;
}

export function setFlythrough(active) {
  isFlying = active;
  controls.enabled = !active;
  if (active) flyClock = new THREE.Clock();
}

export function resetCamera() {
  flyClock = new THREE.Clock();
  isFlying = true;
  controls.enabled = false;
  controls.reset();
}

/* ------------------------------------------------------------------ */
/*  Terrain tuning API                                                  */
/* ------------------------------------------------------------------ */
export function setElevationScale(val) {
  elevationScale = val;
  if (terrainMesh) {
    const { dispScale } = _computeDispScale(_lastMetadata);
    _applyHeightDisplacement(
      terrainMesh.geometry,
      terrainHeightImage,
      dispScale,
      -dispScale * 0.4,
    );

    // Keep camera far plane large enough for new dispScale
    if (camera && terrainMesh.geometry.boundingSphere) {
      const radius = Math.max(terrainMesh.geometry.boundingSphere.radius || 5, 4);
      camera.far = Math.max(200, radius * 20);
      camera.updateProjectionMatrix();
    }

    // Also re-scale the mesh overlay to stay registered
    if (meshOverlayGroup) {
      meshOverlayGroup.scale.set(4.0, dispScale, 4.0);
      meshOverlayGroup.position.y = -dispScale * 0.4;
    }

    // FIX: keep flood simulator in sync with the new displacement scale
    if (_flood.mesh) {
      _flood.dispScale = dispScale;
      _flood.dispBias = -dispScale * 0.4;
      // Rebuild vertex Y cache since terrain geometry just changed
      const positions = terrainMesh.geometry.attributes.position;
      _flood.vertexYPositions = new Float32Array(positions.count);
      for (let i = 0; i < positions.count; i++) {
        _flood.vertexYPositions[i] = positions.getY(i);
      }
      _flood.vertexCount = positions.count;
      // Re-apply current water level with the corrected scale
      setFloodLevel(_flood.currentLevel);
    }
  }
}

export function setFlySpeed(val) {
  flySpeed = val;
}
export function setFlyHeight(val) {
  flyPathHeight = val;
  flyPathRadius = val * 1.8;
}

/** Returns the viewer identifier string for use by index.html */
export function getViewerType() {
  return "threejs";
}

/* ------------------------------------------------------------------ */
/*  Flood Level Simulation API                                          */
/* ------------------------------------------------------------------ */

/**
 * Initialises the flood simulator after a terrain has been loaded.
 * Samples the 8-bit heightmap PNG into an offscreen canvas and builds
 * a flat Float32Array of per-pixel elevations (in meters).
 *
 * @param {string} heightmapUrl  – URL of the 8-bit preview PNG
 * @param {object} metadata      – Pipeline metadata (elevation_metrics required)
 */
export async function initFloodSimulator(heightmapUrl, metadata) {
  disposeFloodSimulator();

  const { dispScale, minElev, maxElev } = _computeDispScale(metadata);
  _flood.dispScale = dispScale;
  _flood.dispBias = -dispScale * 0.4;
  _flood.minElev = minElev;
  _flood.maxElev = maxElev;
  _flood.currentLevel = _flood.minElev;

  // ── 1. Build per-vertex elevation array from terrain mesh ──────────
  if (terrainMesh) {
    const positions = terrainMesh.geometry.attributes.position;
    _flood.vertexYPositions = new Float32Array(positions.count);
    for (let i = 0; i < positions.count; i++) {
      _flood.vertexYPositions[i] = positions.getY(i);
    }
    _flood.vertexCount = positions.count;
  }

  // ── 2. Also sample 8-bit heightmap into elevation array (for UI) ───
  await new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const maxDim = 128;
      const scale = Math.min(maxDim / img.width, maxDim / img.height, 1);
      canvas.width = Math.max(1, Math.round(img.width * scale));
      canvas.height = Math.max(1, Math.round(img.height * scale));
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      const range = _flood.maxElev - _flood.minElev || 1;
      const px = new Float32Array(canvas.width * canvas.height);
      for (let i = 0; i < px.length; i++) {
        px[i] = _flood.minElev + (data[i * 4] / 255) * range;
      }
      _flood.elevPixels = px;
      resolve();
    };
    img.onerror = () => resolve();
    img.src = heightmapUrl;
  });

  // ── 3. Create simple blue semi-transparent water plane ─────────────
  const aspect = metadata?.scene_geometry?.aspect_ratio || 1;
  _flood.terrainDepth = 8 / Math.max(aspect, 0.01);
  const waterGeo = new THREE.PlaneGeometry(
    8.4,
    _flood.terrainDepth * 1.05,
    1,
    1,
  );
  waterGeo.rotateX(-Math.PI / 2);
  waterGeo.center();

  // Simple MeshStandardMaterial — no shader hacks, no broken alphaMap
  const waterMat = new THREE.MeshStandardMaterial({
    color: 0x0077ff,
    transparent: true,
    opacity: 0.7,
    roughness: 0.15,
    metalness: 0.6,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  _flood.mesh = new THREE.Mesh(waterGeo, waterMat);
  _flood.mesh.name = "floodWater";
  _flood.mesh.visible = false; // hidden by default until user toggles flood panel
  _flood.mesh.renderOrder = 999;
  _flood.visible = false; // flood state hidden by default

  // Start at terrain min elevation
  _flood.mesh.position.y = _flood.dispBias;
  scene.add(_flood.mesh);

  console.log(
    "[FloodSim] Initialised. Elevation range:",
    _flood.minElev.toFixed(1),
    "—",
    _flood.maxElev.toFixed(1),
    "m",
  );
}

/**
 * Moves the water plane to the specified absolute elevation (meters).
 * Returns { pct, riskLabel } for the UI.
 *
 * @param {number} meters – Target water elevation in meters
 * @returns {{ pct: number, riskLabel: string }}
 */
export function setFloodLevel(meters) {
  if (!_flood.mesh) return { pct: 0, riskLabel: "No terrain" };

  const clampedMeters = Math.max(
    _flood.minElev,
    Math.min(_flood.maxElev, meters),
  );
  _flood.currentLevel = clampedMeters;
  const waterShader = _flood.mesh.material.userData.shader;
  if (waterShader) waterShader.uniforms.waterLevel.value = clampedMeters;

  // Map meters → Three.js scene Y using the same formula as terrain displacement
  const range = _flood.maxElev - _flood.minElev || 1;
  const normElev = (clampedMeters - _flood.minElev) / range; // 0..1
  const sceneY = normElev * _flood.dispScale + _flood.dispBias;
  _flood.mesh.position.y = sceneY;

  // ── Compute % submerged from vertex height checks ─────────────────
  let pct = 0;
  if (_flood.vertexYPositions && _flood.vertexCount > 0) {
    let below = 0;
    for (let i = 0; i < _flood.vertexCount; i++) {
      if (_flood.vertexYPositions[i] <= sceneY) below++;
    }
    pct = (below / _flood.vertexCount) * 100;
  }

  // ── Risk label ────────────────────────────────────────────────────
  let riskLabel, riskClass;
  if (pct < 5) {
    riskLabel = "Minimal";
    riskClass = "safe";
  } else if (pct < 15) {
    riskLabel = "Low";
    riskClass = "low";
  } else if (pct < 35) {
    riskLabel = "Moderate";
    riskClass = "moderate";
  } else if (pct < 60) {
    riskLabel = "Severe";
    riskClass = "severe";
  } else if (pct < 85) {
    riskLabel = "Extreme";
    riskClass = "extreme";
  } else {
    riskLabel = "Catastrophic";
    riskClass = "catastrophic";
  }

  const result = { meters: clampedMeters, pct, riskLabel, riskClass };
  _floodEvent(result);
  return result;
}

/**
 * Convenience: set flood level by fraction (0 = min elev, 1 = max elev).
 * @param {number} frac – 0..1
 */
export function setFloodLevelPct(frac) {
  const meters = _flood.minElev + frac * (_flood.maxElev - _flood.minElev);
  return setFloodLevel(meters);
}

/**
 * Animates water rising smoothly from the current level to targetMeters.
 * @param {number} targetMeters
 * @param {number} durationMs – default 4000ms
 */
export function animateFloodRising(targetMeters = null, durationMs = 4000) {
  if (!_flood.mesh) return;
  if (_flood.animRaf) cancelAnimationFrame(_flood.animRaf);

  const target =
    targetMeters ?? _flood.minElev + (_flood.maxElev - _flood.minElev) * 0.75;
  const startMeters = _flood.currentLevel;
  const startTime = performance.now();

  function step(now) {
    const elapsed = now - startTime;
    const t = Math.min(elapsed / durationMs, 1);
    const eased = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    setFloodLevel(startMeters + (target - startMeters) * eased);
    if (t < 1) {
      _flood.animRaf = requestAnimationFrame(step);
    } else {
      _flood.animRaf = null;
      // Ensure water plane stays visible after animation completes
      if (_flood.mesh) {
        _flood.mesh.visible = _flood.visible;
        console.log(
          "[FloodSim] Animation complete." +
          ` Final position.y=${_flood.mesh.position.y.toFixed(4)}` +
          ` visible=${_flood.mesh.visible}` +
          ` currentLevel=${_flood.currentLevel.toFixed(2)}m` +
          ` dispScale=${_flood.dispScale.toFixed(4)}` +
          ` dispBias=${_flood.dispBias.toFixed(4)}`
        );
      }
    }
  }

  _flood.animRaf = requestAnimationFrame(step);
}

/**
 * Stops any in-progress flood animation.
 */
export function stopFloodAnimation() {
  if (_flood.animRaf) {
    cancelAnimationFrame(_flood.animRaf);
    _flood.animRaf = null;
  }
}

/**
 * Show or hide the flood water plane.
 * @returns {boolean} New visible state
 */
export function toggleFloodVisible() {
  _flood.visible = !_flood.visible;
  if (_flood.mesh) _flood.mesh.visible = _flood.visible;
  return _flood.visible;
}

/**
 * Sets flood visibility explicitly.
 * @param {boolean} on
 */
export function setFloodVisible(on) {
  _flood.visible = on;
  if (_flood.mesh) _flood.mesh.visible = on;
}

/** Returns the current flood state for external queries. */
export function getFloodState() {
  return {
    active: _flood.visible,
    meters: _flood.currentLevel,
    minElev: _flood.minElev,
    maxElev: _flood.maxElev,
  };
}

/**
 * Dispose the flood simulator — removes water plane from scene.
 */
export function disposeFloodSimulator() {
  if (_flood.animRaf) {
    cancelAnimationFrame(_flood.animRaf);
    _flood.animRaf = null;
  }
  if (_flood.mesh) {
    scene.remove(_flood.mesh);
    _flood.mesh.geometry.dispose();
    _flood.mesh.material.dispose();
    _flood.mesh = null;
  }
  _flood.visible = false;
  _flood.elevPixels = null;
  _flood.rippleTime = 0;
}

export function setSunAngle(degrees) {
  sunAngle = degrees;
  const rad = THREE.MathUtils.degToRad(degrees);
  const sun = scene.getObjectByName("sun");
  if (sun) {
    sun.position.set(Math.cos(rad) * 360, 300, Math.sin(rad) * 360);
    sun.target.position.set(0, 0, 0);
    sun.target.updateMatrixWorld();
  }
}

/* ------------------------------------------------------------------ */
/*  Render loop                                                         */
/* ------------------------------------------------------------------ */
function _startRenderLoop() {
  if (animFrameId) cancelAnimationFrame(animFrameId);

  function tick() {
    animFrameId = requestAnimationFrame(tick);

    if (isFlying) {
      const t = flyClock.getElapsedTime() * flySpeed;
      camera.position.x = Math.sin(t) * flyPathRadius;
      camera.position.z = Math.cos(t) * flyPathRadius;
      camera.position.y = flyPathHeight + Math.sin(t * 0.5) * 0.4;
      camera.lookAt(0, 0.2, 0);
    } else {
      controls.update();
    }

    // ── Flood water ripple animation ──────────────────────────────────
    if (_flood.mesh && _flood.mesh.visible) {
      _flood.rippleTime += 0.008;
      if (_flood.mesh.material.normalMap) {
        _flood.mesh.material.normalMap.offset.x = _flood.rippleTime * 0.015;
        _flood.mesh.material.normalMap.offset.y = _flood.rippleTime * 0.009;
      }
    }

    renderer.render(scene, camera);
  }

  tick();
}

/* ------------------------------------------------------------------ */
/*  Private helpers                                                     */
/* ------------------------------------------------------------------ */

let currentViewMode = "combined"; // 'mesh-only', 'threejs-only', 'combined'
let userMeshOpacity = 0.88;

/**
 * Sets the 3D model viewing mode:
 *   - 'mesh-only'   : Display ONLY the GLB reconstructed 3D mesh (skeleton)
 *   - 'threejs-only': Display ONLY the Three.js terrain plane
 *   - 'combined'    : Overlay GLB mesh on top of terrain plane
 */
export function setViewMode(mode) {
  currentViewMode = mode;
  if (terrainMesh) {
    terrainMesh.visible = mode === "threejs-only" || mode === "combined";
  }
  if (meshOverlayGroup) {
    if (mode === "mesh-only") {
      meshOverlayGroup.visible = true;
      setMeshOpacity(1.0);
    } else if (mode === "threejs-only") {
      meshOverlayGroup.visible = false;
    } else if (mode === "combined") {
      meshOverlayGroup.visible = meshOverlayVisible;
      setMeshOpacity(userMeshOpacity);
    }
  }
}

/**
 * Computes the scene displacement scale from the elevation range metadata.
 * Uses suggested_disp_scale sweet-spot from Module B if available, or fallback.
 */
function _computeDispScale(metadata) {
  const minElev = metadata?.elevation_metrics?.min_elevation_meters ?? 0;
  const maxElev = metadata?.elevation_metrics?.max_elevation_meters ?? 100;
  const elevRange = maxElev - minElev || 1;
  const baseDisp = metadata?.elevation_metrics?.suggested_disp_scale 
    ?? Math.min(1.2, Math.max(0.35, 0.35 + (elevRange / 500.0) * 0.45));
  const dispScale = baseDisp * elevationScale;
  return { elevRange, dispScale, minElev, maxElev };
}

function _applyHeightDisplacement(geometry, image, scale, bias) {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  const pixels = context.getImageData(0, 0, image.width, image.height).data;

  // Build a 2D heightmap array and find data range
  const imgW = image.width;
  const imgH = image.height;
  const heightMap = new Float32Array(imgW * imgH);
  let dataMin = 255, dataMax = 0;
  for (let i = 0; i < imgW * imgH; i++) {
    const v = pixels[i * 4]; // red channel
    heightMap[i] = v;
    if (v < dataMin) dataMin = v;
    if (v > dataMax) dataMax = v;
  }
  
  // Clamp to 1st-99th percentile to remove extreme outliers before normalization
  const sortedValues = Array.from(heightMap).sort((a, b) => a - b);
  const p1Index = Math.floor(sortedValues.length * 0.01);
  const p99Index = Math.floor(sortedValues.length * 0.99);
  const p1 = sortedValues[p1Index];
  const p99 = sortedValues[p99Index];
  for (let i = 0; i < imgW * imgH; i++) {
    heightMap[i] = Math.max(p1, Math.min(p99, heightMap[i]));
  }
  
  // Recalculate data range after percentile clamping
  dataMin = p1;
  dataMax = p99;
  const dataRange = Math.max(dataMax - dataMin, 1);

  // Multi-pass 5x5 mean filter to remove pixel noise & spiky needle artifacts,
  // creating smooth natural surfaces for buildings and terrain
  for (let pass = 0; pass < 4; pass++) {
    const smoothed = new Float32Array(imgW * imgH);
    for (let y = 0; y < imgH; y++) {
      for (let x = 0; x < imgW; x++) {
        let sum = 0;
        let count = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const ny = y + dy;
            const nx = x + dx;
            if (ny >= 0 && ny < imgH && nx >= 0 && nx < imgW) {
              sum += heightMap[ny * imgW + nx];
              count++;
            }
          }
        }
        smoothed[y * imgW + x] = sum / count;
      }
    }
    for (let i = 0; i < imgW * imgH; i++) {
      heightMap[i] = smoothed[i];
    }
  }

  // Smooth edge-tapering along the outer 4% border to prevent perimeter wall curtain spikes
  const borderMarginX = Math.round(imgW * 0.04);
  const borderMarginY = Math.round(imgH * 0.04);
  for (let y = 0; y < imgH; y++) {
    for (let x = 0; x < imgW; x++) {
      let edgeFactor = 1.0;
      if (x < borderMarginX) edgeFactor *= (x / borderMarginX);
      else if (x >= imgW - borderMarginX) edgeFactor *= ((imgW - 1 - x) / borderMarginX);
      if (y < borderMarginY) edgeFactor *= (y / borderMarginY);
      else if (y >= imgH - borderMarginY) edgeFactor *= ((imgH - 1 - y) / borderMarginY);
      
      // Smooth S-curve transition
      edgeFactor = edgeFactor * edgeFactor * (3.0 - 2.0 * edgeFactor);
      heightMap[y * imgW + x] = dataMin + (heightMap[y * imgW + x] - dataMin) * edgeFactor;
    }
  }

  // Apply displacement to geometry vertices
  const positions = geometry.attributes.position;
  const uvs = geometry.attributes.uv;
  for (let index = 0; index < positions.count; index++) {
    const u = uvs.getX(index);
    const v = uvs.getY(index);
    const x = Math.min(imgW - 1, Math.max(0, Math.round(u * (imgW - 1))));
    const y = Math.min(imgH - 1, Math.max(0, Math.round((1 - v) * (imgH - 1))));
    const depthPixel = heightMap[y * imgW + x];
    // The backend already inverts depth→elevation in _calibrate_relative_fallback
    // and SRTM calibration uses a fitted alpha (which can be negative).
    // The 8-bit PNG is therefore bright = high elevation. Read it straight.
    const normalized = (depthPixel - dataMin) / dataRange; // 0..1 — high = high elevation
    positions.setY(index, bias + normalized * scale);
  }
  positions.needsUpdate = true;

  // MUST compute normals AFTER all displacement is applied
  geometry.computeVertexNormals();
  geometry.attributes.normal.needsUpdate = true;
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  terrainHeightImage = image;
}

function _frameTerrain(geometry) {
  geometry.computeBoundingSphere();
  const radius = Math.max(geometry.boundingSphere?.radius || 5, 4);
  camera.near = 0.01;
  // Use a generous far plane: at least 4× the terrain radius, minimum 200 units,
  // to ensure the water plane never gets clipped for large elevation ranges.
  camera.far = Math.max(200, radius * 20);
  camera.position.set(radius * 0.95, radius * 0.7, radius * 1.15);
  camera.lookAt(0, 0, 0);
  controls.target.set(0, 0, 0);
  controls.maxDistance = radius * 4;
  controls.update();
}

function _prepareHeightSampler(image) {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  terrainHeightData = context.getImageData(
    0,
    0,
    canvas.width,
    canvas.height,
  ).data;
  terrainHeightWidth = canvas.width;
  terrainHeightHeight = canvas.height;
  terrainHeightImage = image;
}

function _inspectTerrain(event) {
  if (!terrainMesh || !terrainHeightData || !renderer || !inspectionCard)
    return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(terrainMesh, false)[0];
  if (!hit || !hit.uv) return;

  const pixelX = Math.min(
    terrainHeightWidth - 1,
    Math.max(0, Math.round(hit.uv.x * (terrainHeightWidth - 1))),
  );
  const pixelY = Math.min(
    terrainHeightHeight - 1,
    Math.max(0, Math.round((1 - hit.uv.y) * (terrainHeightHeight - 1))),
  );
  const value =
    terrainHeightData[(pixelY * terrainHeightWidth + pixelX) * 4] / 255;
  const metrics = _lastMetadata?.elevation_metrics || {};
  const minElevation = Number(metrics.min_elevation_meters ?? 0);
  const elevationRange = Number(metrics.elevation_range_meters ?? 1);
  const elevation = minElevation + value * elevationRange;
  const slope = _sampleTerrainSlope(pixelX, pixelY, elevationRange);
  const geo = _lastMetadata?.geospatial_metadata;
  let coordinate = `Pixel (${pixelX}, ${pixelY})`;
  if (geo?.bounds && geo.bounds.length >= 4) {
    const lon =
      geo.bounds[0] +
      (pixelX / Math.max(1, terrainHeightWidth - 1)) *
        (geo.bounds[2] - geo.bounds[0]);
    const lat =
      geo.bounds[3] -
      (pixelY / Math.max(1, terrainHeightHeight - 1)) *
        (geo.bounds[3] - geo.bounds[1]);
    coordinate = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
  }
  inspectionCard.innerHTML = `<strong>Terrain Inspection</strong><br>Elevation: ${elevation.toFixed(1)} m<br>Terrain slope: ${slope.toFixed(1)}°<br>Coordinates: ${coordinate}`;
  const wrap = renderer.domElement.parentElement.getBoundingClientRect();
  inspectionCard.style.left = `${event.clientX - wrap.left + 12}px`;
  inspectionCard.style.top = `${event.clientY - wrap.top + 12}px`;
  inspectionCard.style.display = "block";
}

function _sampleTerrainSlope(pixelX, pixelY, elevationRange) {
  const widthPixels = Math.max(1, terrainHeightWidth - 1);
  const heightPixels = Math.max(1, terrainHeightHeight - 1);
  const sample = (x, y) => {
    const xClamped = Math.max(0, Math.min(terrainHeightWidth - 1, x));
    const yClamped = Math.max(0, Math.min(terrainHeightHeight - 1, y));
    return (
      terrainHeightData[(yClamped * terrainHeightWidth + xClamped) * 4] / 255
    );
  };
  // Use larger sample span for more stable gradient computation
  const span = 3;
  const aspect = _lastMetadata?.scene_geometry?.aspect_ratio || 1;
  
  // Compute gradients using central difference
  const dx = (sample(pixelX + span, pixelY) - sample(pixelX - span, pixelY)) / (2 * span);
  const dy = (sample(pixelX, pixelY + span) - sample(pixelX, pixelY - span)) / (2 * span);
  
  // Convert to real-world slope in degrees
  // Gradient magnitude in normalized depth units per pixel
  const gradientMagnitude = Math.sqrt(dx * dx + dy * dy);
  
  // Scale by elevation range and account for aspect ratio
  const realWorldGradient = gradientMagnitude * elevationRange;
  
  // Convert to degrees: slope = atan(gradient)
  const slopeDegrees = (Math.atan(realWorldGradient) * 180) / Math.PI;
  
  return Math.max(0, slopeDegrees);
}

function _terrainGeometry(metadata, image) {
  const aspect =
    metadata?.scene_geometry?.aspect_ratio || image?.width / image?.height || 1;
  const width = 8;
  const depth = 8 / Math.max(aspect, 0.01);
  // Fixed 512×512 segments for consistent, high-quality mesh
  return { width, depth, segmentsX: 512, segmentsY: 512 };
}

function _heightNormalMap(image) {
  const canvas = document.createElement("canvas");
  const size = 256;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, size, size);
  const source = ctx.getImageData(0, 0, size, size).data;
  const normal = new Uint8Array(size * size * 3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const at = (y * size + x) * 4;
      const left = source[(y * size + Math.max(0, x - 1)) * 4] / 255;
      const right = source[(y * size + Math.min(size - 1, x + 1)) * 4] / 255;
      const up = source[(Math.max(0, y - 1) * size + x) * 4] / 255;
      const down = source[(Math.min(size - 1, y + 1) * size + x) * 4] / 255;
      const nx = (left - right) * 2.0;
      const ny = (up - down) * 2.0;
      const nz = 1.0;
      const length = Math.hypot(nx, ny, nz);
      const out = (y * size + x) * 3;
      normal[out] = ((nx / length) * 0.5 + 0.5) * 255;
      normal[out + 1] = ((ny / length) * 0.5 + 0.5) * 255;
      normal[out + 2] = ((nz / length) * 0.5 + 0.5) * 255;
    }
  }
  const texture = new THREE.DataTexture(normal, size, size, THREE.RGBFormat);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.needsUpdate = true;
  return texture;
}

function _waterNormalMap() {
  const size = 64;
  const data = new Uint8Array(size * size * 3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const wave = Math.sin(x * 0.45) * Math.cos(y * 0.31) * 0.16;
      const at = (y * size + x) * 3;
      data[at] = 128 + wave * 255;
      data[at + 1] = 128 - wave * 180;
      data[at + 2] = 255;
    }
  }
  const texture = new THREE.DataTexture(data, size, size, THREE.RGBFormat);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 4);
  texture.needsUpdate = true;
  return texture;
}

function _onResize() {
  if (!renderer) return;
  const wrap = renderer.domElement.parentElement;
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function _onUserInteract() {
  if (isFlying) {
    isFlying = false;
    controls.enabled = true;
    // Notify UI
    document.dispatchEvent(new CustomEvent("dw:orbitMode"));
  }
}

function _addStars() {
  const count = 800;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 50;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 50;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 50;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({
    color: 0xffffff,
    size: 0.04,
    transparent: true,
    opacity: 0.6,
  });
  scene.add(new THREE.Points(geo, mat));
}
