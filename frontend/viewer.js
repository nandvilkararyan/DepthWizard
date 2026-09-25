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
let _lastTerrainWidth = 8.0;
let _lastTerrainDepth = 8.0;

// Flythrough state
let isFlying = false; // disabled by default for better manual 3D control
let flyClock = new THREE.Clock();
let flyPathRadius = 6.0;
let flyPathHeight = 4.0;
let flySpeed = 0.1; // radians per second

// Walk mode (FPS Ground Walk) state
let isWalkMode = false;
let walkYaw = 0;           // horizontal camera rotation (radians)
let walkPitch = 0;         // vertical camera rotation (radians)
let walkEyeHeight = 0.035; // eye height in Three.js world units (~1.7m human height)
let walkSpeedBase = 0.022; // units per frame (standard walking speed)
let _lastValidGroundY = null;
let walkBobTimer = 0;
const _walkRaycaster = new THREE.Raycaster();
const _walkRayOrigin = new THREE.Vector3();
const _downVec = new THREE.Vector3(0, -1, 0);
const _walkFwd = new THREE.Vector3();
const _walkRight = new THREE.Vector3();

// Terrain parameters (updated from metadata)
let elevationScale = 1.0;
let sunAngle = 45;

/* ── Mesh overlay state ───────────────────────────────────────────── */
let meshOverlayGroup = null; // THREE.Group holding the loaded GLB
let meshOverlayVisible = false;

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
    antialias: false,           // disabled — too expensive; visual quality is fine without it
    alpha: false,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // cap at 1.5 — avoids 4× fill rate on HiDPI
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;   // BasicShadowMap is fastest; PCF is a good middle ground
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.3;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070b17);
  scene.fog = new THREE.FogExp2(0x070b17, 0.04);  // light initial fog; adjusted after terrain loads

  // Camera — use window dimensions as safe fallback since the canvas has no
  // pixel dimensions yet (it uses CSS 'w-full h-full' which resolves to 0 at
  // script execution time before the first layout paint).
  const initW = canvasEl.clientWidth  || window.innerWidth;
  const initH = canvasEl.clientHeight || window.innerHeight;
  camera = new THREE.PerspectiveCamera(
    55,
    initW / initH,
    0.01,
    200,                // generous initial far; _frameTerrain will update this
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
  sun.shadow.mapSize.set(1024, 1024);   // 2048 was too expensive; 1024 is indistinguishable at this scale
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
  function _onResize() {
    const parent = canvasEl.parentElement;
    const w = parent.clientWidth  || window.innerWidth;
    const h = parent.clientHeight || window.innerHeight;
    if (!w || !h) return;                       // guard against zero dimensions
    renderer.setSize(w, h, false);              // false = don't set canvas CSS size
    canvasEl.style.width  = w + 'px';
    canvasEl.style.height = h + 'px';
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  const ro = new ResizeObserver(() => _onResize());
  ro.observe(canvasEl.parentElement);
  _onResizeFn = _onResize;  // expose for external forceResize() calls

  // Defer first resize to the next paint so the browser has laid out the DOM
  // and the parent element has real pixel dimensions (avoids 0/0 aspect NaN).
  requestAnimationFrame(() => _onResize());

  // Mouse interaction → pause flythrough, enable orbit
  canvasEl.addEventListener("mousedown", _onUserInteract);
  canvasEl.addEventListener("touchstart", _onUserInteract, { passive: true });
  canvasEl.addEventListener("click", _inspectTerrain);

  // FPS Ground Walk mouse look with Pointer Lock
  function _onMouseMove(e) {
    if (!isWalkMode || !camera) return;
    if (document.pointerLockElement !== canvasEl) return;

    const movementX = e.movementX || e.mozMovementX || e.webkitMovementX || 0;
    const movementY = e.movementY || e.mozMovementY || e.webkitMovementY || 0;

    walkYaw -= movementX * 0.0022;
    walkPitch -= movementY * 0.0022;

    const maxPitch = Math.PI / 2.2; // ~81 degrees
    walkPitch = Math.max(-maxPitch, Math.min(maxPitch, walkPitch));

    camera.quaternion.setFromEuler(new THREE.Euler(walkPitch, walkYaw, 0, 'YXZ'));
  }
  document.addEventListener("mousemove", _onMouseMove);

  // Re-request pointer lock when clicking viewport in walk mode
  canvasEl.addEventListener("click", () => {
    if (isWalkMode && document.pointerLockElement !== canvasEl) {
      try {
        canvasEl.requestPointerLock();
      } catch (err) {}
    }
  });

  document.addEventListener("pointerlockchange", () => {
    const isLocked = document.pointerLockElement === canvasEl;
    document.dispatchEvent(new CustomEvent("dw:pointerLockChange", { detail: { locked: isLocked } }));
  });

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
/*  Public resize helper — call after canvas becomes visible           */
/* ------------------------------------------------------------------ */
let _onResizeFn = null;   // stored by initViewer so callers can trigger it

export function forceResize() {
  if (_onResizeFn) _onResizeFn();
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
  _lastTerrainWidth = width;
  _lastTerrainDepth = depth;
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
    wireframe: true,
  });

  terrainMesh = new THREE.Mesh(geo, mat);
  terrainMesh.receiveShadow = true;
  terrainMesh.castShadow = false;
  scene.add(terrainMesh);
  _frameTerrain(geo);
  _updateWalkScale(metadata);

  // Set camera to orbit mode by default for immediate responsive control
  flyClock = new THREE.Clock();
  isFlying = false;
  controls.enabled = true;

  // Adjust fog to terrain size — low density so wireframe/mesh is always visible
  scene.fog.density = 0.03;

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
  _lastTerrainWidth = width;
  _lastTerrainDepth = depth;
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
  _updateWalkScale(metadata);

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
 * @param {string}  glbUrl         – URL served by FastAPI /files/
 * @param {object}  metadata       – From the /process response
 * @param {number}  opacity        – Initial mesh opacity (default 0.10)
 * @param {boolean} initialVisible – Whether visible initially (default false)
 * @returns {Promise<THREE.Group>}
 */
export function loadMeshOverlay(glbUrl, metadata, opacity = 0.10, initialVisible = false) {
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
        // Terrain plane is width×depth units (centered at origin).
        // Scale X and Z to match the terrain plane end to end:
        // scaleX = width / 2, scaleZ = depth / 2.
        const aspect =
          metadata?.scene_geometry?.aspect_ratio ||
          _lastMetadata?.scene_geometry?.aspect_ratio ||
          (_lastTerrainWidth / _lastTerrainDepth) ||
          1.0;
        const width = _lastTerrainWidth || 8.0;
        const depth = _lastTerrainDepth || (8.0 / Math.max(aspect, 0.01));
        const scaleX = width / 2.0;
        const scaleZ = depth / 2.0;

        meshOverlayGroup.scale.set(scaleX, dispScale, scaleZ);
        // Match terrain displacement bias (the plane is shifted down by 40% of dispScale)
        meshOverlayGroup.position.set(0, -dispScale * 0.4, 0);

        // ── Materials ─────────────────────────────────────────────────
        meshOverlayGroup.traverse((child) => {
          if (!child.isMesh) return;

          // Replace with a MeshStandardMaterial that shows vertex colors if available, or vibrant cyan wireframe
          const hasColors = !!(child.geometry && child.geometry.attributes && child.geometry.attributes.color);
          const newMat = new THREE.MeshStandardMaterial({
            color: hasColors ? 0xffffff : 0x00f0ff,
            vertexColors: hasColors,
            roughness: 0.65,
            metalness: 0.05,
            transparent: true,
            opacity: opacity,
            depthWrite: opacity >= 0.99, // disable depth write when transparent
            wireframe: true, // Default wireframe mode enabled
            side: THREE.DoubleSide,
            polygonOffset: true,
            polygonOffsetFactor: -1,
            polygonOffsetUnits: -1,
          });

          if (child.material) {
            if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
            else child.material.dispose();
          }
          child.material = newMat;
          child.castShadow = true;
          child.receiveShadow = true;
        });

        meshOverlayVisible = initialVisible;
        meshOverlayGroup.visible = initialVisible;
        userMeshOpacity = opacity;
        scene.add(meshOverlayGroup);

        console.log(
          `[Viewer] Mesh overlay loaded: ${glbUrl}  ` +
            `(dispScale=${dispScale.toFixed(3)}, opacity=${opacity}, visible=${initialVisible})`,
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
 * Explicitly set the mesh overlay visibility.
 * @param {boolean} visible
 */
export function setMeshVisible(visible) {
  if (!meshOverlayGroup) return;
  meshOverlayVisible = !!visible;
  meshOverlayGroup.visible = meshOverlayVisible;
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
 * Switch the terrain plane to wireframe mode.
 * @param {boolean} on
 */
export function setTerrainWireframe(on) {
  if (terrainMesh && terrainMesh.material) {
    terrainMesh.material.wireframe = on;
    terrainMesh.material.needsUpdate = true;
  }
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
/*  Camera & Flight Controls API                                        */
/* ------------------------------------------------------------------ */
const _keysDown = {};

export function setKeyDown(key) {
  if (key) _keysDown[key.toLowerCase()] = true;
}

export function setKeyUp(key) {
  if (key) _keysDown[key.toLowerCase()] = false;
}

export function toggleFlythrough() {
  if (!isFlying && isWalkMode) setCameraWalk(false);
  isFlying = !isFlying;
  controls.enabled = !isFlying;
  if (isFlying) flyClock = new THREE.Clock();
  return isFlying;
}

export function setFlythrough(active) {
  if (active && isWalkMode) setCameraWalk(false);
  isFlying = active;
  controls.enabled = !active;
  if (active) flyClock = new THREE.Clock();
}

/**
 * Fast O(1) bilinear interpolation of terrain height at world coordinates (x, z).
 * Direct array lookup against displaced PlaneGeometry vertex positions — zero GC, 60fps stable.
 */
export function getTerrainElevationAt(x, z) {
  if (!terrainMesh || !terrainMesh.geometry) return null;
  const positions = terrainMesh.geometry.attributes.position;
  if (!positions) return null;

  const w = _lastTerrainWidth || 8.0;
  const d = _lastTerrainDepth || 8.0;
  const halfW = w / 2;
  const halfD = d / 2;

  const clampedX = Math.max(-halfW, Math.min(halfW, x));
  const clampedZ = Math.max(-halfD, Math.min(halfD, z));

  const u = (clampedX + halfW) / w;
  const v = (clampedZ + halfD) / d;

  const segX = 256;
  const segY = 256;
  const gx = THREE.MathUtils.clamp(u * segX, 0, segX);
  const gy = THREE.MathUtils.clamp(v * segY, 0, segY);

  const x0 = Math.floor(gx);
  const x1 = Math.min(segX, x0 + 1);
  const y0 = Math.floor(gy);
  const y1 = Math.min(segY, y0 + 1);

  const fx = gx - x0;
  const fy = gy - y0;

  const stride = segX + 1;
  const idx00 = y0 * stride + x0;
  const idx10 = y0 * stride + x1;
  const idx01 = y1 * stride + x0;
  const idx11 = y1 * stride + x1;

  if (idx11 >= positions.count) return null;

  const h00 = positions.getY(idx00);
  const h10 = positions.getY(idx10);
  const h01 = positions.getY(idx01);
  const h11 = positions.getY(idx11);

  return (h00 * (1 - fx) + h10 * fx) * (1 - fy) + (h01 * (1 - fx) + h11 * fx) * fy;
}

function _sampleGroundHeightRay(x, z) {
  if (!terrainMesh) return null;
  _walkRayOrigin.set(x, 50, z);
  _walkRaycaster.set(_walkRayOrigin, _downVec);
  const hits = _walkRaycaster.intersectObject(terrainMesh, false);
  if (hits.length > 0) {
    return hits[0].point.y;
  }
  return null;
}

function _updateWalkScale(metadata) {
  const meta = metadata || _lastMetadata;
  const widthM = Number(
    meta?.scene_geometry?.extent_width_m ||
    ((meta?.calibration?.gsd_m || 0) * (meta?.scene_geometry?.width_pixels || 0)) ||
    0
  );
  const terrainW = _lastTerrainWidth || 8.0;

  if (widthM > 0) {
    const unitsPerMeter = terrainW / widthM;
    walkEyeHeight = Math.max(0.025, Math.min(0.12, 1.7 * unitsPerMeter));
    walkSpeedBase = Math.max(0.015, Math.min(0.06, 1.4 * unitsPerMeter));
  } else {
    // Relative rDSM default: 8 units ≈ 500m -> 1.7m human eye height ≈ 0.035 units
    walkEyeHeight = 0.035;
    walkSpeedBase = 0.022;
  }
}

function _spawnWalkPlayer() {
  const w = _lastTerrainWidth || 8.0;
  const d = _lastTerrainDepth || 8.0;
  const margin = 0.4;
  const halfW = (w / 2) - margin;
  const halfD = (d / 2) - margin;

  let spawnX = camera.position.x;
  let spawnZ = camera.position.z;

  if (Math.abs(spawnX) > halfW || Math.abs(spawnZ) > halfD || !isFinite(spawnX) || !isFinite(spawnZ)) {
    spawnX = 0;
    spawnZ = 0;
  }

  const elev = getTerrainElevationAt(spawnX, spawnZ) ?? _sampleGroundHeightRay(spawnX, spawnZ) ?? 0;
  _lastValidGroundY = elev;

  camera.position.set(spawnX, elev + walkEyeHeight, spawnZ);
  walkPitch = 0;
  walkYaw = 0;
  camera.quaternion.setFromEuler(new THREE.Euler(walkPitch, walkYaw, 0, 'YXZ'));
}

export function setCameraWalk(active) {
  isWalkMode = active;
  if (active) {
    isFlying = false;
    if (controls) controls.enabled = false;
    _updateWalkScale();
    _spawnWalkPlayer();
    if (renderer && renderer.domElement) {
      try {
        renderer.domElement.requestPointerLock();
      } catch (e) {}
    }
  } else {
    if (document.pointerLockElement === renderer?.domElement) {
      try {
        document.exitPointerLock();
      } catch (e) {}
    }
    if (controls) controls.enabled = true;
  }
}

export function getIsWalkMode() {
  return isWalkMode;
}

export function getWalkTelemetry() {
  if (!isWalkMode) return null;
  const deg = Math.round(((THREE.MathUtils.radToDeg(-walkYaw) % 360) + 360) % 360);
  const compassDirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const dir = compassDirs[Math.floor((deg + 22.5) / 45) % 8];
  const pitchDeg = Math.round(THREE.MathUtils.radToDeg(walkPitch));

  const isMoving = _keysDown['w'] || _keysDown['s'] || _keysDown['a'] || _keysDown['d'];
  const isSprint = _keysDown['shift'];
  const speedMs = isMoving ? (isSprint ? 3.5 : 1.4) : 0.0;

  return {
    headingDeg: String(deg).padStart(3, '0'),
    headingDir: dir,
    eyeHeightM: (1.7).toFixed(1),
    speedMs: speedMs.toFixed(1),
    pitchDeg: pitchDeg >= 0 ? `+${pitchDeg}` : `${pitchDeg}`,
  };
}

export function resetCamera() {
  if (isWalkMode) setCameraWalk(false);
  isFlying = false;
  if (controls) {
    controls.enabled = true;
    controls.target.set(0, 0, 0);
  }
  if (camera) {
    camera.position.set(5, 4, 5);
    camera.lookAt(0, 0, 0);
  }
  if (controls) controls.update();
}

export function setCameraNadir() {
  if (isWalkMode) setCameraWalk(false);
  isFlying = false;
  if (controls) {
    controls.enabled = true;
    controls.target.set(0, 0, 0);
  }
  if (camera) {
    camera.position.set(0, 9, 0.001);
    camera.lookAt(0, 0, 0);
  }
  if (controls) controls.update();
}

export async function setTerrainTexture(url) {
  if (!terrainMesh || !url) return;
  const loader = new THREE.TextureLoader();
  try {
    const tex = await loader.loadAsync(url);
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    if (terrainMesh.material) {
      terrainMesh.material.map = tex;
      terrainMesh.material.needsUpdate = true;
    }
  } catch (err) {
    console.warn("[Viewer] Failed to switch terrain texture:", err);
  }
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
      const aspect =
        _lastMetadata?.scene_geometry?.aspect_ratio ||
        (_lastTerrainWidth / _lastTerrainDepth) ||
        1.0;
      const width = _lastTerrainWidth || 8.0;
      const depth = _lastTerrainDepth || (8.0 / Math.max(aspect, 0.01));
      meshOverlayGroup.scale.set(width / 2.0, dispScale, depth / 2.0);
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

    _updateWalkScale(_lastMetadata);
    if (isWalkMode && camera) {
      const elev = getTerrainElevationAt(camera.position.x, camera.position.z);
      if (elev !== null) camera.position.y = elev + walkEyeHeight;
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

// Pre-allocated vectors — avoids `new THREE.Vector3()` every frame (GC pressure)
const _fwdVec = new THREE.Vector3();
const _rightVec = new THREE.Vector3();

function _startRenderLoop() {
  if (animFrameId) cancelAnimationFrame(animFrameId);

  // Target 60 FPS max (≈16.67ms per frame). On 120 Hz monitors this halves
  // GPU load with no perceptible visual difference for a terrain viewer.
  const TARGET_MS = 1000 / 60;
  let _lastFrameTime = 0;

  function tick(now) {
    animFrameId = requestAnimationFrame(tick);

    // Frame rate cap — skip render if we're ahead of schedule
    const elapsed = now - _lastFrameTime;
    if (elapsed < TARGET_MS - 1) return;   // -1ms tolerance for timer jitter
    _lastFrameTime = now - (elapsed % TARGET_MS);

    const hasFlightKeys = !!(_keysDown['w'] || _keysDown['s'] || _keysDown['a'] || _keysDown['d'] || _keysDown['q'] || _keysDown['e']);

    if (isWalkMode && camera) {
      // ── Ground Walk (Pedestrian First-Person Mode) ───────────────────
      _walkFwd.set(-Math.sin(walkYaw), 0, -Math.cos(walkYaw)).normalize();
      _walkRight.set(Math.cos(walkYaw), 0, -Math.sin(walkYaw)).normalize();

      const isSprint = !!_keysDown['shift'];
      const speedMultiplier = (flySpeed || 0.1) / 0.1;
      const currentSpeed = (walkSpeedBase || 0.022) * speedMultiplier * (isSprint ? 2.2 : 1.0);

      let moveX = 0;
      let moveZ = 0;
      if (_keysDown['w']) {
        moveX += _walkFwd.x;
        moveZ += _walkFwd.z;
      }
      if (_keysDown['s']) {
        moveX -= _walkFwd.x;
        moveZ -= _walkFwd.z;
      }
      if (_keysDown['a']) {
        moveX -= _walkRight.x;
        moveZ -= _walkRight.z;
      }
      if (_keysDown['d']) {
        moveX += _walkRight.x;
        moveZ += _walkRight.z;
      }

      const moveLen = Math.hypot(moveX, moveZ);
      if (moveLen > 0.0001) {
        camera.position.x += (moveX / moveLen) * currentSpeed;
        camera.position.z += (moveZ / moveLen) * currentSpeed;
        walkBobTimer += (isSprint ? 0.22 : 0.14);
      } else {
        walkBobTimer = 0;
      }

      // Boundary clamp to terrain footprint
      const margin = 0.15;
      const halfW = (_lastTerrainWidth / 2) - margin;
      const halfD = (_lastTerrainDepth / 2) - margin;
      camera.position.x = Math.max(-halfW, Math.min(halfW, camera.position.x));
      camera.position.z = Math.max(-halfD, Math.min(halfD, camera.position.z));

      // Terrain following (Gravity & Elevation)
      const groundElev = getTerrainElevationAt(camera.position.x, camera.position.z)
        ?? _sampleGroundHeightRay(camera.position.x, camera.position.z);

      if (groundElev !== null) {
        _lastValidGroundY = groundElev;
      }

      if (_lastValidGroundY !== null) {
        const bob = Math.sin(walkBobTimer) * (walkEyeHeight * 0.06);
        const targetY = _lastValidGroundY + walkEyeHeight + bob;
        // Smooth lerp to ground level
        camera.position.y += (targetY - camera.position.y) * 0.4;
      }
    } else if (hasFlightKeys && camera) {
      const speed = Math.max(0.04, (flySpeed || 0.1) * 1.5);
      camera.getWorldDirection(_fwdVec);
      _rightVec.crossVectors(_fwdVec, camera.up).normalize();

      if (_keysDown['w']) camera.position.addScaledVector(_fwdVec, speed);
      if (_keysDown['s']) camera.position.addScaledVector(_fwdVec, -speed);
      if (_keysDown['a']) camera.position.addScaledVector(_rightVec, -speed);
      if (_keysDown['d']) camera.position.addScaledVector(_rightVec, speed);
      if (_keysDown['q']) camera.position.y += speed;
      if (_keysDown['e']) camera.position.y -= speed;
      if (controls && controls.enabled) {
        controls.target.addScaledVector(_fwdVec, (_keysDown['w'] ? speed : 0) - (_keysDown['s'] ? speed : 0));
      }
    } else if (isFlying) {
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

  tick(0);
}

/* ------------------------------------------------------------------ */
/*  Private helpers                                                     */
/* ------------------------------------------------------------------ */

let currentViewMode = "combined"; // 'mesh-only', 'threejs-only', 'combined'
let userMeshOpacity = 0.10;

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
  const minElev = Number(
    metadata?.elevation_metrics?.min_elevation_meters ??
    metadata?.calibration?.min_elevation_m ??
    0
  );
  const maxElev = Number(
    metadata?.elevation_metrics?.max_elevation_meters ??
    metadata?.calibration?.max_elevation_m ??
    100
  );
  const elevRange = Math.max(maxElev - minElev, 1);

  // Use backend's suggested_disp_scale if present (preferred — it's calibrated to scene type).
  // Fallback: target ~20% height-to-width ratio on the 8-unit terrain plane (dispScale ≈ 1.5).
  // The old formula (0.35 + range/500 * 0.45) always produced near-minimum values (≈0.35–0.42)
  // for typical relative rDSM ranges, making all terrain appear flat.
  const baseDisp = metadata?.elevation_metrics?.suggested_disp_scale
    ?? Math.min(2.5, Math.max(0.8, 0.8 + (elevRange / 100.0) * 1.2));
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

  // Multi-pass 3×3 mean filter to remove pixel noise & spiky needle artifacts.
  // 2 passes of a 3×3 kernel is equivalent to a wider smooth but 4× faster than
  // the previous 4-pass 5×5 (100M ops → ~18M ops on a 1024² image).
  for (let pass = 0; pass < 2; pass++) {
    const smoothed = new Float32Array(imgW * imgH);
    for (let y = 0; y < imgH; y++) {
      for (let x = 0; x < imgW; x++) {
        let sum = 0;
        let count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
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
  camera.updateProjectionMatrix();   // CRITICAL: must call after changing near/far
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
  if (isWalkMode) return;
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
  // 256×256 segments gives 66K vertices — ¼ of the old 512×512 (263K verts).
  // Visual quality is identical since height values are sampled from a smooth
  // 8-bit image; extra segments add no new detail at this scale.
  return { width, depth, segmentsX: 256, segmentsY: 256 };
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
  if (isWalkMode) return;
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
