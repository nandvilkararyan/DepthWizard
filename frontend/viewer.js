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

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader }    from 'three/addons/loaders/GLTFLoader.js';

/* ------------------------------------------------------------------ */
/*  Scene globals                                                       */
/* ------------------------------------------------------------------ */
let renderer, scene, camera, controls;
let terrainMesh     = null;
let animFrameId     = null;
let _lastMetadata   = null;   // saved for dispScale re-calc by setElevationScale

// Flythrough state
let isFlying       = true;
let flyClock       = new THREE.Clock();
let flyPathRadius  = 4.5;
let flyPathHeight  = 2.5;
let flySpeed       = 0.10;          // radians per second

// Terrain parameters (updated from metadata)
let elevationScale = 1.5;
let sunAngle       = 45;

const SEGMENT_COUNT = 256;    // geometry resolution — higher = smoother but heavier

/* ── Mesh overlay state ───────────────────────────────────────────── */
let meshOverlayGroup   = null;   // THREE.Group holding the loaded GLB
let meshOverlayVisible = true;

/* ------------------------------------------------------------------ */
/*  Init                                                                */
/* ------------------------------------------------------------------ */
export function initViewer(canvasEl) {
  renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070b17);
  scene.fog = new THREE.FogExp2(0x070b17, 0.12);

  // Camera
  camera = new THREE.PerspectiveCamera(55, canvasEl.clientWidth / canvasEl.clientHeight, 0.01, 100);
  camera.position.set(0, flyPathHeight, flyPathRadius);
  camera.lookAt(0, 0, 0);

  // OrbitControls
  controls = new OrbitControls(camera, canvasEl);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI / 2.05;
  controls.minDistance = 1;
  controls.maxDistance = 12;
  controls.enabled = false; // disabled until user grabs mouse

  // Lights
  const ambient = new THREE.AmbientLight(0x1a2744, 1.0);
  ambient.name = 'ambient';
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xfff5e0, 2.2);
  sun.name = 'sun';
  sun.position.set(3, 5, 2);
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
  canvasEl.addEventListener('mousedown', _onUserInteract);
  canvasEl.addEventListener('touchstart', _onUserInteract, { passive: true });

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
  // Also dispose any stale mesh overlay (new image loaded)
  disposeMeshOverlay();

  const { dispScale } = _computeDispScale(metadata);

  const loader = new THREE.TextureLoader();

  const [heightTex, colorTex] = await Promise.all([
    loader.loadAsync(heightmapUrl),
    loader.loadAsync(textureUrl),
  ]);

  heightTex.wrapS = heightTex.wrapT = THREE.ClampToEdgeWrapping;
  colorTex.wrapS  = colorTex.wrapT  = THREE.ClampToEdgeWrapping;
  colorTex.colorSpace = THREE.SRGBColorSpace;

  const geo = new THREE.PlaneGeometry(8, 8, SEGMENT_COUNT, SEGMENT_COUNT);
  geo.rotateX(-Math.PI / 2);

  const mat = new THREE.MeshStandardMaterial({
    map:              colorTex,
    displacementMap:  heightTex,
    displacementScale: dispScale,
    displacementBias: -dispScale * 0.4,
    roughness:  0.88,
    metalness:  0.04,
  });

  terrainMesh = new THREE.Mesh(geo, mat);
  terrainMesh.receiveShadow = true;
  terrainMesh.castShadow    = false;
  scene.add(terrainMesh);

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

  const { dispScale } = _computeDispScale(metadata);
  const loader        = new THREE.TextureLoader();
  const heightTex     = await loader.loadAsync(heightmapUrl);
  heightTex.wrapS = heightTex.wrapT = THREE.ClampToEdgeWrapping;

  const geo = new THREE.PlaneGeometry(8, 8, SEGMENT_COUNT, SEGMENT_COUNT);
  geo.rotateX(-Math.PI / 2);

  // Hypsometric gradient: green → brown → snow
  const mat = new THREE.MeshStandardMaterial({
    color:            0x4a7c59,
    displacementMap:  heightTex,
    displacementScale: dispScale,
    displacementBias: -dispScale * 0.4,
    roughness: 0.9,
    metalness: 0.02,
  });

  terrainMesh = new THREE.Mesh(geo, mat);
  terrainMesh.receiveShadow = true;
  scene.add(terrainMesh);

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
            roughness:    0.65,
            metalness:    0.05,
            transparent:  opacity < 1.0,
            opacity:      opacity,
            depthWrite:   opacity >= 0.99,   // disable depth write when transparent
          });

          // Keep original material side-setting if available
          if (child.material && child.material.side !== undefined) {
            newMat.side = THREE.DoubleSide;
          }

          child.material.dispose();
          child.material = newMat;
          child.castShadow    = true;
          child.receiveShadow = true;
        });

        scene.add(meshOverlayGroup);
        meshOverlayVisible = true;

        console.log(
          `[Viewer] Mesh overlay loaded: ${glbUrl}  ` +
          `(dispScale=${dispScale.toFixed(3)}, opacity=${opacity})`
        );
        resolve(meshOverlayGroup);
      },
      undefined,  // progress callback
      (err) => {
        console.error('[Viewer] GLB load failed:', err);
        reject(err);
      }
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
    child.material.opacity    = val;
    child.material.transparent = val < 1.0;
    child.material.depthWrite  = val >= 0.99;
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
    child.material.wireframe   = on;
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
  meshOverlayGroup   = null;
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
    terrainMesh.material.displacementScale = dispScale;
    terrainMesh.material.displacementBias  = -dispScale * 0.4;

    // Also re-scale the mesh overlay to stay registered
    if (meshOverlayGroup) {
      meshOverlayGroup.scale.set(4.0, dispScale, 4.0);
      meshOverlayGroup.position.y = -dispScale * 0.4;
    }
  }
}

export function setFlySpeed(val) { flySpeed = val; }
export function setFlyHeight(val) { flyPathHeight = val; flyPathRadius = val * 1.8; }

/** Returns the viewer identifier string for use by index.html */
export function getViewerType() { return 'threejs'; }

export function setSunAngle(degrees) {
  sunAngle = degrees;
  const rad = THREE.MathUtils.degToRad(degrees);
  const sun = scene.getObjectByName('sun');
  if (sun) {
    sun.position.set(Math.cos(rad) * 5, 5, Math.sin(rad) * 5);
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

    renderer.render(scene, camera);
  }

  tick();
}

/* ------------------------------------------------------------------ */
/*  Private helpers                                                     */
/* ------------------------------------------------------------------ */

let currentViewMode = 'combined'; // 'mesh-only', 'threejs-only', 'combined'
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
    terrainMesh.visible = (mode === 'threejs-only' || mode === 'combined');
  }
  if (meshOverlayGroup) {
    if (mode === 'mesh-only') {
      meshOverlayGroup.visible = true;
      setMeshOpacity(1.0);
    } else if (mode === 'threejs-only') {
      meshOverlayGroup.visible = false;
    } else if (mode === 'combined') {
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
  const elevRange = metadata?.elevation_metrics?.elevation_range_meters || 50;
  let baseScale = metadata?.elevation_metrics?.suggested_disp_scale;
  if (baseScale === undefined || baseScale === null) {
    baseScale = Math.min((elevRange / 10) * 0.12, 3.0);
  }
  const dispScale = baseScale * elevationScale;
  return { elevRange, dispScale };
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
    document.dispatchEvent(new CustomEvent('dw:orbitMode'));
  }
}

function _addStars() {
  const count = 800;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    positions[i * 3]     = (Math.random() - 0.5) * 50;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 50;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 50;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.04, transparent: true, opacity: 0.6 });
  scene.add(new THREE.Points(geo, mat));
}
