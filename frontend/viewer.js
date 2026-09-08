/**
 * DepthWizard — Three.js 3D Terrain Viewer
 * Loads a grayscale 8-bit heightmap PNG and drapes an optical texture over
 * a displaced PlaneGeometry. Includes auto-flythrough camera animation and
 * manual OrbitControls.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/* ------------------------------------------------------------------ */
/*  Scene globals                                                       */
/* ------------------------------------------------------------------ */
let renderer, scene, camera, controls;
let terrainMesh = null;
let animFrameId = null;

// Flythrough state
let isFlying = true;
let flyClock = new THREE.Clock();
let flyPathRadius = 4.5;
let flyPathHeight = 2.5;
let flySpeed = 0.10;          // radians per second

// Terrain parameters (updated from metadata)
let elevationScale = 1.5;     // displacement intensity multiplier
let ambientIntensity = 0.5;
let sunAngle = 45;

const SEGMENT_COUNT = 256;    // geometry resolution — higher = smoother but heavier

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
  const ambient = new THREE.AmbientLight(0x1a2744, ambientIntensity * 2);
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
  // Dispose old terrain
  if (terrainMesh) {
    scene.remove(terrainMesh);
    terrainMesh.geometry.dispose();
    terrainMesh.material.dispose();
    terrainMesh = null;
  }

  // Compute displacement scale from real-world elevation range
  const elevRange = metadata?.elevation_metrics?.elevation_range_meters || 50;
  // Map to scene units: 10m → 0.1 scene units; cap at 3 scene units
  const dispScale = Math.min((elevRange / 10) * 0.12, 3.0) * elevationScale;

  const loader = new THREE.TextureLoader();

  const [heightTex, colorTex] = await Promise.all([
    loader.loadAsync(heightmapUrl),
    loader.loadAsync(textureUrl),
  ]);

  heightTex.wrapS = heightTex.wrapT = THREE.ClampToEdgeWrapping;
  colorTex.wrapS = colorTex.wrapT = THREE.ClampToEdgeWrapping;
  colorTex.colorSpace = THREE.SRGBColorSpace;

  const geo = new THREE.PlaneGeometry(8, 8, SEGMENT_COUNT, SEGMENT_COUNT);
  geo.rotateX(-Math.PI / 2);

  const mat = new THREE.MeshStandardMaterial({
    map: colorTex,
    displacementMap: heightTex,
    displacementScale: dispScale,
    displacementBias: -dispScale * 0.4,
    roughness: 0.88,
    metalness: 0.04,
  });

  terrainMesh = new THREE.Mesh(geo, mat);
  terrainMesh.receiveShadow = true;
  terrainMesh.castShadow = false;
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
  if (terrainMesh) {
    scene.remove(terrainMesh);
    terrainMesh.geometry.dispose();
    terrainMesh.material.dispose();
    terrainMesh = null;
  }

  const elevRange = metadata?.elevation_metrics?.elevation_range_meters || 50;
  const dispScale = Math.min((elevRange / 10) * 0.12, 3.0) * elevationScale;

  const loader = new THREE.TextureLoader();
  const heightTex = await loader.loadAsync(heightmapUrl);
  heightTex.wrapS = heightTex.wrapT = THREE.ClampToEdgeWrapping;

  const geo = new THREE.PlaneGeometry(8, 8, SEGMENT_COUNT, SEGMENT_COUNT);
  geo.rotateX(-Math.PI / 2);

  // Hypsometric gradient: sea-blue → green → brown → snow-white
  const mat = new THREE.MeshStandardMaterial({
    color: 0x4a7c59,
    displacementMap: heightTex,
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
    const elevRange = 50; // default; could store last metadata
    const dispScale = Math.min((elevRange / 10) * 0.12, 3.0) * elevationScale;
    terrainMesh.material.displacementScale = dispScale;
    terrainMesh.material.displacementBias = -dispScale * 0.4;
  }
}

export function setFlySpeed(val) { flySpeed = val; }
export function setFlyHeight(val) { flyPathHeight = val; flyPathRadius = val * 1.8; }

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
/*  Helpers                                                             */
/* ------------------------------------------------------------------ */
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
