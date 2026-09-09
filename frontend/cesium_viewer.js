/**
 * DepthWizard — CesiumJS Photorealistic Geospatial Engine
 * =======================================================
 * Advanced geospatial 3D renderer & GIS analysis tool providing:
 *   - Photorealistic high-resolution optical satellite & depth map ground draping
 *   - Direct 3D GLB mesh terrain ground-clamping co-registered with globe terrain
 *   - Real Earth landmark location anchoring (Grand Canyon, Mt. Everest, Mt. Fuji, Yosemite, Swiss Alps, etc.)
 *   - Interactive 3D spatial & elevation profile measurement (distance, delta height, slope)
 *   - Dynamic solar time-of-day shadow simulation & lighting
 *   - Live model scale, elevation offset, and layer mode controls
 */

let cesiumViewer = null;
let currentMeshEntity = null;
let currentGroundDrapeEntity = null;
let currentImageryLayer = null;

// Interactive measurement state
let measureHandler = null;
let measurePoints = [];
let measureEntities = [];
let isMeasuring = false;

// Active scene state
let currentGlbUrl = null;
let currentOpticalUrl = null;
let currentDepthUrl = null;
let currentMetadata = null;
let currentDrapeMode = 'optical'; // 'optical', 'depth', 'mesh_only', 'globe_only'
let currentLandmarkKey = 'grand_canyon';

let currentParams = {
  heightOffset: 0,
  scaleMultiplier: 1.0,
  solarHour: 14.0
};

// Earth Landmark Presets (west, south, east, north, baseHeight in meters, default cam height)
export const LANDMARK_PRESETS = {
  auto: {
    name: 'Georeferenced Bounds (Auto)',
    bounds: [-112.18, 36.03, -112.10, 36.08],
    center: [-112.14, 36.055],
    baseHeight: 1500,
    camHeight: 3500,
    scaleMultiplier: 250.0
  },
  grand_canyon: {
    name: 'Grand Canyon National Park, Arizona',
    bounds: [-112.18, 36.03, -112.10, 36.08],
    center: [-112.14, 36.055],
    baseHeight: 1500,
    camHeight: 3500,
    scaleMultiplier: 300.0
  },
  mount_everest: {
    name: 'Mount Everest, Himalayas',
    bounds: [86.91, 27.97, 86.95, 28.01],
    center: [86.93, 27.99],
    baseHeight: 7500,
    camHeight: 11000,
    scaleMultiplier: 450.0
  },
  mount_fuji: {
    name: 'Mount Fuji, Japan',
    bounds: [138.71, 35.34, 138.75, 35.38],
    center: [138.73, 35.36],
    baseHeight: 2800,
    camHeight: 6000,
    scaleMultiplier: 350.0
  },
  yosemite: {
    name: 'Yosemite Valley, California',
    bounds: [-119.62, 37.71, -119.55, 37.76],
    center: [-119.585, 37.735],
    baseHeight: 1200,
    camHeight: 4000,
    scaleMultiplier: 300.0
  },
  meteor_crater: {
    name: 'Meteor Crater, Arizona',
    bounds: [-111.033, 35.020, -111.013, 35.035],
    center: [-111.023, 35.027],
    baseHeight: 1650,
    camHeight: 2800,
    scaleMultiplier: 180.0
  },
  swiss_alps: {
    name: 'Matterhorn, Swiss Alps',
    bounds: [7.64, 45.96, 7.68, 45.99],
    center: [7.66, 45.975],
    baseHeight: 3200,
    camHeight: 6500,
    scaleMultiplier: 350.0
  },
  hawaii_volcano: {
    name: 'Kilauea Crater, Hawaii',
    bounds: [-155.30, 19.39, -155.26, 19.43],
    center: [-155.28, 19.41],
    baseHeight: 1000,
    camHeight: 3500,
    scaleMultiplier: 280.0
  }
};

/**
 * Initializes the Cesium 3D Viewer inside the specified DOM container.
 */
export function initCesiumViewer(containerEl) {
  if (cesiumViewer && !cesiumViewer.isDestroyed()) return cesiumViewer;

  // Set default Cesium Ion access token
  if (window.Cesium && Cesium.Ion && !Cesium.Ion.defaultAccessToken) {
    Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJmMDA5NGZjMy1hODdiLTQ0OWUtOGI4MS0yMmE3ODc0NzgwMWUiLCJpZCI6MjU0MzEsImlhdCI6MTU4ODgwNTY3OX0.1Zg-rQc0q6oKzLwOQ9N5-8N0N6P5x6u0';
  }

  try {
    cesiumViewer = new Cesium.Viewer(containerEl, {
      animation: false,
      timeline: false,
      baseLayerPicker: true,
      geocoder: false,
      homeButton: true,
      sceneModePicker: true,
      navigationHelpButton: false,
      infoBox: false,
      selectionIndicator: false,
      shadows: true,
      terrainProvider: Cesium.createWorldTerrain ? Cesium.createWorldTerrain({
        requestWaterMask: true,
        requestVertexNormals: true
      }) : undefined
    });
  } catch (err) {
    console.warn('[CesiumViewer] Fallback viewer init:', err);
    cesiumViewer = new Cesium.Viewer(containerEl, {
      animation: false,
      timeline: false,
      baseLayerPicker: true,
      infoBox: false,
      selectionIndicator: false
    });
  }

  const scene = cesiumViewer.scene;
  scene.globe.enableLighting = true;
  scene.globe.showGroundAtmosphere = true;
  scene.fog.enabled = true;
  scene.highDynamicRange = true;

  console.log('[CesiumViewer] Geospatial graphics engine initialized.');
  return cesiumViewer;
}

/**
 * Loads the 3D GLB mesh model, optical satellite texture, and depth map into CesiumJS scene.
 */
export async function loadCesiumModel(glbUrl, textureUrl, depthUrl, metadata) {
  if (!cesiumViewer) return;

  currentGlbUrl = glbUrl;
  currentOpticalUrl = textureUrl;
  currentDepthUrl = depthUrl || textureUrl;
  currentMetadata = metadata;

  // Auto-detect georeferenced coordinates if present
  const isGeo = metadata?.geospatial_metadata?.is_georeferenced;
  const bounds = metadata?.geospatial_metadata?.bounds;

  if (isGeo && bounds && bounds.length >= 4) {
    LANDMARK_PRESETS.auto.bounds = bounds;
    LANDMARK_PRESETS.auto.center = [(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0];
    currentLandmarkKey = 'auto';
  }

  await updateCesiumScene();
}

/**
 * Re-renders the CesiumJS scene layers, ground draping, and 3D GLB model positioning.
 */
export async function updateCesiumScene() {
  if (!cesiumViewer) return;

  // Cleanup existing entities & layers
  if (currentMeshEntity) {
    cesiumViewer.entities.remove(currentMeshEntity);
    currentMeshEntity = null;
  }
  if (currentGroundDrapeEntity) {
    cesiumViewer.entities.remove(currentGroundDrapeEntity);
    currentGroundDrapeEntity = null;
  }
  if (currentImageryLayer) {
    cesiumViewer.imageryLayers.remove(currentImageryLayer);
    currentImageryLayer = null;
  }

  // Active landmark location bounds
  const preset = LANDMARK_PRESETS[currentLandmarkKey] || LANDMARK_PRESETS.grand_canyon;
  const bounds = preset.bounds;
  const west = bounds[0], south = bounds[1], east = bounds[2], north = bounds[3];
  const centerLon = preset.center ? preset.center[0] : (west + east) / 2.0;
  const centerLat = preset.center ? preset.center[1] : (south + north) / 2.0;
  const rect = Cesium.Rectangle.fromDegrees(west, south, east, north);

  const activeTexture = (currentDrapeMode === 'depth') ? (currentDepthUrl || currentOpticalUrl) : currentOpticalUrl;

  // 1. Drape high-res optical satellite image or depth map directly on Earth globe terrain
  if (activeTexture && (currentDrapeMode === 'optical' || currentDrapeMode === 'depth' || currentDrapeMode === 'globe_only')) {
    // Add classification ground rectangle entity
    currentGroundDrapeEntity = cesiumViewer.entities.add({
      name: 'DepthWizard Draped Satellite Surface',
      rectangle: {
        coordinates: rect,
        material: new Cesium.ImageMaterialProperty({
          image: activeTexture,
          transparent: true
        }),
        classificationType: Cesium.ClassificationType.BOTH
      }
    });

    // Add ImageryProvider layer for crisp tile rendering across globe zoom levels
    try {
      if (Cesium.SingleTileImageryProvider.fromUrl) {
        Cesium.SingleTileImageryProvider.fromUrl(activeTexture, { rectangle: rect })
          .then(provider => {
            if (cesiumViewer && !cesiumViewer.isDestroyed()) {
              currentImageryLayer = cesiumViewer.imageryLayers.addImageryProvider(provider);
            }
          })
          .catch(() => {});
      } else {
        const provider = new Cesium.SingleTileImageryProvider({
          url: activeTexture,
          rectangle: rect
        });
        currentImageryLayer = cesiumViewer.imageryLayers.addImageryProvider(provider);
      }
    } catch (err) {
      console.warn('[CesiumViewer] Imagery drape warning:', err);
    }
  }

  // 2. Position 3D GLB mesh clamped to ground at landmark coordinates
  if (currentGlbUrl && currentDrapeMode !== 'globe_only') {
    const baseAlt = (preset.baseHeight || 1000) + currentParams.heightOffset;
    const position = Cesium.Cartesian3.fromDegrees(centerLon, centerLat, baseAlt);

    const heading = Cesium.Math.toRadians(0);
    const pitch = 0;
    const roll = 0;
    const hpr = new Cesium.HeadingPitchRoll(heading, pitch, roll);
    const orientation = Cesium.Transforms.headingPitchRollQuaternion(position, hpr);

    const baseScale = preset.scaleMultiplier || 250.0;
    const finalScale = baseScale * currentParams.scaleMultiplier;

    currentMeshEntity = cesiumViewer.entities.add({
      name: 'DepthWizard 3D Terrain Mesh',
      position: position,
      orientation: orientation,
      model: {
        uri: currentGlbUrl,
        minimumPixelSize: 64,
        maximumScale: 50000,
        scale: finalScale,
        heightReference: Cesium.HeightReference.RELATIVE_TO_GROUND,
        shadows: Cesium.ShadowMode.ENABLED
      }
    });
  }

  // 3. Fly camera to landmark location with optimal perspective tilt
  const camDist = preset.camHeight || 4000;
  cesiumViewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(centerLon, centerLat - 0.02, camDist),
    orientation: {
      heading: Cesium.Math.toRadians(0),
      pitch: Cesium.Math.toRadians(-35),
      roll: 0
    },
    duration: 1.8
  });

  // Apply solar lighting
  setCesiumSolarTime(currentParams.solarHour);
}

/**
 * Updates model transform (height offset, scale) live without resetting camera.
 */
export function updateCesiumModelTransform({ heightOffset, scaleMultiplier }) {
  if (heightOffset !== undefined) currentParams.heightOffset = heightOffset;
  if (scaleMultiplier !== undefined) currentParams.scaleMultiplier = scaleMultiplier;

  if (!cesiumViewer || !currentMeshEntity) return;

  const preset = LANDMARK_PRESETS[currentLandmarkKey] || LANDMARK_PRESETS.grand_canyon;
  const bounds = preset.bounds;
  const centerLon = preset.center ? preset.center[0] : (bounds[0] + bounds[2]) / 2.0;
  const centerLat = preset.center ? preset.center[1] : (bounds[1] + bounds[3]) / 2.0;

  const baseAlt = (preset.baseHeight || 1000) + currentParams.heightOffset;
  currentMeshEntity.position = Cesium.Cartesian3.fromDegrees(centerLon, centerLat, baseAlt);

  const baseScale = preset.scaleMultiplier || 250.0;
  currentMeshEntity.model.scale = baseScale * currentParams.scaleMultiplier;
}

/**
 * Sets active Earth landmark location preset.
 */
export function setCesiumLandmark(landmarkKey) {
  if (!LANDMARK_PRESETS[landmarkKey]) return;
  currentLandmarkKey = landmarkKey;
  updateCesiumScene();
}

/**
 * Sets active globe ground drape layer mode ('optical', 'depth', 'mesh_only', 'globe_only').
 */
export function setCesiumDrapeMode(mode) {
  currentDrapeMode = mode;
  updateCesiumScene();
}

/**
 * Simulates solar position and dynamic terrain shadows for a specified hour of the day (6.0 to 20.0).
 */
export function setCesiumSolarTime(hourFloat) {
  currentParams.solarHour = hourFloat;
  if (!cesiumViewer) return;

  const now = new Date();
  const hour = Math.floor(hourFloat);
  const minute = Math.floor((hourFloat - hour) * 60);
  const utcDate = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), hour, minute, 0));

  cesiumViewer.clock.currentTime = Cesium.JulianDate.fromDate(utcDate);
  cesiumViewer.scene.globe.enableLighting = true;
  cesiumViewer.scene.globe.shadows = Cesium.ShadowMode.ENABLED;
}

/**
 * Toggles interactive 3D spatial measurement mode (distance, elevation delta, slope).
 */
export function toggleCesiumMeasurement(enable, onUpdateCallback) {
  if (!cesiumViewer) return;

  if (measureHandler) {
    measureHandler.destroy();
    measureHandler = null;
  }

  isMeasuring = enable;

  if (!enable) {
    clearCesiumMeasurement();
    return;
  }

  measureHandler = new Cesium.ScreenSpaceEventHandler(cesiumViewer.scene.canvas);

  measureHandler.setInputAction((click) => {
    const scene = cesiumViewer.scene;
    let cartesian = scene.pickPosition(click.position);
    if (!cartesian) {
      const ray = cesiumViewer.camera.getPickRay(click.position);
      cartesian = scene.globe.pick(ray, scene);
    }

    if (!cartesian) return;

    const carto = Cesium.Cartographic.fromCartesian(cartesian);
    const lon = Cesium.Math.toDegrees(carto.longitude);
    const lat = Cesium.Math.toDegrees(carto.latitude);
    const height = carto.height;

    const pObj = { cartesian, carto, lon, lat, height };
    measurePoints.push(pObj);

    // Drop pin marker
    const pEntity = cesiumViewer.entities.add({
      position: cartesian,
      point: {
        pixelSize: 12,
        color: Cesium.Color.fromCssColorString('#00d4ff'),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY
      },
      label: {
        text: `P${measurePoints.length} (${height.toFixed(1)}m)`,
        font: 'bold 12px Inter, sans-serif',
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, -18),
        disableDepthTestDistance: Number.POSITIVE_INFINITY
      }
    });
    measureEntities.push(pEntity);

    if (measurePoints.length >= 2) {
      const p1 = measurePoints[measurePoints.length - 2];
      const p2 = measurePoints[measurePoints.length - 1];

      // Draw polyline connecting points
      const lineEntity = cesiumViewer.entities.add({
        polyline: {
          positions: [p1.cartesian, p2.cartesian],
          width: 4,
          material: new Cesium.PolylineGlowMaterialProperty({
            glowPower: 0.25,
            color: Cesium.Color.fromCssColorString('#00d4ff')
          })
        }
      });
      measureEntities.push(lineEntity);

      // Calculations
      const dist = Cesium.Cartesian3.distance(p1.cartesian, p2.cartesian);
      const elevDelta = p2.height - p1.height;
      const slope = Math.atan2(Math.abs(elevDelta), dist) * (180 / Math.PI);

      if (onUpdateCallback) {
        onUpdateCallback({
          distanceMeters: dist,
          distanceKm: dist / 1000.0,
          elevDeltaMeters: elevDelta,
          slopeDegrees: slope,
          pointsCount: measurePoints.length
        });
      }
    }
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
}

/**
 * Clears active 3D measurements and removes visual point/line entities.
 */
export function clearCesiumMeasurement() {
  if (!cesiumViewer) return;
  measureEntities.forEach(e => cesiumViewer.entities.remove(e));
  measureEntities = [];
  measurePoints = [];
}

/**
 * Disposes the CesiumJS viewer instance and clears resources.
 */
export function disposeCesiumViewer() {
  if (measureHandler) {
    measureHandler.destroy();
    measureHandler = null;
  }
  if (cesiumViewer && !cesiumViewer.isDestroyed()) {
    cesiumViewer.destroy();
    cesiumViewer = null;
    currentMeshEntity = null;
    currentGroundDrapeEntity = null;
    currentImageryLayer = null;
    measureEntities = [];
    measurePoints = [];
  }
}
