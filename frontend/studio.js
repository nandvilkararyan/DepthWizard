/**
 * DepthWizard — AI Calibration Studio (Page 1) Controller
 * =======================================================
 * Manages model selection, image upload, elevation calibration parameters,
 * the 2D Interactive Split / Side-by-Side Dual viewers, and task handoff
 * to the 3D Simulator.
 */

(function () {
  'use strict';

  // ── DOM References ────────────────────────────────────────────────────────
  // Header Dynamic Status
  const headerStatusBadge = document.getElementById('header-status-badge');
  const headerStatusDot = document.getElementById('header-status-dot');
  const headerStatusText = document.getElementById('header-status-text');

  // Input Image / Dropzone
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const rasterFilename = document.getElementById('raster-filename');
  const rasterDims = document.getElementById('raster-dims');
  const rasterStatusBadge = document.getElementById('raster-status-badge');

  // Model Selection
  const modelSelect = document.getElementById('model-select');
  const modelDesc = document.getElementById('model-desc');

  // Calibration Form Inputs
  const minAltInput = document.getElementById('min-alt');
  const maxAltInput = document.getElementById('max-alt');
  const tileSizeInput = document.getElementById('tile-size');
  const overlapRatioInput = document.getElementById('overlap-ratio');

  // Primary Actions
  const btnProcess = document.getElementById('btn-process');
  const btnLaunchSim = document.getElementById('btn-launch-sim');
  const btnLoadDemo = document.getElementById('btn-load-demo');

  // Viewport Container & Modes
  const viewportCanvas = document.getElementById('viewport-canvas');
  const splitViewContainer = document.getElementById('split-view-container');
  const dualViewContainer = document.getElementById('dual-view-container');
  const btnSplitMode = document.getElementById('btn-split-mode');
  const btnSideMode = document.getElementById('btn-side-mode');

  // Split View Layers
  const opticalLayer = document.getElementById('optical-layer');
  const splitOverlay = document.getElementById('split-overlay');
  const heatmapTexture = document.getElementById('heatmap-texture');
  const splitDivider = document.getElementById('split-divider');

  // Side-by-Side Dual Layers
  const dualOpticalLayer = document.getElementById('dual-optical-layer');
  const dualDepthLayer = document.getElementById('dual-depth-layer');
  const dualHudLegendMax = document.getElementById('dual-hud-legend-max');

  // Opacity & Color Ramps
  const opacitySlider = document.getElementById('opacity-slider');
  const opacityVal = document.getElementById('opacity-val');

  // Reticle / Point Probe
  const reticleInspector = document.getElementById('reticle-inspector');
  const reticleCoords = document.getElementById('reticle-coords');
  const reticleHeight = document.getElementById('reticle-height');
  const reticleDatum = document.getElementById('reticle-datum');
  const hoverProbeText = document.getElementById('hover-probe-text');

  // Bottom Stats Strip
  const telLatency = document.getElementById('tel-latency');
  const telModel = document.getElementById('tel-model');
  const hudMinElev = document.getElementById('hud-min-elev');
  const hudMeanElev = document.getElementById('hud-mean-elev');
  const hudMaxElev = document.getElementById('hud-max-elev');
  const hudLegendMax = document.getElementById('hud-legend-max');

  // ── State ─────────────────────────────────────────────────────────────────
  let selectedFile = null;
  let activeTask = null;
  let isDraggingDivider = false;
  let isSideBySide = false;
  let activeColorRamp = 'turbo';
  let splitPercentage = 50;

  // Clean model descriptions with NO parameter counts
  const MODEL_METADATA = {
    'depthwizard_finetuned': {
      name: 'DepthWizard Fine-Tuned (Recommended)',
      desc: 'Fine-tuned for satellite & aerial terrain with active reflective-water suppression.'
    },
    'depth_anything_v2_small': {
      name: 'Depth Anything V2 - Small',
      desc: 'Fast and lightweight for rapid depth estimation on CPU or edge hardware.'
    },
    'depth_anything_v2_base': {
      name: 'Depth Anything V2 - Base',
      desc: 'Standard balanced model for high-resolution elevation mapping.'
    },
    'depth_anything_v2_large': {
      name: 'Depth Anything V2 - Large / High',
      desc: 'High-capacity model capturing detailed geomorphic and structural terrain facets.'
    }
  };

  // ── Initialization ────────────────────────────────────────────────────────
  function initStudio() {
    loadModelsFromAPI();
    setupEventListeners();
    checkExistingSessionTask();
    setSystemStatus('ready', 'System Ready');
  }

  // ── Dynamic System Status Indicator ───────────────────────────────────────
  function setSystemStatus(state, message) {
    if (!headerStatusDot || !headerStatusText) return;

    if (state === 'ready') {
      headerStatusDot.className = 'w-2 h-2 rounded-full bg-secondary animate-pulse';
      headerStatusText.className = 'text-[11px] font-mono text-secondary font-medium';
      headerStatusText.textContent = message || 'System Ready';
    } else if (state === 'processing') {
      headerStatusDot.className = 'w-2 h-2 rounded-full bg-amber-400 animate-ping';
      headerStatusText.className = 'text-[11px] font-mono text-amber-400 font-medium';
      headerStatusText.textContent = message || 'Generating 3D Model...';
    } else if (state === 'success') {
      headerStatusDot.className = 'w-2 h-2 rounded-full bg-primary-container';
      headerStatusText.className = 'text-[11px] font-mono text-primary-container font-medium';
      headerStatusText.textContent = message || '3D Model Ready';
    } else if (state === 'error') {
      headerStatusDot.className = 'w-2 h-2 rounded-full bg-red-400';
      headerStatusText.className = 'text-[11px] font-mono text-red-400 font-medium';
      headerStatusText.textContent = message || 'Error';
    }
  }

  // ── Model Catalog ─────────────────────────────────────────────────────────
  async function loadModelsFromAPI() {
    try {
      const resp = await fetch('/api/models');
      if (!resp.ok) return;
      const data = await resp.json();
      if (data && data.models && data.models.length > 0) {
        modelSelect.innerHTML = '';
        data.models.forEach((m) => {
          const opt = document.createElement('option');
          opt.value = m.key;
          // Format cleanly without parameter text
          const meta = MODEL_METADATA[m.key];
          opt.textContent = meta ? meta.name : m.name.replace(/\(HF\)/g, '').trim();
          if (m.key === data.current_key || m.key === 'depthwizard_finetuned') {
            opt.selected = true;
          }
          modelSelect.appendChild(opt);
        });
        updateModelDescription();
      }
    } catch (e) {
      console.warn('[Studio] Fallback to static model catalog:', e);
    }
  }

  function updateModelDescription() {
    const key = modelSelect.value;
    const meta = MODEL_METADATA[key] || {
      name: 'Custom Model',
      desc: 'Selected neural depth estimation model.'
    };
    if (modelDesc) modelDesc.textContent = meta.desc;
  }

  // ── Event Listeners ───────────────────────────────────────────────────────
  function setupEventListeners() {
    // Model Select
    if (modelSelect) {
      modelSelect.addEventListener('change', updateModelDescription);
    }

    // Drag and Drop
    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());
      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('border-primary-container', 'bg-surface-container/60');
      });
      dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('border-primary-container', 'bg-surface-container/60');
      });
      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('border-primary-container', 'bg-surface-container/60');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          handleFileSelection(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          handleFileSelection(e.target.files[0]);
        }
      });
    }

    // Process CTA (Single click generates both depth map and 3D model)
    if (btnProcess) btnProcess.addEventListener('click', runPipeline);

    // Launch 3D Simulator
    if (btnLaunchSim) {
      btnLaunchSim.addEventListener('click', () => {
        if (activeTask && activeTask.task_id) {
          window.location.href = `/app/simulator.html?task_id=${encodeURIComponent(activeTask.task_id)}`;
        } else {
          window.location.href = '/app/simulator.html';
        }
      });
    }

    // Demo Sample Loader
    if (btnLoadDemo) btnLoadDemo.addEventListener('click', loadDemoScene);

    // Split Viewport Divider Dragging & Click Handling
    if (splitDivider && splitViewContainer) {
      splitDivider.addEventListener('mousedown', (e) => {
        isDraggingDivider = true;
        e.preventDefault();
      });

      // Allow dragging across window
      window.addEventListener('mouseup', () => (isDraggingDivider = false));
      window.addEventListener('mousemove', onMouseMove);

      // Also allow clicking directly on split view container to jump divider
      splitViewContainer.addEventListener('mousedown', (e) => {
        if (isSideBySide) return;
        isDraggingDivider = true;
        onMouseMove(e);
      });
    }

    // Viewport Hover Elevation Sampling
    if (viewportCanvas) {
      viewportCanvas.addEventListener('mousemove', onViewportHover);
      viewportCanvas.addEventListener('mouseleave', () => {
        if (reticleInspector) reticleInspector.style.display = 'none';
        if (hoverProbeText) hoverProbeText.textContent = 'Hover image for elevation';
      });
      viewportCanvas.addEventListener('mouseenter', () => {
        if (reticleInspector && (selectedFile || activeTask)) reticleInspector.style.display = 'flex';
      });
    }

    // Opacity Slider
    if (opacitySlider && opacityVal && heatmapTexture) {
      opacitySlider.addEventListener('input', (e) => {
        const val = e.target.value;
        opacityVal.textContent = val + '%';
        heatmapTexture.style.opacity = (val / 100).toString();
        if (dualDepthLayer) dualDepthLayer.style.opacity = (val / 100).toString();
      });
    }

    // Split vs Side-by-Side Dual Mode Buttons
    if (btnSplitMode && btnSideMode) {
      btnSplitMode.addEventListener('click', () => setViewerMode(false));
      btnSideMode.addEventListener('click', () => setViewerMode(true));
    }

    // Color Ramps
    const rampButtons = document.querySelectorAll('[data-ramp]');
    rampButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        rampButtons.forEach((b) => {
          b.className = 'px-2 py-0.5 rounded hover:bg-surface-container text-on-surface-variant font-mono text-xs';
        });
        btn.className = 'px-2 py-0.5 rounded bg-surface-container text-primary font-mono text-xs font-medium';
        applyColorRamp(btn.getAttribute('data-ramp'));
      });
    });
  }

  // ── File Selection Handling ───────────────────────────────────────────────
  function handleFileSelection(file) {
    selectedFile = file;
    const isTiff = file.name.endsWith('.tif') || file.name.endsWith('.tiff');

    if (rasterFilename) rasterFilename.textContent = file.name;
    if (rasterStatusBadge) {
      rasterStatusBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-secondary"></span>READY`;
    }

    const reader = new FileReader();
    reader.onload = function (evt) {
      const img = new Image();
      img.onload = function () {
        if (rasterDims) rasterDims.textContent = `${img.naturalWidth}×${img.naturalHeight}`;
        
        // Update Optical Layer Previews
        const imgUrl = `url('${evt.target.result}')`;
        if (opticalLayer) opticalLayer.style.backgroundImage = imgUrl;
        if (dualOpticalLayer) dualOpticalLayer.style.backgroundImage = imgUrl;
      };
      if (!isTiff) {
        img.src = evt.target.result;
      } else {
        if (rasterDims) rasterDims.textContent = 'GeoTIFF';
      }
    };
    reader.readAsDataURL(file);

    setSystemStatus('ready', 'Image Loaded');
  }

  // ── Pipeline Runner (/process) ────────────────────────────────────────────
  async function runPipeline() {
    if (!selectedFile) {
      alert('Please choose or drop an aerial/satellite image first.');
      return;
    }

    const modelKey = modelSelect.value;
    const minAlt = parseFloat(minAltInput ? minAltInput.value : 0) || 0.0;
    const maxAlt = parseFloat(maxAltInput ? maxAltInput.value : 500) || 500.0;
    const tileSize = parseInt(tileSizeInput ? tileSizeInput.value : 512, 10) || 512;
    const overlapRatio = parseFloat(overlapRatioInput ? overlapRatioInput.value : 0.20) || 0.20;

    // UI Loading state
    if (btnProcess) {
      btnProcess.disabled = true;
      btnProcess.innerHTML = `<span class="material-symbols-outlined text-[18px] animate-spin">refresh</span><span>Processing Pipeline...</span>`;
    }
    setSystemStatus('processing', 'Extracting Depth & 3D Mesh...');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model_id', modelKey);
    formData.append('min_alt', minAlt.toString());
    formData.append('max_alt', maxAlt.toString());
    formData.append('tile_size', tileSize.toString());
    formData.append('overlap_ratio', overlapRatio.toString());

    const t0 = performance.now();

    try {
      const resp = await fetch('/process', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server error HTTP ${resp.status}`);
      }

      const data = await resp.json();
      const elapsedMs = Math.round(performance.now() - t0);
      activeTask = data;

      // Save in sessionStorage so simulator page can immediately load it
      sessionStorage.setItem('depthwizard_current_task', JSON.stringify(data));

      renderPipelineResults(data, elapsedMs);
      setSystemStatus('success', '3D Model Ready');

    } catch (err) {
      alert(`Pipeline error: ${err.message}`);
      setSystemStatus('error', 'Processing Failed');
    } finally {
      if (btnProcess) {
        btnProcess.disabled = false;
        btnProcess.innerHTML = `<span class="material-symbols-outlined text-[18px]">bolt</span><span>Generate 3D Model</span>`;
      }
    }
  }

  // ── Render Results into HUD & 2D Viewport ──────────────────────────────────
  function renderPipelineResults(data, elapsedMs) {
    const urls = data.download_urls || {};
    const meta = data.metadata || {};
    const elevMetrics = meta.elevation_metrics || {};

    const minElev = elevMetrics.min_elevation_meters !== undefined ? elevMetrics.min_elevation_meters : (meta.calibration?.min_elevation_m ?? 0.0);
    const maxElev = elevMetrics.max_elevation_meters !== undefined ? elevMetrics.max_elevation_meters : (meta.calibration?.max_elevation_m ?? 500.0);
    const meanElev = ((minElev + maxElev) / 2.0).toFixed(1);
    const rangeElev = (maxElev - minElev).toFixed(1);

    // Update Textures on Both Viewports
    if (urls.optical_texture_png) {
      const optBg = `url('${urls.optical_texture_png}')`;
      if (opticalLayer) opticalLayer.style.backgroundImage = optBg;
      if (dualOpticalLayer) dualOpticalLayer.style.backgroundImage = optBg;
    }
    if (urls.depth_colorized_png) {
      const depthBg = `url('${urls.depth_colorized_png}')`;
      if (heatmapTexture) heatmapTexture.style.backgroundImage = depthBg;
      if (dualDepthLayer) dualDepthLayer.style.backgroundImage = depthBg;
    }

    // Update Stats Strip
    if (hudMinElev) hudMinElev.textContent = `${minElev.toFixed(1)}m`;
    if (hudMeanElev) hudMeanElev.textContent = `${meanElev}m`;
    if (hudMaxElev) hudMaxElev.textContent = `+${maxElev.toFixed(1)}m`;
    if (hudLegendMax) hudLegendMax.textContent = `+${maxElev.toFixed(0)}m`;
    if (dualHudLegendMax) dualHudLegendMax.textContent = `+${maxElev.toFixed(0)}m`;

    if (telLatency) telLatency.textContent = `${elapsedMs}ms`;
    if (telModel) telModel.textContent = data.model_used || data.model_key || 'DepthWizard';

    // Reveal and enable "Open 3D Simulator ➔" button
    if (btnLaunchSim) {
      btnLaunchSim.classList.remove('hidden');
      btnLaunchSim.disabled = false;
      btnLaunchSim.classList.remove('opacity-50', 'cursor-not-allowed');
    }

    // Update header navigation link to 3D Simulator with active task_id
    const navSimLink = document.querySelector('header nav a[href*="simulator.html"]');
    if (navSimLink && data.task_id) {
      navSimLink.href = `/app/simulator.html?task_id=${encodeURIComponent(data.task_id)}`;
    }
  }

  // ── Split Viewport Drag & Hover Math ───────────────────────────────────────
  function onMouseMove(e) {
    if (!isDraggingDivider || !splitViewContainer || isSideBySide) return;
    const rect = splitViewContainer.getBoundingClientRect();
    let posX = e.clientX - rect.left;
    posX = Math.max(10, Math.min(rect.width - 10, posX));
    splitPercentage = (posX / rect.width) * 100;

    splitDivider.style.left = splitPercentage + '%';
    splitOverlay.style.clipPath = `inset(0 0 0 ${splitPercentage}%)`;
  }

  function onViewportHover(e) {
    if (!reticleInspector || isDraggingDivider || !viewportCanvas) return;
    if (!selectedFile && !activeTask) {
      reticleInspector.style.display = 'none';
      return;
    }
    const rect = viewportCanvas.getBoundingClientRect();
    const relX = e.clientX - rect.left;
    const relY = e.clientY - rect.top;

    if (relX >= 0 && relX <= rect.width && relY >= 0 && relY <= rect.height) {
      reticleInspector.style.left = Math.min(relX, rect.width - 150) + 'px';
      reticleInspector.style.top = Math.min(relY, rect.height - 70) + 'px';

      if (reticleCoords) {
        reticleCoords.textContent = `X: ${Math.round(relX * 2)} Y: ${Math.round(relY * 2)}`;
      }

      let approxHeight = '0.0';
      if (activeTask && activeTask.metadata && activeTask.metadata.elevation_metrics) {
        const em = activeTask.metadata.elevation_metrics;
        const normY = 1.0 - (relY / rect.height);
        approxHeight = (em.min_elevation_meters + normY * (em.max_elevation_meters - em.min_elevation_meters)).toFixed(1);
      } else {
        const normY = 1.0 - (relY / rect.height);
        approxHeight = (normY * 85.0).toFixed(1);
      }

      if (reticleHeight) reticleHeight.textContent = `Height: +${approxHeight}m`;
      if (hoverProbeText) hoverProbeText.textContent = `Elevation: +${approxHeight}m`;
    }
  }

  // ── Viewer Mode: Interactive Split vs Side-by-Side Dual ───────────────────
  function setViewerMode(sideBySide) {
    isSideBySide = sideBySide;

    if (sideBySide) {
      // Show Dual View, Hide Split View
      if (splitViewContainer) splitViewContainer.classList.add('hidden');
      if (dualViewContainer) dualViewContainer.classList.remove('hidden');

      btnSideMode.className = 'px-2.5 py-1 rounded bg-surface-container text-primary-fixed-dim font-mono text-xs font-semibold flex items-center gap-1 transition-all';
      btnSplitMode.className = 'px-2.5 py-1 rounded hover:bg-surface-container text-on-surface-variant font-mono text-xs flex items-center gap-1 transition-all';
    } else {
      // Show Split View, Hide Dual View
      if (dualViewContainer) dualViewContainer.classList.add('hidden');
      if (splitViewContainer) splitViewContainer.classList.remove('hidden');

      btnSplitMode.className = 'px-2.5 py-1 rounded bg-surface-container text-primary-fixed-dim font-mono text-xs font-semibold flex items-center gap-1 transition-all';
      btnSideMode.className = 'px-2.5 py-1 rounded hover:bg-surface-container text-on-surface-variant font-mono text-xs flex items-center gap-1 transition-all';

      // Restore split percentage
      if (splitDivider && splitOverlay) {
        splitDivider.style.left = splitPercentage + '%';
        splitOverlay.style.clipPath = `inset(0 0 0 ${splitPercentage}%)`;
      }
    }
  }

  function applyColorRamp(rampName) {
    activeColorRamp = rampName;
    const filter = rampName === 'viridis'
      ? 'hue-rotate(60deg) saturate(1.2)'
      : rampName === 'magma'
      ? 'hue-rotate(240deg) saturate(1.4) contrast(1.1)'
      : rampName === 'raw'
      ? 'grayscale(100%) contrast(1.2)'
      : 'none';

    if (heatmapTexture) heatmapTexture.style.filter = filter;
    if (dualDepthLayer) dualDepthLayer.style.filter = filter;
  }

  // ── Demo Scene Loader ─────────────────────────────────────────────────────
  async function loadDemoScene() {
    if (btnLoadDemo) {
      btnLoadDemo.innerHTML = `<span class="material-symbols-outlined text-[15px] animate-spin">refresh</span><span>Loading...</span>`;
    }

    try {
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext('2d');

      const grad = ctx.createRadialGradient(256, 256, 20, 256, 256, 256);
      grad.addColorStop(0, '#f2ece1');
      grad.addColorStop(0.3, '#7d8a68');
      grad.addColorStop(0.7, '#4a5b39');
      grad.addColorStop(1, '#243a4a');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 512, 512);

      for (let i = 0; i < 3000; i++) {
        const x = Math.random() * 512;
        const y = Math.random() * 512;
        const b = Math.floor(Math.random() * 50);
        ctx.fillStyle = `rgba(${b},${b + 20},${b},0.3)`;
        ctx.fillRect(x, y, 2, 2);
      }

      canvas.toBlob(async (blob) => {
        const file = new File([blob], 'demo_alpine_summit_survey.png', { type: 'image/png' });
        handleFileSelection(file);
        if (btnLoadDemo) {
          btnLoadDemo.innerHTML = `<span class="material-symbols-outlined text-[15px] text-primary-container">mountain_flag</span><span>Load Sample</span>`;
        }
      }, 'image/png');

    } catch (err) {
      console.warn('Demo scene load error:', err);
      if (btnLoadDemo) {
        btnLoadDemo.innerHTML = `<span class="material-symbols-outlined text-[15px] text-primary-container">mountain_flag</span><span>Load Sample</span>`;
      }
    }
  }

  // ── Session Task Check ────────────────────────────────────────────────────
  function checkExistingSessionTask() {
    try {
      const stored = sessionStorage.getItem('depthwizard_current_task');
      if (stored) {
        const task = JSON.parse(stored);
        if (task && task.download_urls) {
          activeTask = task;
          renderPipelineResults(task, 0);
        }
      }
    } catch (_) {}
  }

  // Auto-init on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initStudio);
  } else {
    initStudio();
  }

})();
