---
name: Precision Geospatial HUD
colors:
  surface: '#0f131d'
  surface-dim: '#0f131d'
  surface-bright: '#353944'
  surface-container-lowest: '#0a0e18'
  surface-container-low: '#171b26'
  surface-container: '#1c1f2a'
  surface-container-high: '#262a35'
  surface-container-highest: '#313540'
  on-surface: '#dfe2f1'
  on-surface-variant: '#b9cacb'
  inverse-surface: '#dfe2f1'
  inverse-on-surface: '#2c303b'
  outline: '#849495'
  outline-variant: '#3b494b'
  surface-tint: '#00dbe9'
  primary: '#dbfcff'
  on-primary: '#00363a'
  primary-container: '#00f0ff'
  on-primary-container: '#006970'
  inverse-primary: '#006970'
  secondary: '#4edea3'
  on-secondary: '#003824'
  secondary-container: '#00a572'
  on-secondary-container: '#00311f'
  tertiary: '#faf3ff'
  on-tertiary: '#3c0091'
  tertiary-container: '#e1d2ff'
  on-tertiary-container: '#6d3bd7'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#7df4ff'
  primary-fixed-dim: '#00dbe9'
  on-primary-fixed: '#002022'
  on-primary-fixed-variant: '#004f54'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#e9ddff'
  tertiary-fixed-dim: '#d0bcff'
  on-tertiary-fixed: '#23005c'
  on-tertiary-fixed-variant: '#5516be'
  background: '#0f131d'
  on-background: '#dfe2f1'
  surface-variant: '#313540'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.01em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 26px
    letterSpacing: 0em
  body-lg:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0.01em
  mono-data-lg:
    fontFamily: JetBrains Mono
    fontSize: 16px
    fontWeight: '500'
    lineHeight: 22px
    letterSpacing: -0.01em
  mono-data-md:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: 0em
  mono-data-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '400'
    lineHeight: 14px
    letterSpacing: 0.02em
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 12px
    letterSpacing: 0.08em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-sm: 0.5rem
  margin: 1.5rem
  margin-sm: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style

This design system targets geospatial data scientists, drone survey pilots, photogrammetry specialists, and infrastructure engineering teams operating at millimeter precision. 

The aesthetic fuses **Tactile Technical HUD** with **Precision Dark Glassmorphism**:
- Deep volumetric spatial backdrops evoke night operations, deep sensor telemetry, and 3D point-cloud environments.
- Surfaces leverage razor-thin illuminated edges, frosted optic overlays, and high-frequency telemetry readouts.
- Visual cadence prioritizes high computational density, absolute signal-to-noise clarity, and zero cognitive drag during mission-critical field maneuvers.
- Interactive elements mimic calibrated instrumentation: tactile toggle clicks, precise coordinate reticles, and crisp luminescence states that signal spatial verification.

## Colors

The palette establishes a high-fidelity dark spectrum calibrated for low-light command centers and high-contrast sunlight field displays:

- **Primary (`#00F0FF` / Cyan Laser):** The lidar pulse. Deployed exclusively for active spatial selection, telemetry vectors, primary viewport reticles, and focused coordinate pins.
- **Secondary (`#10B981` / Radar Emerald):** RTK GPS fix lock, valid telemetry thresholds, operational drone link integrity, and optimal point cloud density states.
- **Tertiary (`#8B5CF6` / Deep Spectral Violet):** Base anchor for the elevation heatmap ramp, low-confidence boundaries, and volumetric mesh bounding volumes.
- **Neutrals (Obsidian to Slate Tiers):**
  - Viewport Void: `#070A11`
  - Base Canvas (`surface-ground`): `#0B0F19`
  - Floating Tooling & Dock Panels (`surface-panel`): `#111827` (with varying alpha states)
  - Surface Outlines & Reticle Dividers: `#1E293B`
  - Neutral High Text / Readouts: `#F8FAFC`
  - Secondary Readouts / Muted Axes: `#94A3B8`
- **Heatmap Ramp (Elevation & Density Analysis):** Continuous sequential gradient mapped strictly across programmatic stops:
  - Base Elevation: `#6366F1` (Indigo)
  - Low-Mid: `#06B6D4` (Cyan)
  - Median: `#10B981` (Emerald)
  - Mid-High: `#EAB308` (Amber)
  - Peak / Collision Hazard: `#EF4444` (Crimson)

## Typography

The type system separates analytical human prose from raw computational feed:

- **Interface UI & Prose (`Inter`):** Applied to view titles, action prompts, modal dialogues, and settings. Weights remain restrained between 400 and 600 to preserve precision without bloat.
- **Telemetry & Spatial Coordinates (`JetBrains Mono`):** Applied to all coordinate streams (WGS84, UTM, ECEF), altitude readouts (MSL/AGL), pitch/yaw/roll values, point density counts, layer meters, and timestamp logs.
- **Tabular Figures & Strict Alignment:** Monospaced styles enforce `tabular-nums` throughout so rapid data fluctuations do not cause visual horizontal jumping.
- **Uppercase Micro-Labels:** `label-caps` must be paired with uppercase transformation and tracking (+0.08em) for HUD telemetry categories (e.g., `LAT`, `LON`, `ALT`, `RTK FIX`, `DOP`).

## Layout & Spacing

The viewport layout employs an unconstrained spatial canvas layered beneath floating orthographic HUD docks:

- **Viewport Underlay:** The 3D WebGL/Lidar spatial canvas occupies 100vw × 100vh with no margin restrictions.
- **HUD Anchor System:** Floating control panels, telemetry headers, and orthomosaic timelines anchor to viewport edges with standard margins:
  - Desktop: `1.5rem` (`24px`) safe margin from canvas boundary.
  - Tablet/Mobile: `0.75rem` (`12px`) safe margin with collapsible docked drawers.
- **Internal Density System:** Components employ dense micro-spacing (`space-xs` to `space-md`) to ensure critical telemetry clusters fit on viewports without occluding drone video feeds or mesh point clouds.
- **Split Multi-Pane Panels:** GIS sidebars utilize a 12-column subgrid or fixed-width drawers (`320px` standard, `420px` inspection detail) with `0.5rem` interior component gutters.

## Elevation & Depth

Spatial depth does not rely on heavy dropped black shadows, which muddy geospatial point clouds. Instead, depth is established through **optical transmissivity and illuminated boundaries**:

1. **Level 0 (World Space):** Raw 3D point cloud, mesh tiles, satellite raster base layers.
2. **Level 1 (Dock Paneling):** 
   - Surface: `rgba(17, 24, 39, 0.75)`
   - Backdrop Blur: `16px`
   - Border: `1px solid rgba(30, 41, 59, 0.7)`
3. **Level 2 (Active Toolbars, Floating Modals):**
   - Surface: `rgba(15, 23, 42, 0.85)`
   - Backdrop Blur: `20px`
   - Border: `1px solid rgba(0, 240, 255, 0.25)`
   - Ambient Glow: `0 0 20px -4px rgba(0, 240, 255, 0.15)`
4. **Level 3 (Focused Overlays & Critical Alerts):**
   - Surface: `rgba(11, 15, 25, 0.95)`
   - Border: `1px solid rgba(0, 240, 255, 0.6)`
   - Inner Inset: `inset 0 0 8px rgba(0, 240, 255, 0.12)`
   - Drop Shadow: `0 8px 32px rgba(0, 0, 0, 0.6)`

## Shapes

The geometry of this design system reinforces industrial hardware and instrumentation:

- **Soft Technical Form (`roundedness: 1`):** Default radii are set to `4px` (`0.25rem`). Large panels sit at `8px` (`0.5rem`).
- **Precision Edges:** High-radius rounded pill structures are strictly forbidden for critical controls; crisp angular architecture keeps visual focus sharp and preserves maximum viewport area.
- **Chamfered HUD Reticles (Visual Motif):** Decorative telemetry crosshairs, compass bezels, and viewport framing corners use 45-degree micro-chamfers (2px to 4px) to accentuate military-grade aerospace instrumentation.

## Components

### Buttons & Trigger Instrumentation
- **Primary Action (Scan / Process / Execute):** Background: `#00F0FF`, Text: `#070A11` (`JetBrains Mono`, 600 weight). Hover: glow state `0 0 16px rgba(0, 240, 255, 0.45)`.
- **Secondary (Inspect / Layer Filter):** Background: `rgba(30, 41, 59, 0.6)`, Border: `1px solid rgba(0, 240, 255, 0.25)`, Text: `#F8FAFC`. Hover: Border shifts to full `#00F0FF`, background gains subtle `rgba(0, 240, 255, 0.08)` tint.
- **Destructive / Abort Mission:** Background: `rgba(239, 68, 68, 0.15)`, Border: `1px solid #EF4444`, Text: `#EF4444`.

### GIS Sliders & Gradient Range Selectors
- **Elevation Heatmap Slider:** Dual-point thumb controls over an 8px gradient track displaying the full dynamic range. Thumbs are 14px high-contrast vertical pins with real-time numeric value badges (`JetBrains Mono`, 11px) floating directly above during drag states.
- **Lidar Point Density / Opacity Trackers:** Minimal 4px track (`#1E293B`) filled with cyan glow (`#00F0FF`), featuring 10px circular knurled thumbs.

### Telemetry Badges & Chips
- **RTK / Link Status Chips:** Height `22px`, padding `0 8px`, monospaced text. 
  - RTK Fixed: `#10B981` border with a 6px pulsing emerald indicator dot.
  - Float / Search: `#EAB308` border with static amber dot.
  - Disconnected: `#EF4444` border with flashing alert state.

### Input Fields & Spatial Coordinate Groups
- **Coordinate Inputs (Lat, Lon, Alt):** Monospaced numeric entries with integrated pre-appended labels (`LAT`, `LON`) styled in `label-caps` (`#94A3B8`). 
- **Surface:** `rgba(15, 23, 42, 0.8)` with `1px solid #1E293B`. Focused state introduces active cyan stroke (`#00F0FF`) with zero exterior outline fuzz, keeping lines razor-thin.

### Data Cards & HUD Overlays
- **Spatial Metadata Containers:** Floating translucent cards featuring a top-left micro corner tick. Card headers display category metadata separated by hairline dividing borders (`#1E293B`).
- **Telemetry Readout Matrix:** Key-value pairs displayed with muted secondary labels stacked vertically above high-contrast, larger monospaced values for instantaneous scanability in high-vibration cockpit environments.

### Viewport Overlays & Reticles
- **Boresight / Center Crosshairs:** Ultra-thin 1px lines (`rgba(0, 240, 255, 0.4)`) with center target deadzone.
- **Altitude Ladder:** Vertical scale overlaying left-hand viewport boundary with calibrated hash marks every 10 meters and numeric text in `mono-data-sm`.