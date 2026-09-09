"""
Module D: Point Cloud & Surface Mesh Generator
===============================================

Takes the calibrated metric DSM (from Module B) and the original RGB image
and produces:

  1. A colored point cloud   → exported as .ply for download
  2. A surface-reconstructed triangle mesh with RGB vertex colors
     → exported as .glb (binary GLTF) for Three.js overlay rendering

Two computation backends are supported, selected automatically:

  Backend A — Open3D  (preferred, requires open3d)
    • Colored RGBD point cloud
    • Poisson surface reconstruction (depth=8) for watertight, smooth mesh
    • Density filtering to remove border artefacts

  Backend B — NumPy + SciPy + Trimesh  (always-available fallback)
    • Explicit height-field mesh via Delaunay triangulation on a sub-sampled grid
    • Vertex colors baked directly from the RGB image (no nearest-neighbour needed)
    • No external binary dependencies beyond the already-required SciPy + Trimesh

Backend A produces higher-quality Poisson meshes with sharp features; Backend B
is fast, deterministic, and works on any Python version.

Coordinate convention (both backends export in the same normalized space)
--------------------------------------------------------------------------
  X ∈ [-1,  1]   image columns (left → right)
  Y ∈ [ 0,  1]   normalized elevation (ground → max)
  Z ∈ [-1,  1]   image rows  (top → bottom)

The Three.js loadMeshOverlay() call scales this to match the terrain plane:
  scale.set(4, dispScale, 4)

Graceful degradation
--------------------
If neither backend can produce a mesh, is_available() returns False and
main.py skips Module D, returning null URLs — the frontend degrades silently.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional dependency guards ──────────────────────────────────────────────

try:
    import open3d as o3d  # type: ignore
    _O3D_OK = True
    logger.info("[PointCloudGenerator] open3d available — using Poisson backend.")
except ImportError:
    _O3D_OK = False
    logger.info(
        "[PointCloudGenerator] open3d not available — falling back to SciPy Delaunay backend. "
        "For best results install open3d (Python ≤ 3.12): pip install open3d"
    )

try:
    import trimesh  # type: ignore
    _TRIMESH_OK = True
except ImportError:
    _TRIMESH_OK = False
    logger.warning(
        "[PointCloudGenerator] trimesh not installed — GLB export disabled. "
        "Install with: pip install trimesh"
    )

try:
    from scipy.spatial import Delaunay  # type: ignore
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ── Main class ───────────────────────────────────────────────────────────────

class PointCloudGenerator:
    """
    Module D: Point Cloud & Surface Mesh Generator.

    Usage:
        if PointCloudGenerator.is_available():
            gen = PointCloudGenerator()
            pcd_data, mesh_data = gen.generate(rgb_image, metric_dsm)
            PointCloudGenerator.export_glb(mesh_data, path)
            PointCloudGenerator.export_ply(pcd_data, path)
    """

    # ── Tunable parameters ─────────────────────────────────────────────────

    # Open3D backend
    MAX_POINTS_O3D: int  = 250_000
    POISSON_DEPTH:  int  = 8
    DENSITY_TRIM:   float = 0.05   # remove lowest 5 % density vertices

    # SciPy/Delaunay backend
    MAX_GRID_RES:   int  = 300     # maximum grid resolution (each axis) for fallback
    # → max triangles ≈ 2 × MAX_GRID_RES² = 180 000

    @staticmethod
    def is_available() -> bool:
        """
        Returns True if at least one mesh backend AND the GLB exporter are ready.
        Open3D backend: open3d installed.
        Fallback backend: scipy + trimesh installed.
        """
        glb_ok   = _TRIMESH_OK
        mesh_ok  = _O3D_OK or (_SCIPY_OK and _TRIMESH_OK)
        return glb_ok and mesh_ok

    # ------------------------------------------------------------------
    # Public: generate
    # ------------------------------------------------------------------

    def generate(
        self,
        rgb_image:  np.ndarray,
        metric_dsm: np.ndarray,
        max_points: Optional[int] = None,
    ) -> Tuple[object, object]:
        """
        Build a colored point cloud and triangle surface mesh.

        Args:
            rgb_image:   uint8 RGB array of shape (H, W, 3).
            metric_dsm:  float32 metric elevation array of shape (H, W).
            max_points:  (Open3D backend only) cap before Poisson.
                         Defaults to MAX_POINTS_O3D.

        Returns:
            Tuple (pcd_data, mesh_data) where the types depend on the backend:
              Open3D:   (o3d.PointCloud, o3d.TriangleMesh)
              Fallback: (dict,           trimesh.Trimesh)
        """
        if not self.is_available():
            raise RuntimeError(
                "No mesh backend available. "
                "Install open3d (Python ≤ 3.12) or scipy + trimesh."
            )

        if _O3D_OK:
            return self._generate_open3d(
                rgb_image, metric_dsm,
                max_points=max_points or self.MAX_POINTS_O3D
            )
        else:
            return self._generate_scipy(rgb_image, metric_dsm)

    # ------------------------------------------------------------------
    # Backend A — Open3D / Poisson
    # ------------------------------------------------------------------

    def _generate_open3d(
        self,
        rgb_image:  np.ndarray,
        metric_dsm: np.ndarray,
        max_points: int = 250_000,
    ) -> Tuple["o3d.geometry.PointCloud", "o3d.geometry.TriangleMesh"]:
        t0 = time.perf_counter()
        H, W = metric_dsm.shape

        # ── Elevation stats ────────────────────────────────────────────────
        min_elev   = float(np.nanmin(metric_dsm))
        max_elev   = float(np.nanmax(metric_dsm))
        elev_range = max(max_elev - min_elev, 1.0)

        # ── Build normalized point grid ────────────────────────────────────
        rows, cols = np.mgrid[0:H, 0:W]
        x = ((cols.astype(np.float32) / max(W - 1, 1)) - 0.5) * 2.0  # [-1, 1]
        z = ((rows.astype(np.float32) / max(H - 1, 1)) - 0.5) * 2.0  # [-1, 1]
        y = (metric_dsm.astype(np.float32) - min_elev) / elev_range   # [ 0, 1]
        y = np.where(np.isfinite(y), y, 0.0)

        points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
        colors = (rgb_image.reshape(-1, 3) / 255.0).astype(np.float64)

        logger.info(f"[PointCloudGenerator/O3D] Raw grid: {len(points):,} points")

        # ── Uniform subsample ──────────────────────────────────────────────
        if len(points) > max_points:
            step   = max(1, len(points) // max_points)
            points = points[::step]
            colors = colors[::step]
            logger.info(f"[PointCloudGenerator/O3D] Step-sampled → {len(points):,} points")

        # ── Create Open3D PCD ──────────────────────────────────────────────
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = pcd.voxel_down_sample(voxel_size=0.004)
        logger.info(f"[PointCloudGenerator/O3D] After voxel downsample: {len(pcd.points):,}")

        # ── Normals (orient upward — nadir satellite) ──────────────────────
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=30)
        )
        pcd.orient_normals_toward_camera_location(camera_location=[0.0, 20.0, 0.0])

        # ── Poisson reconstruction ─────────────────────────────────────────
        logger.info(f"[PointCloudGenerator/O3D] Poisson (depth={self.POISSON_DEPTH}) …")
        t_p = time.perf_counter()
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.POISSON_DEPTH, n_threads=-1
        )
        logger.info(
            f"[PointCloudGenerator/O3D] Poisson done in {time.perf_counter()-t_p:.1f}s — "
            f"{len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris"
        )

        # ── Density filter ─────────────────────────────────────────────────
        dens = np.asarray(densities)
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, self.DENSITY_TRIM))
        logger.info(
            f"[PointCloudGenerator/O3D] After density filter: "
            f"{len(mesh.vertices):,} verts"
        )

        # ── Bake vertex colors ─────────────────────────────────────────────
        mesh = self._bake_vertex_colors_o3d(mesh, pcd)
        mesh.compute_vertex_normals()

        logger.info(
            f"[PointCloudGenerator/O3D] Total: {time.perf_counter()-t0:.1f}s"
        )
        return pcd, mesh

    @staticmethod
    def _bake_vertex_colors_o3d(
        mesh: "o3d.geometry.TriangleMesh",
        pcd:  "o3d.geometry.PointCloud",
    ) -> "o3d.geometry.TriangleMesh":
        """Nearest-neighbour color transfer from PCD → mesh vertices."""
        pcd_colors = np.asarray(pcd.colors)
        mesh_verts = np.asarray(mesh.vertices)

        if len(pcd_colors) == 0:
            mesh.paint_uniform_color([0.6, 0.6, 0.6])
            return mesh

        tree    = o3d.geometry.KDTreeFlann(pcd)
        vc      = np.empty((len(mesh_verts), 3), dtype=np.float64)
        for i, v in enumerate(mesh_verts):
            _, idx, _ = tree.search_knn_vector_3d(v, 1)
            vc[i] = pcd_colors[idx[0]] if idx else [0.5, 0.5, 0.5]
        mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
        return mesh

    # ------------------------------------------------------------------
    # Backend B — SciPy Delaunay (always-available fallback)
    # ------------------------------------------------------------------

    def _generate_scipy(
        self,
        rgb_image:  np.ndarray,
        metric_dsm: np.ndarray,
    ) -> Tuple[dict, "trimesh.Trimesh"]:
        """
        Builds a dense height-field mesh using SciPy's Delaunay triangulation
        on a regularly-spaced grid subsampled from the full DSM.

        The grid resolution is capped at MAX_GRID_RES × MAX_GRID_RES.
        Vertex colors are sampled directly from the nearest pixel in the RGB
        image, so no KD-tree search is needed.

        Returns:
            (pcd_dict, trimesh.Trimesh)
            pcd_dict has keys: 'points' (Nx3 float32), 'colors' (Nx3 float32 [0,1])
        """
        t0 = time.perf_counter()
        H, W = metric_dsm.shape

        # ── Choose grid resolution ─────────────────────────────────────────
        step_h = max(1, H // self.MAX_GRID_RES)
        step_w = max(1, W // self.MAX_GRID_RES)
        # Build the sampling grid
        row_idx = np.arange(0, H, step_h)
        col_idx = np.arange(0, W, step_w)
        GH, GW  = len(row_idx), len(col_idx)
        logger.info(
            f"[PointCloudGenerator/SciPy] Grid: {GH}×{GW} "
            f"(step_h={step_h}, step_w={step_w})"
        )

        # ── Elevation stats ────────────────────────────────────────────────
        dsm_sub  = metric_dsm[np.ix_(row_idx, col_idx)]
        min_elev = float(np.nanmin(dsm_sub))
        max_elev = float(np.nanmax(dsm_sub))
        elev_rng = max(max_elev - min_elev, 1.0)

        # ── Normalized vertex coordinates ──────────────────────────────────
        # X: columns → [-1, 1]
        # Z: rows    → [-1, 1]
        # Y: elevation → [0, 1]
        col_norm = (col_idx / max(W - 1, 1) - 0.5) * 2.0   # (GW,)
        row_norm = (row_idx / max(H - 1, 1) - 0.5) * 2.0   # (GH,)

        # Build vertex arrays
        col_grid, row_grid = np.meshgrid(col_norm, row_norm)  # (GH, GW)
        x_flat = col_grid.ravel().astype(np.float32)
        z_flat = row_grid.ravel().astype(np.float32)

        y_sub   = (dsm_sub - min_elev) / elev_rng
        y_sub   = np.where(np.isfinite(y_sub), y_sub, 0.0)
        y_flat  = y_sub.ravel().astype(np.float32)

        vertices = np.stack([x_flat, y_flat, z_flat], axis=-1)  # (N, 3)

        # ── Vertex colors from RGB image ───────────────────────────────────
        # Sample the image at grid positions (row_idx, col_idx)
        col_grid_px, row_grid_px = np.meshgrid(col_idx, row_idx)  # pixel indices
        rgb_sub = rgb_image[row_grid_px.ravel(), col_grid_px.ravel()]  # (N, 3) uint8
        colors  = (rgb_sub / 255.0).astype(np.float32)

        # ── Delaunay triangulation on (X, Z) ──────────────────────────────
        # We triangulate in 2D (X, Z) and use Y for height.
        logger.info("[PointCloudGenerator/SciPy] Running Delaunay triangulation …")
        t_d = time.perf_counter()
        xz  = vertices[:, [0, 2]]   # (N, 2)
        tri = Delaunay(xz)
        faces = tri.simplices.astype(np.int32)   # (M, 3)
        logger.info(
            f"[PointCloudGenerator/SciPy] Delaunay done in "
            f"{time.perf_counter()-t_d:.1f}s — "
            f"{len(vertices):,} verts, {len(faces):,} tris"
        )

        # ── Compute per-vertex normals ────────────────────────────────────
        # Use cross-product of edge vectors for each face, accumulate per vertex
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0).astype(np.float32)   # (M, 3) face normals

        # Ensure all face normals point upwards (+Y direction)
        down_mask = fn[:, 1] < 0
        if np.any(down_mask):
            faces[down_mask] = faces[down_mask][:, [0, 2, 1]]  # flip winding
            fn[down_mask] = -fn[down_mask]

        fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
        fn_len = np.where(fn_len > 1e-12, fn_len, 1.0)
        fn /= fn_len

        vn = np.zeros_like(vertices)
        for i in range(3):
            np.add.at(vn, faces[:, i], fn)
        vn_len = np.linalg.norm(vn, axis=1, keepdims=True)
        vn_len = np.where(vn_len > 1e-12, vn_len, 1.0)
        vn /= vn_len

        # ── Build Trimesh ──────────────────────────────────────────────────
        colors_rgba = np.hstack([
            (np.clip(colors, 0, 1) * 255).astype(np.uint8),
            np.full((len(colors), 1), 255, dtype=np.uint8)
        ])
        t_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=vn,
            vertex_colors=colors_rgba,
            process=False,
        )

        # ── Build point cloud dict ─────────────────────────────────────────
        pcd_dict = {
            "points": vertices,   # (N, 3) float32  normalized
            "colors": colors,     # (N, 3) float32  [0, 1]
        }

        logger.info(
            f"[PointCloudGenerator/SciPy] Total: {time.perf_counter()-t0:.1f}s"
        )
        return pcd_dict, t_mesh

    # ------------------------------------------------------------------
    # Public: export_glb
    # ------------------------------------------------------------------

    @staticmethod
    def export_glb(mesh_data: object, output_filepath: str) -> str:
        """
        Export a mesh to binary GLTF (.glb).

        Accepts either:
          • open3d.geometry.TriangleMesh  (Open3D backend)
          • trimesh.Trimesh               (SciPy fallback backend)

        Args:
            mesh_data:        Mesh object from generate().
            output_filepath:  Destination .glb path.

        Returns:
            output_filepath on success.
        """
        if not _TRIMESH_OK:
            raise RuntimeError("trimesh is required for GLB export: pip install trimesh")

        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

        # ── Open3D mesh → Trimesh → GLB ────────────────────────────────────
        if _O3D_OK and isinstance(mesh_data, o3d.geometry.TriangleMesh):
            vertices  = np.asarray(mesh_data.vertices,       dtype=np.float64)
            faces     = np.asarray(mesh_data.triangles,      dtype=np.int32)
            normals   = np.asarray(mesh_data.vertex_normals, dtype=np.float64)
            col_float = np.asarray(mesh_data.vertex_colors,  dtype=np.float64)
            col_uint8 = (np.clip(col_float, 0, 1) * 255).astype(np.uint8)
            alpha     = np.full((len(col_uint8), 1), 255, dtype=np.uint8)
            col_rgba  = np.hstack([col_uint8, alpha])

            t_mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces,
                vertex_normals=normals, vertex_colors=col_rgba,
                process=False,
            )
            t_mesh.export(output_filepath, file_type="glb")

        # ── Trimesh mesh (from SciPy backend) ─────────────────────────────
        elif isinstance(mesh_data, trimesh.Trimesh):
            mesh_data.export(output_filepath, file_type="glb")

        else:
            raise TypeError(
                f"Unsupported mesh type: {type(mesh_data)}. "
                "Expected open3d.TriangleMesh or trimesh.Trimesh."
            )

        size_mb = os.path.getsize(output_filepath) / 1_048_576
        logger.info(f"[PointCloudGenerator] GLB: {output_filepath}  ({size_mb:.1f} MB)")
        return output_filepath

    # ------------------------------------------------------------------
    # Public: export_ply
    # ------------------------------------------------------------------

    @staticmethod
    def export_ply(pcd_data: object, output_filepath: str) -> str:
        """
        Export a point cloud to binary PLY with RGB colors.

        Accepts either:
          • open3d.geometry.PointCloud  (Open3D backend)
          • dict with keys 'points' (Nx3) and 'colors' (Nx3 [0,1])  (SciPy backend)

        Returns:
            output_filepath on success.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

        # ── Open3D PointCloud ──────────────────────────────────────────────
        if _O3D_OK and isinstance(pcd_data, o3d.geometry.PointCloud):
            o3d.io.write_point_cloud(output_filepath, pcd_data, write_ascii=False)

        # ── dict (SciPy backend) → write ASCII PLY header manually ────────
        elif isinstance(pcd_data, dict):
            pts = pcd_data["points"]           # (N, 3) float32
            col = pcd_data["colors"]           # (N, 3) float32 [0,1]
            rgb = (np.clip(col, 0, 1) * 255).astype(np.uint8)
            N   = len(pts)

            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {N}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            )
            # Interleave XYZ (float32) + RGB (uint8) per vertex
            xyzrgb = np.hstack([
                pts.astype(np.float32),
                rgb,
            ])
            # Pack as structured array: 3×f32 + 3×u8 = 15 bytes per vertex
            dtype = np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('r', np.uint8),   ('g', np.uint8),   ('b', np.uint8),
            ])
            packed = np.empty(N, dtype=dtype)
            packed['x'] = pts[:, 0].astype(np.float32)
            packed['y'] = pts[:, 1].astype(np.float32)
            packed['z'] = pts[:, 2].astype(np.float32)
            packed['r'] = rgb[:, 0]
            packed['g'] = rgb[:, 1]
            packed['b'] = rgb[:, 2]

            with open(output_filepath, 'wb') as f:
                f.write(header.encode('ascii'))
                f.write(packed.tobytes())
        else:
            raise TypeError(
                f"Unsupported pcd type: {type(pcd_data)}. "
                "Expected open3d.PointCloud or dict."
            )

        size_mb = os.path.getsize(output_filepath) / 1_048_576
        logger.info(f"[PointCloudGenerator] PLY: {output_filepath}  ({size_mb:.1f} MB)")
        return output_filepath
