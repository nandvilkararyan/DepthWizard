import os
import sys
import unittest
import numpy as np
import cv2
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from fastapi.testclient import TestClient

from app.config import OUTPUT_DIR
from app.utils.image_io import load_optical_image, load_reference_dem, GeoMetadata
from app.modules.depth_extractor import DepthExtractor
from app.modules.scale_calibrator import ScaleCalibrator, CalibrationResult
from app.modules.formatter import OutputFormatter
from app.main import app


class TestDSMPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = OUTPUT_DIR / "test_scratch"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # Create a synthetic 8-bit RGB PNG image (256x256)
        cls.png_path = str(cls.test_dir / "test_rgb.png")
        rgb_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(cls.png_path, cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))

        # Create a synthetic GeoTIFF image (128x128) with CRS EPSG:4326
        cls.geotiff_path = str(cls.test_dir / "test_geo.tif")
        cls.crs_wkt = "EPSG:4326"
        cls.transform = Affine(0.0001, 0, 77.0, 0, -0.0001, 28.0)

        geo_arr = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        with rasterio.open(
            cls.geotiff_path,
            "w",
            driver="GTiff",
            height=128,
            width=128,
            count=3,
            dtype="uint8",
            crs=CRS.from_string(cls.crs_wkt),
            transform=cls.transform,
        ) as dst:
            for b in range(3):
                dst.write(geo_arr[:, :, b], b + 1)

        # Create a synthetic low-res reference DEM GeoTIFF (64x64)
        cls.ref_dem_path = str(cls.test_dir / "test_ref_dem.tif")
        # True elevation equation: H_gt = 2.5 * relative_depth + 10.0
        ref_transform = Affine(0.0002, 0, 77.0, 0, -0.0002, 28.0)
        dem_data = (np.random.rand(64, 64) * 50.0 + 100.0).astype(np.float32)
        with rasterio.open(
            cls.ref_dem_path,
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=1,
            dtype="float32",
            crs=CRS.from_string(cls.crs_wkt),
            transform=ref_transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(dem_data, 1)

    def test_01_image_io(self):
        """Test image I/O and geospatial metadata extraction."""
        rgb, geo_meta = load_optical_image(self.png_path)
        self.assertEqual(rgb.shape, (256, 256, 3))
        self.assertFalse(geo_meta.is_georeferenced)

        rgb_geo, geo_meta_geo = load_optical_image(self.geotiff_path)
        self.assertEqual(rgb_geo.shape, (128, 128, 3))
        self.assertTrue(geo_meta_geo.is_georeferenced)
        self.assertIsNotNone(geo_meta_geo.crs)

    def test_02_hann_window(self):
        """Test 2D Hann window generation for tile blending."""
        win = DepthExtractor.create_2d_hann_window(512, 512)
        self.assertEqual(win.shape, (512, 512))
        self.assertTrue(np.all(win > 0.0))
        self.assertAlmostEqual(win[256, 256], 1.0, places=2)

    def test_03_scale_calibrator_ols(self):
        """Test Module B least-squares linear calibration against ground truth DEM."""
        calibrator = ScaleCalibrator()
        
        # Synthetic relative depth: D_pred in [0.0, 10.0]
        d_pred = np.linspace(0.0, 10.0, 100).reshape(10, 10).astype(np.float32)
        # True metric elevation: H_true = 3.0 * D_pred + 15.0 (alpha=3.0, beta=15.0)
        h_true = (3.0 * d_pred + 15.0).astype(np.float32)

        geo_meta = GeoMetadata(is_georeferenced=True, crs="EPSG:4326")
        dem_meta = GeoMetadata(is_georeferenced=True, crs="EPSG:4326", nodata=-9999.0)

        result = calibrator.calibrate_depth(
            predicted_depth=d_pred,
            geo_meta=geo_meta,
            reference_dem=h_true,
            reference_dem_meta=dem_meta
        )

        self.assertEqual(result.calibration_type, "reference_dem")
        self.assertAlmostEqual(result.alpha, 3.0, places=3)
        self.assertAlmostEqual(result.beta, 15.0, places=3)
        self.assertLess(result.rmse, 1e-3)
        self.assertLess(result.mae, 1e-3)

    def test_04_output_formatter(self):
        """Test Module C Unity 16-bit PNG, float32 GeoTIFF, and metadata JSON export."""
        dsm_mock = np.linspace(10.0, 60.0, 100).reshape(10, 10).astype(np.float32)
        calib_res = CalibrationResult(
            metric_dsm=dsm_mock,
            alpha=2.5,
            beta=10.0,
            min_elevation=10.0,
            max_elevation=60.0,
            elevation_range=50.0,
            rmse=0.05,
            mae=0.04,
            calibration_type="reference_dem"
        )
        geo_meta = GeoMetadata(is_georeferenced=True, crs="EPSG:4326", transform=(0.1, 0, 0, 0, -0.1, 0))

        png_out = str(self.test_dir / "out_unity16.png")
        tif_out = str(self.test_dir / "out_dsm.tif")
        json_out = str(self.test_dir / "out_meta.json")

        OutputFormatter.export_unity_16bit_png(dsm_mock, png_out)
        OutputFormatter.export_geotiff(dsm_mock, geo_meta, tif_out)
        meta = OutputFormatter.generate_metadata_json(calib_res, geo_meta, (10, 10), json_out)

        # Verify Unity 16-bit PNG file
        img16 = cv2.imread(png_out, cv2.IMREAD_UNCHANGED)
        self.assertEqual(img16.dtype, np.uint16)
        self.assertEqual(img16.min(), 0)
        self.assertEqual(img16.max(), 65535)

        # Verify GeoTIFF header
        with rasterio.open(tif_out) as src:
            self.assertEqual(src.dtypes[0], "float32")
            self.assertEqual(src.width, 10)
            self.assertEqual(src.height, 10)

        # Verify metadata JSON fields
        self.assertIn("elevation_metrics", meta)
        self.assertEqual(meta["elevation_metrics"]["min_elevation_meters"], 10.0)
        self.assertEqual(meta["elevation_metrics"]["max_elevation_meters"], 60.0)

    def test_05_fastapi_health_endpoint(self):
        """Test FastAPI /health diagnostic endpoint."""
        client = TestClient(app)
        res = client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("device", data)

    def test_06_fastapi_process_endpoint(self):
        """Test FastAPI POST /process endpoint with synthetic image upload."""
        client = TestClient(app)
        
        with open(self.png_path, "rb") as f_img:
            response = client.post(
                "/process",
                files={"file": ("test_rgb.png", f_img, "image/png")},
                data={
                    "min_alt": "0.0",
                    "max_alt": "50.0",
                    "tile_size": "512",
                    "overlap_ratio": "0.20"
                }
            )

        self.assertEqual(response.status_code, 200)
        json_resp = response.json()
        self.assertEqual(json_resp["status"], "success")
        self.assertIn("download_urls", json_resp)
        self.assertIn("heightmap_16bit_png", json_resp["download_urls"])
        self.assertIn("geotiff_dsm_32bit", json_resp["download_urls"])


if __name__ == "__main__":
    unittest.main()
