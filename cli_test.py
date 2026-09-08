import sys
import os
import cv2
import numpy as np

from app.config import OUTPUT_DIR, DEFAULT_MODEL_ID
from app.utils.image_io import load_optical_image
from app.modules.depth_extractor import DepthExtractor
from app.modules.scale_calibrator import ScaleCalibrator
from app.modules.formatter import OutputFormatter


def test_single_image(image_path: str, min_alt: float = 0.0, max_alt: float = 50.0):
    """
    Runs Depth Anything v2 inference and metric calibration on a 2D input image,
    saving heightmaps, GeoTIFFs, metadata, and a side-by-side visual comparison image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Input image file '{image_path}' not found.")
        return

    print(f"\n=======================================================")
    print(f" Depth Anything v2 DSM Estimation Tester")
    print(f" Input Image: {image_path}")
    print(f"=======================================================\n")

    # 1. Load optical image
    print("[1/4] Loading image & extracting metadata...")
    rgb_image, geo_meta = load_optical_image(image_path)
    print(f"     Image Dimensions: {rgb_image.shape[1]}x{rgb_image.shape[0]} | Georeferenced: {geo_meta.is_georeferenced}")

    # 2. Run Module A: Single-View Depth Extraction
    print("[2/4] Running Depth Anything v2 (Tiled Hann Window Blending)...")
    extractor = DepthExtractor(model_id=DEFAULT_MODEL_ID)
    relative_depth = extractor.extract_depth(rgb_image, tile_size=512, overlap_ratio=0.20)

    # 3. Run Module B: Metric Calibration
    print("[3/4] Calibrating metric scale & shift...")
    calibrator = ScaleCalibrator(default_min_alt=min_alt, default_max_alt=max_alt)
    calib_res = calibrator.calibrate_depth(relative_depth, geo_meta)

    print(f"\n----- DSM Elevation Stats -----")
    print(f"  Min Elevation    : {calib_res.min_elevation:.2f} meters")
    print(f"  Max Elevation    : {calib_res.max_elevation:.2f} meters")
    print(f"  Elevation Range  : {calib_res.elevation_range:.2f} meters")
    print(f"  Scale Alpha      : {calib_res.alpha:.4f}")
    print(f"  Shift Beta       : {calib_res.beta:.4f} meters")
    print(f"-------------------------------\n")

    # 4. Save formatted outputs
    print("[4/4] Formatting & saving outputs to 'output/'...")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    unity_png_path = str(OUTPUT_DIR / f"{base_name}_unity16.png")
    geotiff_path = str(OUTPUT_DIR / f"{base_name}_dsm.tif")
    metadata_json_path = str(OUTPUT_DIR / f"{base_name}_meta.json")

    OutputFormatter.export_unity_16bit_png(calib_res.metric_dsm, unity_png_path)
    OutputFormatter.export_geotiff(calib_res.metric_dsm, geo_meta, geotiff_path)
    meta = OutputFormatter.generate_metadata_json(calib_res, geo_meta, rgb_image.shape[:2], metadata_json_path)

    print(f"  [SUCCESS] Exported Unity 16-bit Heightmap PNG: {unity_png_path}")
    print(f"  [SUCCESS] Exported 32-bit Float GeoTIFF DSM : {geotiff_path}")
    print(f"  [SUCCESS] Exported Calibration Metadata JSON: {metadata_json_path}")

    # Create depth colormap preview for visual side-by-side display using OpenCV
    norm_depth = (calib_res.metric_dsm - calib_res.min_elevation) / (calib_res.elevation_range if calib_res.elevation_range > 0 else 1.0)
    depth_uint8 = (norm_depth * 255.0).astype(np.uint8)
    depth_colored_bgr = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    optical_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Resize depth preview if needed to match height
    if optical_bgr.shape[:2] != depth_colored_bgr.shape[:2]:
        depth_colored_bgr = cv2.resize(depth_colored_bgr, (optical_bgr.shape[1], optical_bgr.shape[0]))

    # Stack optical image and Turbo colormap depth side-by-side
    comparison_img = np.hstack([optical_bgr, depth_colored_bgr])
    preview_path = str(OUTPUT_DIR / f"{base_name}_side_by_side_preview.png")
    cv2.imwrite(preview_path, comparison_img)

    print(f"  [SUCCESS] Exported Side-by-Side Preview     : {preview_path}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_p = sys.argv[1]
    else:
        # Default test fallback
        img_p = str(OUTPUT_DIR / "test_scratch" / "test_rgb.png")
        if not os.path.exists(img_p):
            os.makedirs(os.path.dirname(img_p), exist_ok=True)
            arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            cv2.imwrite(img_p, arr)

    test_single_image(img_p)
