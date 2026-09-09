using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace DepthWizard.GIS
{
    #region Data Contracts for FastAPI Response
    [Serializable]
    public class ProcessResponse
    {
        public string task_id;
        public string status;
        public SceneMetadata metadata;
        public DownloadUrls download_urls;
    }

    [Serializable]
    public class SceneMetadata
    {
        public SceneGeometry scene_geometry;
        public ElevationMetrics elevation_metrics;
        public CalibrationParameters calibration_parameters;
        public GeospatialMetadata geospatial_metadata;
    }

    [Serializable]
    public class SceneGeometry
    {
        public int width_pixels;
        public int height_pixels;
        public float aspect_ratio;
    }

    [Serializable]
    public class ElevationMetrics
    {
        public float min_elevation_meters;
        public float max_elevation_meters;
        public float elevation_range_meters;
    }

    [Serializable]
    public class CalibrationParameters
    {
        public float scale_alpha;
        public float shift_beta_meters;
        public string calibration_type;
        public float rmse_meters;
        public float mae_meters;
    }

    [Serializable]
    public class GeospatialMetadata
    {
        public bool is_georeferenced;
        public string crs;
    }

    [Serializable]
    public class DownloadUrls
    {
        public string heightmap_16bit_png;
        public string heightmap_8bit_preview_png;
        public string optical_texture_png;
        public string depth_colorized_png;
        public string geotiff_dsm_32bit;
        public string calibration_metadata_json;
    }
    #endregion

    /// <summary>
    /// API Bridge & Pipeline Coordinator.
    /// Sends optical images to the local FastAPI backend (http://127.0.0.1:8000/process),
    /// parses scene metadata JSON, downloads textures, and triggers dynamic mesh generation.
    /// </summary>
    public class AppManager : MonoBehaviour
    {
        [Header("FastAPI Server Configuration")]
        [Tooltip("Local FastAPI backend URL endpoint.")]
        public string serverUrl = "http://127.0.0.1:8000/process";

        [Tooltip("Base URL host for file downloads.")]
        public string baseHostUrl = "http://127.0.0.1:8000";

        [Header("Target Image Input")]
        [Tooltip("Local file path to optical input image (PNG, JPG, or GeoTIFF).")]
        public string opticalImagePath = "";

        [Tooltip("Optional local file path to reference DEM (GeoTIFF).")]
        public string referenceDemPath = "";

        [Header("Processing Parameters")]
        public float fallbackMinElevation = 0f;
        public float fallbackMaxElevation = 50f;
        public int tileSize = 512;
        public float overlapRatio = 0.20f;

        [Header("Target Visualizer Components")]
        public TerrainMeshGenerator meshGenerator;
        public Material targetDisplacementMaterial;

        [Header("Status Diagnostics")]
        public string currentTaskId = "";
        public bool isProcessing = false;
        public string statusMessage = "Idle";

        private void Start()
        {
            if (meshGenerator == null)
            {
                meshGenerator = FindObjectOfType<TerrainMeshGenerator>();
            }
        }

        /// <summary>
        /// Public entry point to trigger DSM processing pipeline from a local file path.
        /// Used by the standalone/editor workflow.
        /// </summary>
        public void ProcessImagePipeline(string imagePath, string refDemPath = "")
        {
            if (isProcessing)
            {
                Debug.LogWarning("[AppManager] Processing already in progress...");
                return;
            }

            opticalImagePath = imagePath;
            referenceDemPath = refDemPath;

            StartCoroutine(CoProcessPipeline());
        }

        /// <summary>
        /// WebGL entry point: downloads textures directly from HTTP URLs and builds
        /// the 3D terrain. No local file access is required.
        /// Called by WebGLBridge.ReceivePayload() after the parent page POSTs to FastAPI.
        /// </summary>
        /// <param name="heightmapUrl">Absolute URL to the 16-bit heightmap PNG.</param>
        /// <param name="textureUrl">Absolute URL to the optical texture PNG.</param>
        /// <param name="meta">Scene metadata parsed from the FastAPI JSON response.</param>
        /// <param name="onComplete">Callback invoked when terrain is fully rendered.</param>
        /// <param name="onError">Callback invoked on failure, receives an error message.</param>
        public void LoadFromUrls(
            string heightmapUrl,
            string textureUrl,
            SceneMetadata meta,
            Action onComplete = null,
            Action<string> onError = null)
        {
            if (isProcessing)
            {
                Debug.LogWarning("[AppManager] LoadFromUrls called while processing is already active.");
                return;
            }
            StartCoroutine(CoLoadFromUrls(heightmapUrl, textureUrl, meta, onComplete, onError));
        }

        private IEnumerator CoLoadFromUrls(
            string heightmapUrl,
            string textureUrl,
            SceneMetadata meta,
            Action onComplete,
            Action<string> onError)
        {
            isProcessing = true;
            statusMessage = "Downloading terrain assets from server...";
            Debug.Log($"[AppManager] CoLoadFromUrls — heightmap: {heightmapUrl}");

            Texture2D heightmapTexture = null;
            Texture2D opticalTexture   = null;

            yield return StartCoroutine(CoDownloadTexture(heightmapUrl, tex => heightmapTexture = tex));
            yield return StartCoroutine(CoDownloadTexture(textureUrl,   tex => opticalTexture   = tex));

            if (heightmapTexture == null || opticalTexture == null)
            {
                string err = "Failed to download one or more terrain textures.";
                statusMessage = err;
                Debug.LogError($"[AppManager] {err}");
                isProcessing = false;
                onError?.Invoke(err);
                yield break;
            }

            float minElev    = meta?.elevation_metrics?.min_elevation_meters  ?? 0f;
            float maxElev    = meta?.elevation_metrics?.max_elevation_meters   ?? 50f;
            float heightRange = meta?.elevation_metrics?.elevation_range_meters ?? 50f;

            // Update material
            if (targetDisplacementMaterial != null)
            {
                targetDisplacementMaterial.SetTexture("_MainTex",    opticalTexture);
                targetDisplacementMaterial.SetTexture("_HeightMap",  heightmapTexture);
                targetDisplacementMaterial.SetFloat("_HeightScale",  heightRange);
                targetDisplacementMaterial.SetFloat("_MinElevation", minElev);
            }

            // Rebuild mesh and bake CPU heights for physics
            if (meshGenerator != null)
            {
                meshGenerator.GenerateGridMesh(
                    meshGenerator.resolutionX,
                    meshGenerator.resolutionZ,
                    meshGenerator.meshSizeX,
                    meshGenerator.meshSizeZ
                );
                meshGenerator.ApplyHeightmapToMesh(heightmapTexture, heightRange, minElev);
            }

            statusMessage = "WebGL terrain loaded successfully!";
            Debug.Log($"[AppManager] {statusMessage}");
            isProcessing = false;
            onComplete?.Invoke();
        }

        private IEnumerator CoProcessPipeline()
        {
            if (!File.Exists(opticalImagePath))
            {
                statusMessage = $"Error: Optical image file not found at {opticalImagePath}";
                Debug.LogError($"[AppManager] {statusMessage}");
                yield break;
            }

            isProcessing = true;
            statusMessage = "Uploading image to FastAPI server...";
            Debug.Log($"[AppManager] {statusMessage}");

            WWWForm form = new WWWForm();
            byte[] imageBytes = File.ReadAllBytes(opticalImagePath);
            string fileName = Path.GetFileName(opticalImagePath);
            form.AddBinaryData("file", imageBytes, fileName, "image/png");

            if (!string.IsNullOrEmpty(referenceDemPath) && File.Exists(referenceDemPath))
            {
                byte[] refBytes = File.ReadAllBytes(referenceDemPath);
                string refName = Path.GetFileName(referenceDemPath);
                form.AddBinaryData("ref_dem", refBytes, refName, "image/tiff");
            }

            form.AddField("min_alt", fallbackMinElevation.ToString());
            form.AddField("max_alt", fallbackMaxElevation.ToString());
            form.AddField("tile_size", tileSize.ToString());
            form.AddField("overlap_ratio", overlapRatio.ToString());

            using (UnityWebRequest req = UnityWebRequest.Post(serverUrl, form))
            {
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    statusMessage = $"FastAPI Request Failed: {req.error} | {req.downloadHandler.text}";
                    Debug.LogError($"[AppManager] {statusMessage}");
                    isProcessing = false;
                    yield break;
                }

                string jsonResponse = req.downloadHandler.text;
                Debug.Log($"[AppManager] FastAPI Response:\n{jsonResponse}");

                ProcessResponse response = null;
                try
                {
                    response = JsonUtility.FromJson<ProcessResponse>(jsonResponse);
                }
                catch (Exception ex)
                {
                    statusMessage = $"JSON Deserialization Error: {ex.Message}";
                    Debug.LogError($"[AppManager] {statusMessage}");
                    isProcessing = false;
                    yield break;
                }

                if (response == null || response.status != "success")
                {
                    statusMessage = "FastAPI returned error status.";
                    Debug.LogError($"[AppManager] {statusMessage}");
                    isProcessing = false;
                    yield break;
                }

                currentTaskId = response.task_id;
                statusMessage = $"Processing Task {currentTaskId} complete. Downloading assets...";

                // Download optical texture PNG & 16-bit heightmap PNG
                string opticalUrl = baseHostUrl + response.download_urls.optical_texture_png;
                string heightmapUrl = baseHostUrl + response.download_urls.heightmap_16bit_png;

                Texture2D opticalTexture = null;
                Texture2D heightmapTexture = null;

                yield return StartCoroutine(CoDownloadTexture(opticalUrl, tex => opticalTexture = tex));
                yield return StartCoroutine(CoDownloadTexture(heightmapUrl, tex => heightmapTexture = tex));

                if (opticalTexture != null && heightmapTexture != null)
                {
                    float minElev = response.metadata.elevation_metrics.min_elevation_meters;
                    float maxElev = response.metadata.elevation_metrics.max_elevation_meters;
                    float heightRange = response.metadata.elevation_metrics.elevation_range_meters;

                    // Update Material properties
                    if (targetDisplacementMaterial != null)
                    {
                        targetDisplacementMaterial.SetTexture("_MainTex", opticalTexture);
                        targetDisplacementMaterial.SetTexture("_HeightMap", heightmapTexture);
                        targetDisplacementMaterial.SetFloat("_HeightScale", heightRange);
                        targetDisplacementMaterial.SetFloat("_MinElevation", minElev);
                    }

                    // Rebuild procedural mesh and bake heights for physics raycasting
                    if (meshGenerator != null)
                    {
                        meshGenerator.GenerateGridMesh(
                            meshGenerator.resolutionX,
                            meshGenerator.resolutionZ,
                            meshGenerator.meshSizeX,
                            meshGenerator.meshSizeZ
                        );

                        meshGenerator.ApplyHeightmapToMesh(heightmapTexture, heightRange, minElev);
                    }

                    statusMessage = $"Successfully rendered 3D DSM for Task {currentTaskId}!";
                    Debug.Log($"[AppManager] {statusMessage}");
                }
                else
                {
                    statusMessage = "Failed to download generated heightmap or optical textures.";
                    Debug.LogError($"[AppManager] {statusMessage}");
                }

                isProcessing = false;
            }
        }

        private IEnumerator CoDownloadTexture(string url, Action<Texture2D> callback)
        {
            using (UnityWebRequest uwr = UnityWebRequestTexture.GetTexture(url))
            {
                yield return uwr.SendWebRequest();

                if (uwr.result == UnityWebRequest.Result.Success)
                {
                    Texture2D tex = DownloadHandlerTexture.GetContent(uwr);
                    callback?.Invoke(tex);
                }
                else
                {
                    Debug.LogError($"[AppManager] Failed to download texture from {url}: {uwr.error}");
                    callback?.Invoke(null);
                }
            }
        }
    }
}
