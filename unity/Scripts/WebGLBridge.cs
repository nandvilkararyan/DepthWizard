using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace DepthWizard.GIS
{
    /// <summary>
    /// WebGLBridge — postMessage ↔ Unity C# communication layer.
    ///
    /// Attach this MonoBehaviour to a GameObject named exactly "WebGLBridge"
    /// in the Unity scene. The .jslib plugin forwards window.postMessage events
    /// from the parent page into ReceivePayload() via Unity's SendMessage().
    ///
    /// Flow:
    ///   1. DW_RegisterMessageListener() wires up the browser event listener.
    ///   2. DW_NotifyReady() tells the parent page Unity is ready to receive data.
    ///   3. Parent posts { type:"dw:loadTerrain", payload: { FastAPI JSON } }.
    ///   4. ReceivePayload() deserialises and calls AppManager.LoadFromUrls().
    ///   5. DW_NotifyTerrainLoaded() fires when the terrain is on-screen.
    /// </summary>
    public class WebGLBridge : MonoBehaviour
    {
        // ── Inspector wiring ──────────────────────────────────────────────────
        [Header("Scene References")]
        [Tooltip("Reference to the AppManager in the scene.")]
        public AppManager appManager;

        [Tooltip("Base host URL used to resolve relative paths from the FastAPI response.")]
        public string baseHostUrl = "http://127.0.0.1:8000";

        // ── jslib imports ─────────────────────────────────────────────────────
#if !UNITY_EDITOR && UNITY_WEBGL
        [DllImport("__Internal")]
        private static extern void DW_RegisterMessageListener();

        [DllImport("__Internal")]
        private static extern void DW_NotifyReady();

        [DllImport("__Internal")]
        private static extern void DW_NotifyTerrainLoaded();

        [DllImport("__Internal")]
        private static extern void DW_NotifyError(string message);
#else
        // Editor / standalone stubs — keep compilation clean
        private static void DW_RegisterMessageListener() { }
        private static void DW_NotifyReady() { }
        private static void DW_NotifyTerrainLoaded() { }
        private static void DW_NotifyError(string message) =>
            Debug.LogWarning($"[WebGLBridge] Error (stub): {message}");
#endif

        // ── Lifecycle ─────────────────────────────────────────────────────────
        private void Awake()
        {
            // Ensure this GameObject name matches the SendMessage target in the jslib
            if (gameObject.name != "WebGLBridge")
            {
                Debug.LogWarning("[WebGLBridge] GameObject must be named 'WebGLBridge' for SendMessage to work.");
            }

            if (appManager == null)
                appManager = FindObjectOfType<AppManager>();
        }

        private void Start()
        {
            // Wire up the browser-side message listener first, then announce ready
            DW_RegisterMessageListener();
            DW_NotifyReady();
            Debug.Log("[WebGLBridge] Registered postMessage listener and notified parent: dw:unityReady");
        }

        // ── Public entry point called by jslib via SendMessage ────────────────

        /// <summary>
        /// Called by the .jslib plugin when the parent page posts a
        /// { type:"dw:loadTerrain", payload: {...} } message.
        /// </summary>
        /// <param name="jsonPayload">Serialised FastAPI /process response JSON.</param>
        public void ReceivePayload(string jsonPayload)
        {
            if (string.IsNullOrEmpty(jsonPayload))
            {
                string err = "ReceivePayload called with empty JSON.";
                Debug.LogError($"[WebGLBridge] {err}");
                DW_NotifyError(err);
                return;
            }

            Debug.Log($"[WebGLBridge] Received terrain payload ({jsonPayload.Length} bytes).");

            WebGLTerrainPayload payload = null;
            try
            {
                payload = JsonUtility.FromJson<WebGLTerrainPayload>(jsonPayload);
            }
            catch (Exception ex)
            {
                string err = $"JSON parse error: {ex.Message}";
                Debug.LogError($"[WebGLBridge] {err}");
                DW_NotifyError(err);
                return;
            }

            if (payload == null || payload.download_urls == null)
            {
                string err = "Parsed payload is null or missing download_urls.";
                Debug.LogError($"[WebGLBridge] {err}");
                DW_NotifyError(err);
                return;
            }

            // Resolve relative URLs against the base host
            string heightmapUrl = ResolveUrl(payload.download_urls.heightmap_16bit_png);
            string textureUrl   = ResolveUrl(payload.download_urls.optical_texture_png);

            if (appManager == null)
            {
                string err = "AppManager reference is missing.";
                Debug.LogError($"[WebGLBridge] {err}");
                DW_NotifyError(err);
                return;
            }

            // Kick off download + terrain generation via AppManager
            appManager.LoadFromUrls(
                heightmapUrl,
                textureUrl,
                payload.metadata,
                onComplete: () => DW_NotifyTerrainLoaded(),
                onError:    (msg) => DW_NotifyError(msg)
            );
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        private string ResolveUrl(string path)
        {
            if (string.IsNullOrEmpty(path)) return "";
            if (path.StartsWith("http://") || path.StartsWith("https://")) return path;
            return baseHostUrl.TrimEnd('/') + "/" + path.TrimStart('/');
        }
    }

    // ── JSON-serialisable payload matching the frontend postMessage structure ──

    [Serializable]
    public class WebGLTerrainPayload
    {
        public string task_id;
        public string status;
        public SceneMetadata metadata;
        public WebGLDownloadUrls download_urls;
    }

    [Serializable]
    public class WebGLDownloadUrls
    {
        public string heightmap_16bit_png;
        public string heightmap_8bit_preview_png;
        public string optical_texture_png;
        public string depth_colorized_png;
        public string geotiff_dsm_32bit;
        public string calibration_metadata_json;
    }
}
