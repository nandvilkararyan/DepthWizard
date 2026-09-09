/**
 * DepthWizardBridge.jslib
 *
 * Unity WebGL JavaScript Library — bridges C# calls to the parent browser page
 * via window.parent.postMessage(). Compiled and linked by Unity automatically
 * when placed in Assets/Plugins/WebGL/.
 *
 * Messages emitted to parent:
 *   { type: "dw:unityReady" }           — Unity scene has finished loading
 *   { type: "dw:terrainLoaded" }        — Terrain mesh built and textured
 *   { type: "dw:error", message: "…" }  — An error occurred inside Unity
 */
mergeInto(LibraryManager.library, {

    /**
     * Called from WebGLBridge.cs: NotifyParentReady()
     * Signals the parent window that the Unity WebGL player is initialised and
     * ready to receive terrain data.
     */
    DW_NotifyReady: function () {
        try {
            window.parent.postMessage({ type: "dw:unityReady" }, "*");
        } catch (e) {
            console.warn("[DepthWizardBridge] DW_NotifyReady failed:", e);
        }
    },

    /**
     * Called from WebGLBridge.cs: NotifyTerrainLoaded()
     * Signals the parent window that terrain has been fully rendered.
     */
    DW_NotifyTerrainLoaded: function () {
        try {
            window.parent.postMessage({ type: "dw:terrainLoaded" }, "*");
        } catch (e) {
            console.warn("[DepthWizardBridge] DW_NotifyTerrainLoaded failed:", e);
        }
    },

    /**
     * Called from WebGLBridge.cs: NotifyError(message)
     * Forwards an error string to the parent page so it can display the
     * Three.js fallback toast if needed.
     *
     * @param {number} messagePtr  Emscripten heap pointer to a C-string.
     */
    DW_NotifyError: function (messagePtr) {
        try {
            var message = UTF8ToString(messagePtr);
            window.parent.postMessage({ type: "dw:error", message: message }, "*");
        } catch (e) {
            console.warn("[DepthWizardBridge] DW_NotifyError failed:", e);
        }
    },

    /**
     * Called from WebGLBridge.cs: RegisterMessageListener()
     * Attaches a window.addEventListener("message") handler that forwards
     * JSON payload from the parent to the Unity C# side via
     * SendMessage("WebGLBridge", "ReceivePayload", jsonString).
     *
     * This is called once during Awake/Start so Unity can receive the
     * FastAPI result as soon as the parent posts it.
     */
    DW_RegisterMessageListener: function () {
        if (window._dwListenerRegistered) return;
        window._dwListenerRegistered = true;

        window.addEventListener("message", function (event) {
            try {
                var data = event.data;
                if (!data || data.type !== "dw:loadTerrain") return;

                // Forward JSON string to C# WebGLBridge.ReceivePayload()
                var jsonStr = JSON.stringify(data.payload);
                // SendMessage is injected by the Unity WebGL loader
                if (typeof SendMessage === "function") {
                    SendMessage("WebGLBridge", "ReceivePayload", jsonStr);
                } else {
                    console.warn("[DepthWizardBridge] SendMessage not available yet.");
                }
            } catch (e) {
                console.warn("[DepthWizardBridge] message handler error:", e);
            }
        });
    }
});
