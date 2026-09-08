using System;
using UnityEngine;
using UnityEngine.UI;

namespace DepthWizard.GIS
{
    /// <summary>
    /// Real-time Metric Elevation and Slope Inspector.
    /// Performs physics raycasting from mouse cursor into terrain mesh collider,
    /// calculating absolute metric elevation, surface slope in degrees, and coordinates.
    /// </summary>
    public class TerrainInspector : MonoBehaviour
    {
        [Header("Target Components")]
        [Tooltip("Main GIS Fly camera reference.")]
        public Camera targetCamera;

        [Tooltip("Reference to TerrainMeshGenerator.")]
        public TerrainMeshGenerator terrainMeshGenerator;

        [Header("UI Readout References (Optional)")]
        public Text heightText;
        public Text slopeText;
        public Text coordinatesText;

        [Header("Inspector Settings")]
        public LayerMask terrainLayer = ~0;
        public bool showOnGUIOverlay = true;

        private bool isHittingTerrain = false;
        private Vector3 hitWorldPosition;
        private Vector3 hitNormal;
        private Vector2 hitUV;
        private float currentElevationMeters = 0f;
        private float currentSlopeDegrees = 0f;

        private void Start()
        {
            if (targetCamera == null)
            {
                targetCamera = Camera.main;
            }
        }

        private void Update()
        {
            PerformRaycast();
            UpdateUI();
        }

        private void PerformRaycast()
        {
            if (targetCamera == null) return;

            Ray ray = targetCamera.ScreenPointToRay(Input.mousePosition);

            if (Physics.Raycast(ray, out RaycastHit hit, 5000f, terrainLayer))
            {
                isHittingTerrain = true;
                hitWorldPosition = hit.point;
                hitNormal = hit.normal;
                hitUV = hit.textureCoord;

                // 1. Calculate surface slope in degrees using surface normal vector
                currentSlopeDegrees = Vector3.Angle(hitNormal, Vector3.up);

                // 2. Calculate absolute metric elevation in meters
                if (terrainMeshGenerator != null)
                {
                    // Sample precise bilinear elevation from heightmap metadata
                    currentElevationMeters = terrainMeshGenerator.GetHeightAtUV(hitUV);
                }
                else
                {
                    // Fallback to Y position of hit point
                    currentElevationMeters = hitWorldPosition.y;
                }
            }
            else
            {
                isHittingTerrain = false;
            }
        }

        private void UpdateUI()
        {
            if (!isHittingTerrain)
            {
                if (heightText != null) heightText.text = "Elevation: --";
                if (slopeText != null) slopeText.text = "Slope: --";
                if (coordinatesText != null) coordinatesText.text = "Coordinates: --";
                return;
            }

            string heightStr = $"Elevation: {currentElevationMeters:F2} m";
            string slopeStr = $"Slope: {currentSlopeDegrees:F1}°";
            string coordStr = $"World: ({hitWorldPosition.x:F1}, {hitWorldPosition.y:F1}, {hitWorldPosition.z:F1}) | UV: ({hitUV.x:F3}, {hitUV.y:F3})";

            if (heightText != null) heightText.text = heightStr;
            if (slopeText != null) slopeText.text = slopeStr;
            if (coordinatesText != null) coordinatesText.text = coordStr;
        }

        private void OnGUI()
        {
            if (!showOnGUIOverlay || !isHittingTerrain) return;

            // Render clean IMGUI diagnostic panel in top-left corner if UI Text is unassigned
            float width = 380f;
            float height = 110f;
            Rect panelRect = new Rect(20f, 20f, width, height);

            GUI.Box(panelRect, "ISRO DepthWizard - GIS Inspector");

            GUILayout.BeginArea(new Rect(30f, 45f, width - 20f, height - 30f));
            GUILayout.Label($"<b>Elevation:</b> {currentElevationMeters:F2} meters");
            GUILayout.Label($"<b>Slope Angle:</b> {currentSlopeDegrees:F1}° (Degrees)");
            GUILayout.Label($"<b>Coordinates:</b> ({hitWorldPosition.x:F1}, {hitWorldPosition.z:F1}) | UV: ({hitUV.x:F3}, {hitUV.y:F3})");
            GUILayout.EndArea();
        }
    }
}
