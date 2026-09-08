#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using DepthWizard.GIS;

namespace DepthWizard.Editor
{
    /// <summary>
    /// Unity Editor Utility for the ISRO DepthWizard Challenge.
    /// One-click setup script that automatically builds the 3D elevation scene hierarchy,
    /// instantiates components, creates the URP displacement material, and wires up references.
    /// </summary>
    public static class SetupDepthWizardScene
    {
        [MenuItem("DepthWizard/Setup 3D Elevation Scene", false, 10)]
        public static void BuildSceneHierarchy()
        {
            // 1. Create or locate Main Camera
            Camera mainCam = Camera.main;
            if (mainCam == null)
            {
                GameObject camObj = new GameObject("Main Camera");
                mainCam = camObj.AddComponent<Camera>();
                camObj.tag = "MainCamera";
                camObj.AddComponent<AudioListener>();
            }

            // Align camera for GIS view
            mainCam.transform.position = new Vector3(0f, 150f, -150f);
            mainCam.transform.rotation = Quaternion.Euler(45f, 0f, 0f);

            // Attach FlyCameraController to Main Camera
            FlyCameraController flyCam = mainCam.GetComponent<FlyCameraController>();
            if (flyCam == null)
            {
                flyCam = mainCam.gameObject.AddComponent<FlyCameraController>();
            }

            // 2. Create URP HeightDisplacement Material
            Shader displacementShader = Shader.Find("Custom/URP/HeightDisplacement");
            Material terrainMat = null;

            if (displacementShader != null)
            {
                string matPath = "Assets/HeightDisplacementMaterial.mat";
                terrainMat = AssetDatabase.LoadAssetAtPath<Material>(matPath);
                if (terrainMat == null)
                {
                    terrainMat = new Material(displacementShader)
                    {
                        name = "HeightDisplacementMaterial"
                    };
                    AssetDatabase.CreateAsset(terrainMat, matPath);
                    AssetDatabase.SaveAssets();
                }
            }
            else
            {
                Debug.LogWarning("[DepthWizard Editor] Shader 'Custom/URP/HeightDisplacement' not found in project. Please ensure URP is configured.");
            }

            // 3. Create TerrainVisualizer GameObject
            GameObject terrainObj = GameObject.Find("TerrainVisualizer");
            if (terrainObj == null)
            {
                terrainObj = new GameObject("TerrainVisualizer");
            }

            MeshFilter filter = terrainObj.GetComponent<MeshFilter>();
            if (filter == null) filter = terrainObj.AddComponent<MeshFilter>();

            MeshRenderer renderer = terrainObj.GetComponent<MeshRenderer>();
            if (renderer == null) renderer = terrainObj.AddComponent<MeshRenderer>();

            MeshCollider collider = terrainObj.GetComponent<MeshCollider>();
            if (collider == null) collider = terrainObj.AddComponent<MeshCollider>();

            TerrainMeshGenerator meshGen = terrainObj.GetComponent<TerrainMeshGenerator>();
            if (meshGen == null) meshGen = terrainObj.AddComponent<TerrainMeshGenerator>();

            if (terrainMat != null)
            {
                renderer.sharedMaterial = terrainMat;
            }

            flyCam.terrainCollider = collider;

            // 4. Create AppManager GameObject
            GameObject managerObj = GameObject.Find("AppManager");
            if (managerObj == null)
            {
                managerObj = new GameObject("AppManager");
            }

            AppManager appMgr = managerObj.GetComponent<AppManager>();
            if (appMgr == null) appMgr = managerObj.AddComponent<AppManager>();

            appMgr.serverUrl = "http://127.0.0.1:8000/process";
            appMgr.baseHostUrl = "http://127.0.0.1:8000";
            appMgr.meshGenerator = meshGen;
            appMgr.targetDisplacementMaterial = terrainMat;

            // 5. Create TerrainInspector & UI Overlay
            GameObject inspectorObj = GameObject.Find("TerrainInspector");
            if (inspectorObj == null)
            {
                inspectorObj = new GameObject("TerrainInspector");
            }

            TerrainInspector inspector = inspectorObj.GetComponent<TerrainInspector>();
            if (inspector == null) inspector = inspectorObj.AddComponent<TerrainInspector>();

            inspector.targetCamera = mainCam;
            inspector.terrainMeshGenerator = meshGen;
            inspector.showOnGUIOverlay = true;

            Selection.activeGameObject = managerObj;
            Undo.RegisterCreatedObjectUndo(terrainObj, "Setup DepthWizard Scene");

            Debug.Log("<color=green>[DepthWizard Editor] 3D Satellite Elevation Scene setup complete!</color> Select AppManager in the Hierarchy to specify an optical image and test processing.");
        }
    }
}
#endif
