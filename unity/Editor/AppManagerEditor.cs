#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using DepthWizard.GIS;

namespace DepthWizard.Editor
{
    [CustomEditor(typeof(AppManager))]
    public class AppManagerEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            AppManager appMgr = (AppManager)target;

            EditorGUILayout.Space(15);
            GUI.backgroundColor = new Color(0.2f, 0.7f, 1.0f);

            if (GUILayout.Button("Process 2D Image & Render 3D Terrain", GUILayout.Height(38)))
            {
                if (!Application.isPlaying)
                {
                    EditorUtility.DisplayDialog(
                        "Enter Play Mode Required",
                        "Please enter Unity Play Mode first, then click this button to process the image asynchronously.",
                        "OK"
                    );
                    return;
                }

                if (string.IsNullOrEmpty(appMgr.opticalImagePath))
                {
                    string selectedPath = EditorUtility.OpenFilePanel("Select 2D Optical Image", "", "png,jpg,jpeg,tif,tiff");
                    if (!string.IsNullOrEmpty(selectedPath))
                    {
                        appMgr.opticalImagePath = selectedPath;
                    }
                    else
                    {
                        return;
                    }
                }

                appMgr.ProcessImagePipeline(appMgr.opticalImagePath, appMgr.referenceDemPath);
            }

            GUI.backgroundColor = Color.white;

            if (appMgr.isProcessing)
            {
                EditorGUILayout.HelpBox($"Status: {appMgr.statusMessage}", MessageType.Info);
            }
        }
    }
}
#endif
