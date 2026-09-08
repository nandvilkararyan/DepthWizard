using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace DepthWizard.GIS
{
    /// <summary>
    /// Generates a dense subdivided grid plane mesh programmatically at runtime
    /// and bakes heightmap displacements for URP rendering and Physics Raycasting.
    /// </summary>
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer), typeof(MeshCollider))]
    public class TerrainMeshGenerator : MonoBehaviour
    {
        [Header("Grid Dimensions & Resolution")]
        [Tooltip("Number of grid subdivisions along X-axis (e.g., 256 or 512).")]
        [Range(16, 512)]
        public int resolutionX = 256;

        [Tooltip("Number of grid subdivisions along Z-axis (e.g., 256 or 512).")]
        [Range(16, 512)]
        public int resolutionZ = 256;

        [Tooltip("Physical mesh width in Unity units along X-axis.")]
        public float meshSizeX = 100f;

        [Tooltip("Physical mesh length in Unity units along Z-axis.")]
        public float meshSizeZ = 100f;

        [Header("Heightmap CPU Baking")]
        [Tooltip("Scale factor (in meters) for vertical Y-axis displacement.")]
        public float currentHeightScale = 50f;

        [Tooltip("Minimum elevation offset (in meters).")]
        public float currentMinElevation = 0f;

        private MeshFilter meshFilter;
        private MeshRenderer meshRenderer;
        private MeshCollider meshCollider;
        private Mesh proceduralMesh;

        private Vector3[] baseVertices;
        private Vector3[] displacedVertices;
        private Vector2[] uvs;
        private int[] triangles;
        private Texture2D cachedHeightmap;

        private void Awake()
        {
            meshFilter = GetComponent<MeshFilter>();
            meshRenderer = GetComponent<MeshRenderer>();
            meshCollider = GetComponent<MeshCollider>();
        }

        private void Start()
        {
            if (proceduralMesh == null)
            {
                GenerateGridMesh(resolutionX, resolutionZ, meshSizeX, meshSizeZ);
            }
        }

        /// <summary>
        /// Programmatically constructs a subdivided grid plane mesh using IndexFormat.UInt32.
        /// </summary>
        public void GenerateGridMesh(int resX, int resZ, float width, float length)
        {
            resolutionX = Mathf.Clamp(resX, 16, 512);
            resolutionZ = Mathf.Clamp(resZ, 16, 512);
            meshSizeX = width;
            meshSizeZ = length;

            if (meshFilter == null) meshFilter = GetComponent<MeshFilter>();
            if (meshCollider == null) meshCollider = GetComponent<MeshCollider>();

            proceduralMesh = new Mesh
            {
                name = "ProceduralTerrainGrid",
                // Enable 32-bit index buffer to support > 65,535 vertices
                indexFormat = IndexFormat.UInt32
            };

            int numVertices = (resolutionX + 1) * (resolutionZ + 1);
            int numTriangles = resolutionX * resolutionZ * 6;

            baseVertices = new Vector3[numVertices];
            displacedVertices = new Vector3[numVertices];
            uvs = new Vector2[numVertices];
            triangles = new int[numTriangles];

            float dx = meshSizeX / resolutionX;
            float dz = meshSizeZ / resolutionZ;

            int vertIdx = 0;
            for (int z = 0; z <= resolutionZ; z++)
            {
                float zPos = z * dz - (meshSizeZ * 0.5f);
                float v = (float)z / resolutionZ;

                for (int x = 0; x <= resolutionX; x++)
                {
                    float xPos = x * dx - (meshSizeX * 0.5f);
                    float u = (float)x / resolutionX;

                    Vector3 pos = new Vector3(xPos, 0f, zPos);
                    baseVertices[vertIdx] = pos;
                    displacedVertices[vertIdx] = pos;
                    uvs[vertIdx] = new Vector2(u, v);

                    vertIdx++;
                }
            }

            int triIdx = 0;
            for (int z = 0; z < resolutionZ; z++)
            {
                for (int x = 0; x < resolutionX; x++)
                {
                    int current = z * (resolutionX + 1) + x;
                    int next = current + resolutionX + 1;

                    // First triangle
                    triangles[triIdx++] = current;
                    triangles[triIdx++] = next;
                    triangles[triIdx++] = current + 1;

                    // Second triangle
                    triangles[triIdx++] = current + 1;
                    triangles[triIdx++] = next;
                    triangles[triIdx++] = next + 1;
                }
            }

            proceduralMesh.vertices = displacedVertices;
            proceduralMesh.uv = uvs;
            proceduralMesh.triangles = triangles;

            proceduralMesh.RecalculateNormals();
            proceduralMesh.RecalculateTangents();
            proceduralMesh.RecalculateBounds();

            meshFilter.sharedMesh = proceduralMesh;
            meshCollider.sharedMesh = proceduralMesh;
        }

        /// <summary>
        /// Bakes CPU heightmap displacements into vertex Y positions to align MeshCollider with URP displacement shader.
        /// </summary>
        public void ApplyHeightmapToMesh(Texture2D heightmap, float heightScale, float minElevation)
        {
            if (heightmap == null || proceduralMesh == null) return;

            cachedHeightmap = heightmap;
            currentHeightScale = heightScale;
            currentMinElevation = minElevation;

            Color[] pixels = heightmap.GetPixels();
            int texW = heightmap.width;
            int texH = heightmap.height;

            for (int i = 0; i < baseVertices.Length; i++)
            {
                Vector2 uv = uvs[i];
                int px = Mathf.Clamp(Mathf.FloorToInt(uv.x * (texW - 1)), 0, texW - 1);
                int py = Mathf.Clamp(Mathf.FloorToInt(uv.y * (texH - 1)), 0, texH - 1);

                float normalizedHeight = pixels[py * texW + px].r;
                float heightMeters = normalizedHeight * heightScale;

                displacedVertices[i] = new Vector3(
                    baseVertices[i].x,
                    heightMeters,
                    baseVertices[i].z
                );
            }

            proceduralMesh.vertices = displacedVertices;
            proceduralMesh.RecalculateNormals();
            proceduralMesh.RecalculateTangents();
            proceduralMesh.RecalculateBounds();

            meshFilter.sharedMesh = proceduralMesh;
            // Update MeshCollider sharedMesh to ensure accurate physics raycasting
            meshCollider.sharedMesh = null;
            meshCollider.sharedMesh = proceduralMesh;
        }

        /// <summary>
        /// Bilinearly interpolates heightmap elevation at given UV coordinates [0, 1].
        /// </summary>
        public float GetHeightAtUV(Vector2 uv)
        {
            if (cachedHeightmap == null) return currentMinElevation;

            float u = Mathf.Clamp01(uv.x);
            float v = Mathf.Clamp01(uv.y);

            Color sample = cachedHeightmap.GetPixelBilinear(u, v);
            return currentMinElevation + (sample.r * currentHeightScale);
        }
    }
}
