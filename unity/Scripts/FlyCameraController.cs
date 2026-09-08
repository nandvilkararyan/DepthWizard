using UnityEngine;

namespace DepthWizard.GIS
{
    /// <summary>
    /// Production-ready standalone GIS FlyCamera controller.
    /// Provides WASD horizontal movement, vertical flight (Q/E, Space/Shift),
    /// RMB rotation, mouse scroll speed tuning, and terrain clearance clamping.
    /// </summary>
    public class FlyCameraController : MonoBehaviour
    {
        [Header("Movement Settings")]
        [Tooltip("Current flight speed in units/sec.")]
        public float flySpeed = 50f;

        [Tooltip("Minimum flight speed limit.")]
        public float minFlySpeed = 5f;

        [Tooltip("Maximum flight speed limit.")]
        public float maxFlySpeed = 500f;

        [Tooltip("Speed multiplier step per mouse scroll notch.")]
        public float scrollSensitivity = 10f;

        [Header("Rotation Settings")]
        [Tooltip("Mouse look sensitivity for Pitch and Yaw.")]
        public float lookSensitivity = 2.5f;

        [Tooltip("Smooth damping factor for camera rotation.")]
        public float rotationSmoothing = 10f;

        [Header("Terrain Collision Clamping")]
        [Tooltip("Optional reference to TerrainMeshGenerator or Terrain MeshCollider for clearance clamping.")]
        public MeshCollider terrainCollider;

        [Tooltip("Minimum vertical clearance (meters) maintained above the terrain surface.")]
        public float minGroundClearance = 5f;

        private float pitch = 0f;
        private float yaw = 0f;
        private float targetPitch = 0f;
        private float targetYaw = 0f;

        private void Start()
        {
            Vector3 angles = transform.eulerAngles;
            pitch = angles.x;
            yaw = angles.y;
            targetPitch = pitch;
            targetYaw = yaw;
        }

        private void Update()
        {
            HandleRotation();
            HandleMovement();
            HandleSpeedAdjustment();
            ClampTerrainClearance();
        }

        private void HandleRotation()
        {
            // Rotate camera when Right Mouse Button (RMB) is held
            if (Input.GetMouseButton(1))
            {
                Cursor.lockState = CursorLockMode.Confined;

                float mouseX = Input.GetAxis("Mouse X") * lookSensitivity;
                float mouseY = Input.GetAxis("Mouse Y") * lookSensitivity;

                targetYaw += mouseX;
                targetPitch -= mouseY;
                targetPitch = Mathf.Clamp(targetPitch, -89f, 89f);
            }
            else
            {
                Cursor.lockState = CursorLockMode.None;
            }

            pitch = Mathf.Lerp(pitch, targetPitch, Time.deltaTime * rotationSmoothing);
            yaw = Mathf.Lerp(yaw, targetYaw, Time.deltaTime * rotationSmoothing);

            transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
        }

        private void HandleMovement()
        {
            Vector3 moveDirection = Vector3.zero;

            // WASD Horizontal movement relative to camera orientation
            if (Input.GetKey(KeyCode.W)) moveDirection += transform.forward;
            if (Input.GetKey(KeyCode.S)) moveDirection -= transform.forward;
            if (Input.GetKey(KeyCode.D)) moveDirection += transform.right;
            if (Input.GetKey(KeyCode.A)) moveDirection -= transform.right;

            // Q / E or Space / LeftShift vertical movement
            if (Input.GetKey(KeyCode.E) || Input.GetKey(KeyCode.Space)) moveDirection += Vector3.up;
            if (Input.GetKey(KeyCode.Q) || Input.GetKey(KeyCode.LeftShift)) moveDirection -= Vector3.up;

            if (moveDirection.sqrMagnitude > 0.01f)
            {
                transform.position += moveDirection.normalized * (flySpeed * Time.deltaTime);
            }
        }

        private void HandleSpeedAdjustment()
        {
            float scroll = Input.GetAxis("Mouse ScrollWheel");
            if (Mathf.Abs(scroll) > 0.01f)
            {
                flySpeed = Mathf.Clamp(flySpeed + (scroll * scrollSensitivity * 10f), minFlySpeed, maxFlySpeed);
            }
        }

        private void ClampTerrainClearance()
        {
            // Raycast vertically downwards to prevent flying underneath terrain mesh
            Ray downRay = new Ray(new Vector3(transform.position.x, transform.position.y + 1000f, transform.position.z), Vector3.down);

            if (Physics.Raycast(downRay, out RaycastHit hit, 2000f))
            {
                float terrainY = hit.point.y;
                float minimumY = terrainY + minGroundClearance;

                if (transform.position.y < minimumY)
                {
                    transform.position = new Vector3(transform.position.x, minimumY, transform.position.z);
                }
            }
        }
    }
}
