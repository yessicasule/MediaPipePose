// MonoArmManager.cs
// ===================
// Main scene manager. Wires UdpAngleReceiver → AvatarMuscleController
// and displays live status in the Debug Console and on a world-space HUD.
//
// Setup
// -----
// 1. Create an empty GameObject named "MonoArmManager".
// 2. Attach this component.
// 3. Assign the UdpAngleReceiver and AvatarMuscleController references
//    (or run MonoArm → Build Scene from the menu to do it automatically).

using UnityEngine;

namespace MonoArm
{
    public class MonoArmManager : MonoBehaviour
    {
        [Header("References")]
        public UdpAngleReceiver    receiver;
        public AvatarMuscleController avatarController;

        [Header("Status")]
        [Tooltip("Log angle values to the console every N seconds (0 = disabled).")]
        public float logIntervalSeconds = 0f;

        float _logTimer;

        void Start()
        {
            if (receiver == null)
                receiver = FindFirstObjectByType<UdpAngleReceiver>();

            if (avatarController == null)
                avatarController = FindFirstObjectByType<AvatarMuscleController>();

            if (receiver == null)
                Debug.LogError("[MonoArmManager] UdpAngleReceiver not found in scene.");

            if (avatarController == null)
                Debug.LogError("[MonoArmManager] AvatarMuscleController not found in scene.");
        }

        void Update()
        {
            if (receiver == null || avatarController == null) return;
            if (!receiver.HasData) return;

            // Forward angles from receiver to avatar controller
            avatarController.ApplyAngles(receiver.LatestAngles);
        }
    }
}
