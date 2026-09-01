// MonoArmManager.cs
// ===================
// Main scene manager. Wires UdpAngleReceiver → arm controller(s) and displays
// live status in the Debug Console and on a world-space HUD.
//
// Two controller implementations are supported and either may be used:
//   AvatarMuscleController — writes through Unity's humanoid muscle system
//   ArmBoneController      — writes bone Transform.localRotation directly
// Whichever is present in the scene is discovered automatically. Do not put
// both on the same avatar; they would write to the same bones each frame.
//
// Bilateral support: assign both avatarController (side = Right) and
// leftArmController (side = Left) to drive both arms from a 'B,' packet.
// leftArmController is optional — leaving it unassigned (and the scene
// without a second AvatarMuscleController) reproduces the original
// single-arm ('S,' packet) behavior exactly.
//
// Setup
// -----
// 1. Create an empty GameObject named "MonoArmManager".
// 2. Attach this component.
// 3. Assign the UdpAngleReceiver and AvatarMuscleController reference(s)
//    (or run MonoArm → Build Scene from the menu to do it automatically).

using UnityEngine;

namespace MonoArm
{
    public class MonoArmManager : MonoBehaviour
    {
        [Header("References")]
        public UdpAngleReceiver    receiver;

        [Header("Muscle-space controllers (optional)")]
        public AvatarMuscleController avatarController;      // side = Right (or the sole arm in single-arm scenes)
        public AvatarMuscleController leftArmController;     // side = Left; optional, bilateral only

        [Header("Direct bone-rotation controllers (optional)")]
        [Tooltip("Transform.localRotation / Quaternion.Euler controllers, used " +
                 "instead of the muscle controllers when present.")]
        public ArmBoneController boneControllerRight;
        public ArmBoneController boneControllerLeft;

        [Header("Status")]
        [Tooltip("Log angle values to the console every N seconds (0 = disabled).")]
        public float logIntervalSeconds = 0f;

        float _logTimer;

        void Start()
        {
            if (receiver == null)
                receiver = FindFirstObjectByType<UdpAngleReceiver>();

            if (avatarController == null || leftArmController == null)
            {
                foreach (var c in FindObjectsByType<AvatarMuscleController>(FindObjectsSortMode.None))
                {
                    if (c.side == ArmSide.Right && avatarController == null) avatarController = c;
                    else if (c.side == ArmSide.Left && leftArmController == null) leftArmController = c;
                }
            }

            if (boneControllerRight == null || boneControllerLeft == null)
            {
                foreach (var c in FindObjectsByType<ArmBoneController>(FindObjectsSortMode.None))
                {
                    if (c.side == ArmSide.Right && boneControllerRight == null) boneControllerRight = c;
                    else if (c.side == ArmSide.Left && boneControllerLeft == null) boneControllerLeft = c;
                }
            }

            if (receiver == null)
                Debug.LogError("[MonoArmManager] UdpAngleReceiver not found in scene.");

            if (avatarController == null && boneControllerRight == null)
                Debug.LogError("[MonoArmManager] No right-arm controller found. Add either an " +
                               "AvatarMuscleController or an ArmBoneController to the avatar.");

            if (leftArmController == null && boneControllerLeft == null)
                Debug.Log("[MonoArmManager] No left-arm controller assigned — left-arm data from " +
                          "bilateral packets will be received but not applied to any avatar.");
        }

        void Update()
        {
            if (receiver == null || !receiver.HasData) return;

            // Forward right-arm (or the sole arm, for 'S,' packets) angles.
            if (avatarController   != null) avatarController.ApplyAngles(receiver.LatestAngles);
            if (boneControllerRight != null) boneControllerRight.ApplyAngles(receiver.LatestAngles);

            // Forward left-arm angles when the latest packet was bilateral.
            if (receiver.IsBilateral)
            {
                var left = receiver.LatestBilateralAngles.left;
                if (leftArmController   != null) leftArmController.ApplyAngles(left);
                if (boneControllerLeft  != null) boneControllerLeft.ApplyAngles(left);
            }

            if (logIntervalSeconds > 0f)
            {
                _logTimer += Time.deltaTime;
                if (_logTimer >= logIntervalSeconds)
                {
                    _logTimer = 0f;
                    Debug.Log($"[MonoArmManager] {receiver.PacketCount} packets | " +
                              $"{receiver.LatestAngles}");
                }
            }
        }
    }
}
