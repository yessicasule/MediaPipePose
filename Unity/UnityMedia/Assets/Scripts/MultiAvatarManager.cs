// MultiAvatarManager.cs — Phase 8: Unity Digital Twin controller.
//
// Drives 4 avatar GameObjects from a single UdpAngleReceiver using
// the corresponding data source (MediaPipe, MoveNet, Fusion, GAN).
//
// Setup (automatic via SceneBuilder or manual):
//   1. Attach to any scene root GameObject.
//   2. Assign the UdpAngleReceiver in the 'receiver' field.
//   3. Drag the 4 ArmAngleController components into the avatar slots.

using UnityEngine;
using UnityEngine.UI;

namespace PoseTrackReceiver
{
    public class MultiAvatarManager : MonoBehaviour
    {
        [Header("Shared Receiver (single UDP port)")]
        public UdpAngleReceiver receiver;

        [Header("Avatar Controllers")]
        public ArmAngleController avatarMediaPipe;
        public ArmAngleController avatarMoveNet;
        public ArmAngleController avatarFusion;
        public ArmAngleController avatarGAN;

        [Header("Labels (optional Legacy UI.Text)")]
        public Text labelMediaPipe;
        public Text labelMoveNet;
        public Text labelFusion;
        public Text labelGAN;

        [Header("Uncertainty badge on Fusion avatar (optional)")]
        public Text uncertaintyText;

        void Start()
        {
            if (receiver == null)
            {
                Debug.LogError("[MultiAvatarManager] UdpAngleReceiver not assigned!");
                return;
            }

            if (labelMediaPipe) labelMediaPipe.text = "MediaPipe (Raw)";
            if (labelMoveNet)   labelMoveNet.text   = "MoveNet (Raw)";
            if (labelFusion)    labelFusion.text    = "DeepFusionPose";
            if (labelGAN)       labelGAN.text       = "GAN Refined";
        }

        void Update()
        {
            if (receiver == null || !receiver.HasData) return;

            // ── MediaPipe avatar ─────────────────────────────────────────────
            if (avatarMediaPipe != null)
                avatarMediaPipe.ApplyAngles(
                    receiver.Latest_MP.shoulderPitch,
                    receiver.Latest_MP.shoulderRoll,
                    receiver.Latest_MP.shoulderYaw,
                    receiver.Latest_MP.elbowFlex);

            // ── MoveNet avatar ───────────────────────────────────────────────
            if (avatarMoveNet != null)
                avatarMoveNet.ApplyAngles(
                    receiver.Latest_MV.shoulderPitch,
                    receiver.Latest_MV.shoulderRoll,
                    receiver.Latest_MV.shoulderYaw,
                    receiver.Latest_MV.elbowFlex);

            // ── Fusion avatar (with uncertainty badge) ───────────────────────
            if (avatarFusion != null)
            {
                avatarFusion.ApplyAngles(
                    receiver.Latest_FU.shoulderPitch,
                    receiver.Latest_FU.shoulderRoll,
                    receiver.Latest_FU.shoulderYaw,
                    receiver.Latest_FU.elbowFlex);

                if (uncertaintyText != null)
                {
                    float unc  = receiver.Latest_FU.uncertainty;
                    float conf = 1f / (1f + unc);
                    uncertaintyText.text  = $"Conf: {conf:P0}  ±{unc:F2}°";
                    uncertaintyText.color = conf > 0.85f ? Color.green :
                                            conf > 0.60f ? Color.yellow : Color.red;
                }
            }

            // ── GAN avatar ───────────────────────────────────────────────────
            if (avatarGAN != null)
                avatarGAN.ApplyAngles(
                    receiver.Latest_GR.shoulderPitch,
                    receiver.Latest_GR.shoulderRoll,
                    receiver.Latest_GR.shoulderYaw,
                    receiver.Latest_GR.elbowFlex);
        }
    }
}
