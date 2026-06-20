using UnityEngine;

namespace PoseTrackReceiver
{
    public class ArmAngleController : MonoBehaviour
    {
        [Header("Avatar Bones")]
        public Transform upperArmBone;
        public Transform lowerArmBone;

        [Header("Smoothing (0 = none, 1 = max)")]
        [Range(0.01f, 1f)]
        public float smoothing = 0.15f;

        [Header("Axis Mapping (degrees added to received angle)")]
        public Vector3 shoulderAxisOffset = Vector3.zero;
        public Vector3 elbowAxisOffset = Vector3.zero;

        AngleSmoother _smPitch;
        AngleSmoother _smYaw;
        AngleSmoother _smRoll;
        AngleSmoother _smElbow;

        Vector3 _shoulderEuler;
        Vector3 _elbowEuler;

        void Awake()
        {
            _smPitch = new AngleSmoother(smoothing);
            _smYaw   = new AngleSmoother(smoothing);
            _smRoll  = new AngleSmoother(smoothing);
            _smElbow = new AngleSmoother(smoothing);
        }

        /// <summary>
        /// Called by MultiAvatarManager every frame.
        /// </summary>
        public void ApplyAngles(float pitch, float roll, float yaw, float elbow)
        {
            float sPitch = _smPitch.Update(pitch);
            float sRoll  = _smRoll.Update(roll);
            float sYaw   = _smYaw.Update(yaw);
            float sElbow = _smElbow.Update(elbow);

            if (upperArmBone != null)
            {
                _shoulderEuler.x = sPitch + shoulderAxisOffset.x;
                _shoulderEuler.y = sYaw   + shoulderAxisOffset.y;
                _shoulderEuler.z = sRoll  + shoulderAxisOffset.z;

                upperArmBone.localRotation =
                    Quaternion.Euler(_shoulderEuler);
            }

            if (lowerArmBone != null)
            {
                _elbowEuler.x = sElbow + elbowAxisOffset.x;
                _elbowEuler.y = elbowAxisOffset.y;
                _elbowEuler.z = elbowAxisOffset.z;

                lowerArmBone.localRotation =
                    Quaternion.Euler(_elbowEuler);
            }
        }
    }
}