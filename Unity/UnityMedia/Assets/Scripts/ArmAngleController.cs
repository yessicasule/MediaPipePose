// ArmAngleController — applies received joint angles to the avatar bones.
//
// Update() budget: one lock-free struct copy (done in UdpAngleReceiver.Update),
// four Mathf.LerpAngle calls, two Quaternion.Euler + localRotation assignments.
// No allocations, no string ops, no Debug.Log.  Keep it that way.
// All diagnostics go through PoseDebugUI, never through the console.

using UnityEngine;

namespace PoseTrackReceiver
{
    [RequireComponent(typeof(UdpAngleReceiver))]
    public class ArmAngleController : MonoBehaviour
    {
        [Header("Avatar Bones")]
        public Transform upperArmBone;
        public Transform lowerArmBone;

        [Header("Smoothing  (0 = none, 1 = max)")]
        [Range(0.01f, 1f)] public float smoothing = 0.15f;

        [Header("Axis Mapping  (degrees added to received angle)")]
        public Vector3 shoulderAxisOffset = Vector3.zero;
        public Vector3 elbowAxisOffset    = Vector3.zero;

        UdpAngleReceiver _receiver;
        AngleSmoother    _smPitch, _smYaw, _smRoll, _smElbow;

        // Cached Euler vectors — reused every frame to avoid struct churn
        Vector3 _shoulderEuler;
        Vector3 _elbowEuler;

        void Awake()
        {
            _receiver = GetComponent<UdpAngleReceiver>();
            _smPitch  = new AngleSmoother(smoothing);
            _smYaw    = new AngleSmoother(smoothing);
            _smRoll   = new AngleSmoother(smoothing);
            _smElbow  = new AngleSmoother(smoothing);
        }

        void Update()
        {
            // Driven by UdpAngleReceiver when used standalone
            if (_receiver != null && _receiver.HasData)
            {
                var a = _receiver.Latest;
                ApplyAngles(a.shoulderPitch, a.shoulderRoll, a.shoulderYaw, a.elbowFlex);
            }
        }

        /// <summary>
        /// Public entry-point used by MultiAvatarManager to drive this avatar directly.
        /// Applies smoothing and axis offsets before writing to bones.
        /// </summary>
        public void ApplyAngles(float pitch, float roll, float yaw, float elbow)
        {
            float sPitch = _smPitch.Update(pitch);
            float sRoll  = _smRoll .Update(roll);
            float sYaw   = _smYaw  .Update(yaw);
            float sElbow = _smElbow.Update(elbow);

            if (upperArmBone != null)
            {
                _shoulderEuler.x = sPitch + shoulderAxisOffset.x;
                _shoulderEuler.y = sYaw   + shoulderAxisOffset.y;
                _shoulderEuler.z = sRoll  + shoulderAxisOffset.z;
                upperArmBone.localRotation = Quaternion.Euler(_shoulderEuler);
            }

            if (lowerArmBone != null)
            {
                _elbowEuler.x = sElbow + elbowAxisOffset.x;
                _elbowEuler.y =          elbowAxisOffset.y;
                _elbowEuler.z =          elbowAxisOffset.z;
                lowerArmBone.localRotation = Quaternion.Euler(_elbowEuler);
            }
        }
    }
}
