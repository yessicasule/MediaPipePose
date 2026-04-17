using UnityEngine;

namespace PoseTrackReceiver
{
    [RequireComponent(typeof(UdpAngleReceiver))]
    public class ArmAngleController : MonoBehaviour
    {
        [Header("Avatar Bones")]
        public Transform upperArmBone;
        public Transform lowerArmBone;

        [Header("Smoothing  (0=none, 1=max)")]
        [Range(0.01f, 1f)] public float smoothing = 0.15f;

        [Header("Axis Mapping  (degrees added to received angle)")]
        public Vector3 shoulderAxisOffset = Vector3.zero;
        public Vector3 elbowAxisOffset    = Vector3.zero;

        UdpAngleReceiver _receiver;
        AngleSmoother    _smPitch;
        AngleSmoother    _smYaw;
        AngleSmoother    _smRoll;
        AngleSmoother    _smElbow;

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
            if (!_receiver.HasData) return;
            var a = _receiver.Latest;

            float pitch = _smPitch.Update(a.shoulderPitch);
            float yaw   = _smYaw.Update(a.shoulderYaw);
            float roll  = _smRoll.Update(a.shoulderRoll);
            float elbow = _smElbow.Update(a.elbowFlex);

            if (upperArmBone != null)
            {
                Vector3 euler = new Vector3(pitch, yaw, roll) + shoulderAxisOffset;
                upperArmBone.localRotation = Quaternion.Euler(euler);
            }

            if (lowerArmBone != null)
            {
                Vector3 euler = new Vector3(elbow, 0f, 0f) + elbowAxisOffset;
                lowerArmBone.localRotation = Quaternion.Euler(euler);
            }
        }
    }
}
