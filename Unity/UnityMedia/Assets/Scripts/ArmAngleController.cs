// AvatarMuscleController.cs
// ==========================
// Controls the humanoid avatar's RIGHT arm using Unity's HumanPoseHandler
// and the muscle-space API. This approach is avatar-agnostic: it works with
// any Humanoid-rigged model without hardcoding bone names or local axes.
//
// Muscle Space
// ------------
// Unity's Animator / Avatar system normalises all joint rotations into
// a "muscle space" where each degree of freedom (DOF) maps to a float
// in the range [−1, +1]:
//   −1 → minimum joint angle (as configured in the Avatar / T-pose rig)
//   +1 → maximum joint angle
//    0 → neutral / rest position
//
// Anatomical Mapping
// ------------------
// We map incoming degrees to muscle values via a configurable linear
// transform for each DOF:
//
//   muscle = (angle_deg - neutral_deg) / half_range_deg
//
// where neutral_deg is the angle that should map to muscle = 0, and
// half_range_deg is the degrees from neutral to the +1 extreme.
//
// Right-Arm Muscle Indices
// ------------------------
// Muscle names are searched at Awake() via HumanTrait.MuscleName so that
// the code is not sensitive to Unity version differences in muscle ordering.
//
//   "Right Arm Down-Up"      → shoulder abduction / adduction
//   "Right Arm Front-Back"   → shoulder flexion / extension
//   "Right Arm Roll In-Out"  → shoulder internal / external rotation
//   "Right Forearm Stretch"  → elbow flexion

using System;
using UnityEngine;

namespace MonoArm
{
    [RequireComponent(typeof(Animator))]
    public class AvatarMuscleController : MonoBehaviour
    {
        // ── Inspector ───────────────────────────────────────────────────────

        [Header("Smoothing")]
        [Tooltip("SmoothDamp time for each muscle value. Lower = faster but jitterier.")]
        [Range(0.02f, 0.5f)]
        public float smoothTime = 0.08f;

        [Header("Shoulder Flexion (Front-Back)")]
        [Tooltip("Degrees of flexion that maps to muscle +1. Typical: 120°–180°.")]
        public float flexionMax  = 150f;
        [Tooltip("Degrees of extension that maps to muscle -1. Typical: −30° to −60°.")]
        public float flexionMin  = -30f;

        [Header("Shoulder Abduction (Down-Up)")]
        [Tooltip("Degrees of abduction (arm up/out) that maps to muscle +1.")]
        public float abductionMax = 120f;
        [Tooltip("Degrees of adduction (arm crossing body) that maps to muscle -1.")]
        public float abductionMin = -20f;

        [Header("Shoulder Rotation (Roll In-Out)")]
        [Tooltip("Degrees of internal rotation that maps to muscle +1.")]
        public float rotationMax =  90f;
        [Tooltip("Degrees of external rotation that maps to muscle -1.")]
        public float rotationMin = -90f;

        [Header("Elbow Flexion (Forearm Stretch)")]
        [Tooltip("Degrees of elbow flexion that maps to muscle +1.")]
        public float elbowMax = 150f;

        [Header("Calibration Offsets (degrees)")]
        [Tooltip("Subtracted from incoming flexion before mapping. Use to zero the neutral pose.")]
        public float flexionOffset   = 0f;
        public float abductionOffset = 0f;
        public float rotationOffset  = 0f;
        public float elbowOffset     = 0f;

        // ── Private ─────────────────────────────────────────────────────────

        HumanPoseHandler _poseHandler;
        HumanPose        _pose;

        // Muscle indices resolved at Awake()
        int _idxFlexion   = -1;
        int _idxAbduction = -1;
        int _idxRotation  = -1;
        int _idxElbow     = -1;

        // SmoothDamp targets and velocities
        float _targetFlex, _targetAbd, _targetRot, _targetElbow;
        float _velFlex,    _velAbd,    _velRot,    _velElbow;

        // Last applied muscle values (for debug display)
        public float CurrentMuscleFlex   { get; private set; }
        public float CurrentMuscleAbd    { get; private set; }
        public float CurrentMuscleRot    { get; private set; }
        public float CurrentMuscleElbow  { get; private set; }

        // ── Unity Lifecycle ─────────────────────────────────────────────────

        void Awake()
        {
            var anim = GetComponent<Animator>();
            if (anim == null || !anim.isHuman)
            {
                Debug.LogError("[AvatarMuscleController] Animator not found or not Humanoid. " +
                               "Set Animation Type to Humanoid in the FBX import settings.");
                enabled = false;
                return;
            }

            _poseHandler = new HumanPoseHandler(anim.avatar, anim.transform);
            _poseHandler.GetHumanPose(ref _pose);

            ResolveMusclIndices();
        }

        void OnDestroy()
        {
            _poseHandler?.Dispose();
        }

        // ── Public API ──────────────────────────────────────────────────────

        /// <summary>
        /// Apply a set of anatomical arm angles to the avatar's right arm muscles.
        /// Call this once per frame from MonoArmManager.
        /// </summary>
        /// <param name="angles">Incoming joint angles (degrees) from the UDP receiver.</param>
        public void ApplyAngles(ArmAngles angles)
        {
            if (_poseHandler == null) return;
            if (_idxFlexion < 0)     return;   // muscles not resolved

            // Convert degrees → muscle values [-1, +1]
            float tFlex  = DegreesToMuscle(angles.shoulderFlexion   - flexionOffset,   flexionMin,   flexionMax);
            float tAbd   = DegreesToMuscle(angles.shoulderAbduction  - abductionOffset, abductionMin, abductionMax);
            float tRot   = DegreesToMuscle(angles.shoulderRotation   - rotationOffset,  rotationMin,  rotationMax);
            float tElbow = DegreesToMuscle(angles.elbowFlexion       - elbowOffset,     0f,           elbowMax);

            // SmoothDamp to target muscle values (frame-rate independent)
            float dt = Time.deltaTime;
            _targetFlex  = Mathf.SmoothDamp(_targetFlex,  tFlex,  ref _velFlex,  smoothTime, Mathf.Infinity, dt);
            _targetAbd   = Mathf.SmoothDamp(_targetAbd,   tAbd,   ref _velAbd,   smoothTime, Mathf.Infinity, dt);
            _targetRot   = Mathf.SmoothDamp(_targetRot,   tRot,   ref _velRot,   smoothTime, Mathf.Infinity, dt);
            _targetElbow = Mathf.SmoothDamp(_targetElbow, tElbow, ref _velElbow, smoothTime, Mathf.Infinity, dt);

            // Read current full-body pose, patch right-arm muscles, write back
            _poseHandler.GetHumanPose(ref _pose);

            _pose.muscles[_idxFlexion]   = _targetFlex;
            _pose.muscles[_idxAbduction] = _targetAbd;
            _pose.muscles[_idxRotation]  = _targetRot;
            _pose.muscles[_idxElbow]     = _targetElbow;

            _poseHandler.SetHumanPose(ref _pose);

            // Store for inspector / debug display
            CurrentMuscleFlex  = _targetFlex;
            CurrentMuscleAbd   = _targetAbd;
            CurrentMuscleRot   = _targetRot;
            CurrentMuscleElbow = _targetElbow;
        }

        // ── Private helpers ──────────────────────────────────────────────────

        /// <summary>
        /// Map an angle in degrees to Unity muscle space [−1, +1].
        /// Clamps to the valid muscle range.
        /// </summary>
        static float DegreesToMuscle(float angleDeg, float minDeg, float maxDeg)
        {
            float range  = maxDeg - minDeg;
            if (Mathf.Abs(range) < 1e-3f) return 0f;
            float normalised = (angleDeg - minDeg) / range;          // [0, 1]
            float muscle     = Mathf.Lerp(-1f, 1f, normalised);      // [−1, +1]
            return Mathf.Clamp(muscle, -1f, 1f);
        }

        void ResolveMusclIndices()
        {
            // Find each right-arm muscle by name substring match.
            // HumanTrait.MuscleName is consistent across Unity versions.
            string[] names = HumanTrait.MuscleName;
            for (int i = 0; i < names.Length; i++)
            {
                string n = names[i];
                if (n == "Right Arm Front-Back")  _idxFlexion   = i;
                else if (n == "Right Arm Down-Up")     _idxAbduction = i;
                else if (n == "Right Arm Roll In-Out")  _idxRotation  = i;
                else if (n == "Right Forearm Stretch")  _idxElbow     = i;
            }

            bool ok = _idxFlexion >= 0 && _idxAbduction >= 0 &&
                      _idxRotation >= 0 && _idxElbow >= 0;

            if (!ok)
            {
                Debug.LogError(
                    "[AvatarMuscleController] One or more right-arm muscle indices not found.\n" +
                    "Ensure the avatar is configured as Humanoid in the FBX import settings.\n" +
                    $"  Flexion idx   = {_idxFlexion}\n" +
                    $"  Abduction idx = {_idxAbduction}\n" +
                    $"  Rotation idx  = {_idxRotation}\n" +
                    $"  Elbow idx     = {_idxElbow}");
                enabled = false;
            }
            else
            {
                Debug.Log("[AvatarMuscleController] Right-arm muscles resolved: " +
                          $"flex={_idxFlexion} abd={_idxAbduction} " +
                          $"rot={_idxRotation} elbow={_idxElbow}");
            }
        }
    }
}