// ArmBoneController.cs
// ====================
// Drives one arm of a humanoid avatar by writing directly to the upper-arm and
// forearm bone transforms, using Transform.localRotation and Quaternion.Euler
// as the project specification requires.
//
// Relationship to AvatarMuscleController
// --------------------------------------
// Two controllers are provided because they fail in different ways:
//
//   AvatarMuscleController  writes through Unity's muscle system. It is
//     avatar-agnostic and automatically respects the joint limits configured
//     in the Avatar, but the mapping from degrees to muscle space depends on
//     how the rig was authored.
//
//   ArmBoneController (this file) writes bone rotations directly. It gives
//     exact, inspectable control of each axis and is the mapping described in
//     the specification, but it does not enforce the Avatar's joint limits, so
//     the per-axis ranges below are clamped explicitly.
//
// Put one of the two on an avatar, not both: they would fight over the same
// bones. MonoArmManager selects whichever it finds.
//
// Bone axis convention
// --------------------
// Which local axis of the upper-arm bone corresponds to flexion, abduction or
// axial rotation depends on how the model was rigged, and no single convention
// covers every avatar. Rather than hard-coding one and hoping, each anatomical
// degree of freedom is routed to a named local axis with a sign, exposed in the
// inspector. The defaults match the bundled X Bot (Mixamo) rig, in which the
// arm bones point down the model's X axis in the T-pose.
//
// To retune for a different rig: run the pipeline, hold your arm in one
// reference pose at a time, and adjust the axis and sign for that one degree of
// freedom until the avatar matches. The four DOFs are independent, so they can
// be resolved one at a time.

using UnityEngine;

namespace MonoArm
{
    /// <summary>Local axis a degree of freedom is applied to.</summary>
    public enum BoneAxis { X, Y, Z }

    [RequireComponent(typeof(Animator))]
    public class ArmBoneController : MonoBehaviour
    {
        // ── Inspector ───────────────────────────────────────────────────────

        [Header("Side")]
        [Tooltip("Which arm this controller drives. Add one instance per side.")]
        public ArmSide side = ArmSide.Right;

        [Header("Smoothing")]
        [Tooltip("Time constant in seconds. The interpolation is exponential and " +
                 "frame-rate independent, so the same value behaves identically " +
                 "at 30 and 144 fps.")]
        [Range(0.01f, 0.4f)]
        public float smoothingTau = 0.06f;

        [Header("Shoulder flexion / extension  (+ forward)")]
        public BoneAxis flexionAxis = BoneAxis.Z;
        public bool  flexionInvert  = false;
        public float flexionMin     = -60f;
        public float flexionMax     = 170f;

        [Header("Shoulder abduction / adduction  (+ away from body)")]
        public BoneAxis abductionAxis = BoneAxis.Y;
        public bool  abductionInvert  = true;
        public float abductionMin     = -30f;
        public float abductionMax     = 170f;

        [Header("Shoulder internal / external rotation")]
        public BoneAxis rotationAxis = BoneAxis.X;
        public bool  rotationInvert  = false;
        public float rotationMin     = -90f;
        public float rotationMax     = 90f;

        [Header("Elbow flexion  (0 = straight)")]
        public BoneAxis elbowAxis = BoneAxis.Y;
        public bool  elbowInvert  = true;
        public float elbowMin     = 0f;
        public float elbowMax     = 150f;

        [Header("Neutral offsets (degrees, subtracted before mapping)")]
        public float flexionOffset   = 0f;
        public float abductionOffset = 0f;
        public float rotationOffset  = 0f;
        public float elbowOffset     = 0f;

        // ── State ───────────────────────────────────────────────────────────

        Transform  _upperArm;
        Transform  _foreArm;
        Quaternion _upperRest = Quaternion.identity;
        Quaternion _foreRest  = Quaternion.identity;
        Quaternion _upperTarget = Quaternion.identity;
        Quaternion _foreTarget  = Quaternion.identity;
        bool _ready;

        /// <summary>Angles most recently applied, after clamping. For the debug UI.</summary>
        public Vector4 AppliedDegrees { get; private set; }

        // ── Unity lifecycle ─────────────────────────────────────────────────

        void Awake()
        {
            var anim = GetComponent<Animator>();
            if (anim == null || !anim.isHuman)
            {
                Debug.LogError("[ArmBoneController] Animator missing or not Humanoid. " +
                               "Set Animation Type to Humanoid in the FBX import settings.");
                enabled = false;
                return;
            }

            bool right = side == ArmSide.Right;
            _upperArm = anim.GetBoneTransform(right ? HumanBodyBones.RightUpperArm
                                                    : HumanBodyBones.LeftUpperArm);
            _foreArm  = anim.GetBoneTransform(right ? HumanBodyBones.RightLowerArm
                                                    : HumanBodyBones.LeftLowerArm);

            if (_upperArm == null || _foreArm == null)
            {
                Debug.LogError($"[ArmBoneController:{side}] Upper arm or forearm bone not " +
                               "mapped in the Avatar definition. Open the model's Avatar " +
                               "configuration and assign both bones.");
                enabled = false;
                return;
            }

            // The rest pose is the reference every rotation is applied relative
            // to, so the avatar's authored T-pose is preserved as the zero.
            _upperRest   = _upperArm.localRotation;
            _foreRest    = _foreArm.localRotation;
            _upperTarget = _upperRest;
            _foreTarget  = _foreRest;
            _ready = true;
        }

        void LateUpdate()
        {
            if (!_ready) return;

            // Exponential interpolation toward the target. Using
            // 1 - exp(-dt / tau) rather than a raw dt factor keeps the response
            // identical regardless of the rendering frame rate, which matters
            // because the incoming angle stream runs at its own fixed rate.
            float k = 1f - Mathf.Exp(-Time.deltaTime / Mathf.Max(smoothingTau, 1e-4f));
            _upperArm.localRotation = Quaternion.Slerp(_upperArm.localRotation, _upperTarget, k);
            _foreArm.localRotation  = Quaternion.Slerp(_foreArm.localRotation,  _foreTarget,  k);
        }

        // ── Public API ──────────────────────────────────────────────────────

        /// <summary>
        /// Set the arm pose from anatomical joint angles in degrees. Called once
        /// per received packet; the actual bone write is interpolated in
        /// LateUpdate so animation stays smooth between packets.
        /// </summary>
        public void ApplyAngles(ArmAngles a)
        {
            if (!_ready) return;

            float flex  = Mathf.Clamp(a.shoulderFlexion   - flexionOffset,   flexionMin,   flexionMax);
            float abd   = Mathf.Clamp(a.shoulderAbduction - abductionOffset, abductionMin, abductionMax);
            float rot   = Mathf.Clamp(a.shoulderRotation  - rotationOffset,  rotationMin,  rotationMax);
            float elbow = Mathf.Clamp(a.elbowFlexion      - elbowOffset,     elbowMin,     elbowMax);

            AppliedDegrees = new Vector4(flex, abd, rot, elbow);

            // Shoulder: three degrees of freedom composed onto one Euler vector.
            Vector3 shoulderEuler = Vector3.zero;
            AddAxis(ref shoulderEuler, flexionAxis,   flexionInvert   ? -flex : flex);
            AddAxis(ref shoulderEuler, abductionAxis, abductionInvert ? -abd  : abd);
            AddAxis(ref shoulderEuler, rotationAxis,  rotationInvert  ? -rot  : rot);

            // Elbow: one degree of freedom.
            Vector3 elbowEuler = Vector3.zero;
            AddAxis(ref elbowEuler, elbowAxis, elbowInvert ? -elbow : elbow);

            // Rotations are applied relative to the authored rest pose so the
            // avatar returns exactly to its T-pose when all angles are zero.
            _upperTarget = _upperRest * Quaternion.Euler(shoulderEuler);
            _foreTarget  = _foreRest  * Quaternion.Euler(elbowEuler);
        }

        /// <summary>Return the arm to its authored rest pose.</summary>
        public void ResetToRest()
        {
            _upperTarget = _upperRest;
            _foreTarget  = _foreRest;
        }

        // ── Helpers ─────────────────────────────────────────────────────────

        static void AddAxis(ref Vector3 euler, BoneAxis axis, float degrees)
        {
            switch (axis)
            {
                case BoneAxis.X: euler.x += degrees; break;
                case BoneAxis.Y: euler.y += degrees; break;
                case BoneAxis.Z: euler.z += degrees; break;
            }
        }
    }
}
