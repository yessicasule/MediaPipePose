using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;

namespace PoseTrackReceiver
{
    [CustomEditor(typeof(ArmAngleController))]
    public class ArmRigSetup : Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var ctrl = (ArmAngleController)target;

            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Rig Auto-Setup", EditorStyles.boldLabel);

            if (GUILayout.Button("Auto-Find Humanoid Bones"))
            {
                var animator = ctrl.GetComponentInParent<Animator>();
                if (animator == null)
                    animator = ctrl.GetComponentInChildren<Animator>();

                if (animator != null && animator.isHuman)
                {
                    ctrl.upperArmBone = animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
                    ctrl.lowerArmBone = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
                    EditorUtility.SetDirty(ctrl);
                    Debug.Log("[ArmRigSetup] Bones assigned from Humanoid rig.");
                }
                else
                {
                    EditorUtility.DisplayDialog("ArmRigSetup",
                        "No Humanoid Animator found on this or parent GameObjects.\n" +
                        "Assign bones manually in the Inspector.", "OK");
                }
            }

            EditorGUILayout.HelpBox(
                "1. Attach this component to the same GameObject as UdpAngleReceiver.\n" +
                "2. Click 'Auto-Find Humanoid Bones' or drag bones manually.\n" +
                "3. Use 'shoulderAxisOffset' and 'elbowAxisOffset' to fix coordinate mismatch.",
                MessageType.Info);
        }
    }
}
#endif
