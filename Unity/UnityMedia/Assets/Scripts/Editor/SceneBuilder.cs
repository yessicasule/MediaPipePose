#if UNITY_EDITOR
// SceneBuilder.cs
// ================
// Editor utility to automatically build the MonoArm scene.
//
// Menu: MonoArm / Build Scene
//   1. Finds the humanoid avatar in the scene (or loads X Bot as fallback).
//   2. Attaches AvatarMuscleController, UdpAngleReceiver, MonoArmManager,
//      and PoseDebugUI to a PoseManager GameObject.
//   3. Positions the camera for a clear avatar view.
//   4. Saves the scene.
//
// Menu: MonoArm / Diagnose Scene
//   Reports the status of all required components to the Console.
//
// Menu: MonoArm / Undo Last Build
//   Undoes the last Build Scene operation.

using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace MonoArm
{
    public static class SceneBuilder
    {
        // ── Build ────────────────────────────────────────────────────────────

        [MenuItem("MonoArm/Build Scene")]
        static void Build()
        {
            // Remove old MonoArm manager objects (not the avatar)
            foreach (var name in new[] { "PoseManager", "MonoArmManager" })
            {
                var go = GameObject.Find(name);
                if (go != null) Undo.DestroyObjectImmediate(go);
            }

            // Find or instantiate humanoid avatar
            GameObject avatar = FindHumanoidInScene();
            if (avatar == null)
            {
                // Try X Bot fallback
                string[] guids = AssetDatabase.FindAssets("X Bot t:GameObject");
                if (guids.Length == 0)
                {
                    EditorUtility.DisplayDialog("MonoArm — No Avatar Found",
                        "No Humanoid avatar is in the scene and no 'X Bot' asset was found.\n\n" +
                        "Please drag a Humanoid-rigged FBX into the scene, then run Build Scene again.\n\n" +
                        "You can get free avatars from Mixamo (mixamo.com).",
                        "OK");
                    return;
                }
                string path   = AssetDatabase.GUIDToAssetPath(guids[0]);
                var    prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                avatar        = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                avatar.name   = "Avatar_MonoArm";
                avatar.transform.position = Vector3.zero;
                Debug.Log($"[SceneBuilder] Instantiated {path} as avatar.");
            }

            // Check humanoid rig
            var animator = avatar.GetComponentInChildren<Animator>();
            if (animator == null || !animator.isHuman)
            {
                EditorUtility.DisplayDialog("MonoArm — Humanoid Required",
                    $"'{avatar.name}' is not configured as a Humanoid.\n\n" +
                    "Select the FBX in the Project window → Inspector → Rig → " +
                    "Animation Type = Humanoid → Apply, then run Build Scene again.",
                    "OK");
                return;
            }

            // Remove any stale components
            RemoveComponent<AvatarMuscleController>(avatar);

            // Add AvatarMuscleController to the avatar
            var ctrl = Undo.AddComponent<AvatarMuscleController>(avatar);
            EditorUtility.SetDirty(ctrl);

            // Create PoseManager GameObject
            var poseManagerGO = new GameObject("PoseManager");
            Undo.RegisterCreatedObjectUndo(poseManagerGO, "Build MonoArm Scene");

            var receiver   = poseManagerGO.AddComponent<UdpAngleReceiver>();
            var manager    = poseManagerGO.AddComponent<MonoArmManager>();
            var debugUI    = avatar.AddComponent<PoseDebugUI>();   // parented to avatar

            receiver.listenPort         = 9000;
            manager.receiver            = receiver;
            manager.avatarController    = ctrl;
            debugUI.receiver            = receiver;

            // Camera
            var cam = Camera.main;
            if (cam != null)
            {
                cam.transform.position = new Vector3(0f, 1.2f, -3.5f);
                cam.transform.rotation = Quaternion.Euler(5f, 0f, 0f);
            }

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());

            EditorUtility.DisplayDialog("MonoArm — Scene Built",
                $"Scene built successfully!\n\n" +
                $"Avatar:  {avatar.name}\n" +
                $"UDP port: {receiver.listenPort}\n\n" +
                "Next steps:\n" +
                "1. Save the scene (Ctrl+S)\n" +
                "2. Run the Python pipeline:\n" +
                "   python scripts/run_demo.py\n" +
                "   -- or for testing without camera --\n" +
                "   python scripts/mock_streamer.py --mode sinusoidal\n" +
                "3. Press Play in Unity.",
                "OK");
        }

        // ── Diagnose ─────────────────────────────────────────────────────────

        [MenuItem("MonoArm/Diagnose Scene")]
        static void Diagnose()
        {
            var sb = new System.Text.StringBuilder();

            var receivers = Object.FindObjectsOfType<UdpAngleReceiver>();
            sb.AppendLine($"UdpAngleReceiver count: {receivers.Length}  (should be 1)");
            foreach (var r in receivers)
                sb.AppendLine($"  → on '{r.gameObject.name}' port={r.listenPort}");

            sb.AppendLine();

            var managers = Object.FindObjectsOfType<MonoArmManager>();
            sb.AppendLine($"MonoArmManager count: {managers.Length}  (should be 1)");
            foreach (var m in managers)
            {
                sb.AppendLine($"  receiver linked:   {m.receiver != null}");
                sb.AppendLine($"  controller linked: {m.avatarController != null}");
            }

            sb.AppendLine();

            var ctrls = Object.FindObjectsOfType<AvatarMuscleController>();
            sb.AppendLine($"AvatarMuscleController count: {ctrls.Length}  (should be 1)");
            foreach (var c in ctrls)
                sb.AppendLine($"  → on '{c.gameObject.name}'  enabled={c.enabled}");

            Debug.Log("[MonoArm Diagnose]\n" + sb);
            EditorUtility.DisplayDialog("MonoArm Scene Diagnosis", sb.ToString(), "OK");
        }

        // ── Undo ──────────────────────────────────────────────────────────────

        [MenuItem("MonoArm/Undo Last Build")]
        static void UndoBuild()
        {
            Undo.PerformUndo();
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        static GameObject FindHumanoidInScene()
        {
            foreach (var anim in Object.FindObjectsOfType<Animator>())
                if (anim.isHuman && anim.gameObject.scene.IsValid())
                    return anim.gameObject;
            return null;
        }

        static void RemoveComponent<T>(GameObject go) where T : Component
        {
            var c = go.GetComponent<T>();
            if (c != null) Undo.DestroyObjectImmediate(c);
        }
    }
}
#endif
