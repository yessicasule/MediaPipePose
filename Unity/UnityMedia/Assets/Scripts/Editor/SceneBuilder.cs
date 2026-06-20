#if UNITY_EDITOR
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

namespace PoseTrackReceiver
{
    public static class SceneBuilder
    {
        static readonly Vector3[] POSITIONS = {
            new Vector3(-3f, 0f, 0f),
            new Vector3(-1f, 0f, 0f),
            new Vector3( 1f, 0f, 0f),
            new Vector3( 3f, 0f, 0f),
        };

        static readonly string[] LABELS   = { "MediaPipe (Raw)", "MoveNet (Raw)", "DeepFusionPose", "GAN Refined" };
        static readonly string[] GO_NAMES = { "Avatar_MediaPipe", "Avatar_MoveNet", "Avatar_Fusion", "Avatar_GAN" };

        // ─────────────────────────────────────────────────────────────────────
        // MENU 1: Build/Rebuild the full 4-avatar scene
        // ─────────────────────────────────────────────────────────────────────
        [MenuItem("PoseTrack/Build 4-Avatar Scene")]
        static void Build()
        {
            // Step 1: remove only our own manager and label objects — NOT the avatars
            RemoveOurManagerObjects();

            // Step 2: collect humanoid GameObjects already in the scene
            var humanoids = FindHumanoidsInScene();

            if (humanoids.Count == 0)
            {
                // Try to load X Bot as fallback
                string[] guids = AssetDatabase.FindAssets("X Bot t:GameObject");
                if (guids.Length == 0)
                {
                    EditorUtility.DisplayDialog("SceneBuilder — No Avatars Found",
                        "No Humanoid avatars are in the scene, and no 'X Bot' asset was found.\n\n" +
                        "Please add at least one Humanoid avatar to the scene first, then run this menu again.\n\n" +
                        "You can import a Humanoid FBX (e.g. from Mixamo) into Assets/ and drag it into the scene.",
                        "OK");
                    return;
                }

                // Load and instantiate X Bot 4 times
                string path   = AssetDatabase.GUIDToAssetPath(guids[0]);
                var    prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                for (int i = 0; i < 4; i++)
                {
                    var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                    go.name = GO_NAMES[i];
                    go.transform.position = POSITIONS[i];
                    humanoids.Add(go);
                }
            }
            else if (humanoids.Count == 1)
            {
                // Duplicate the single avatar 3 more times
                var source = humanoids[0];
                source.name = GO_NAMES[0];
                source.transform.position = POSITIONS[0];
                for (int i = 1; i < 4; i++)
                {
                    var copy = Object.Instantiate(source);
                    copy.name = GO_NAMES[i];
                    copy.transform.position = POSITIONS[i];
                    humanoids.Add(copy);
                }
            }
            else
            {
                // Rename and position whatever humanoids exist (up to 4)
                for (int i = 0; i < Mathf.Min(4, humanoids.Count); i++)
                {
                    humanoids[i].name = GO_NAMES[i];
                    humanoids[i].transform.position = POSITIONS[i];
                }
                // If fewer than 4, duplicate the last one to fill
                while (humanoids.Count < 4)
                {
                    var copy = Object.Instantiate(humanoids[humanoids.Count - 1]);
                    copy.name = GO_NAMES[humanoids.Count];
                    copy.transform.position = POSITIONS[humanoids.Count];
                    humanoids.Add(copy);
                }
            }

            // Step 3: add ArmAngleController to each avatar (only if not already present)
            var controllers = new ArmAngleController[4];
            for (int i = 0; i < 4; i++)
            {
                var avatar = humanoids[i];

                // Remove any stale UdpAngleReceiver from avatar (the bug we fixed)
                var staleUdp = avatar.GetComponent<UdpAngleReceiver>();
                if (staleUdp != null) Object.DestroyImmediate(staleUdp);

                var ctrl = avatar.GetComponent<ArmAngleController>()
                        ?? avatar.AddComponent<ArmAngleController>();

                var animator = avatar.GetComponentInChildren<Animator>();
                if (animator != null && animator.isHuman)
                {
                    ctrl.upperArmBone = animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
                    ctrl.lowerArmBone = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
                    Debug.Log($"[SceneBuilder] {avatar.name}: bones assigned — " +
                              $"upperArm={ctrl.upperArmBone?.name}  lowerArm={ctrl.lowerArmBone?.name}");
                }
                else
                {
                    Debug.LogWarning($"[SceneBuilder] {avatar.name}: no Humanoid Animator found. " +
                                     "Set Animation Type to Humanoid in the FBX import settings, then re-run.");
                }

                controllers[i] = ctrl;
                CreateFloatingLabel(LABELS[i], humanoids[i].transform.position + new Vector3(0f, 2.4f, 0f));
            }

            // Step 4: create PoseManager with ONE UdpAngleReceiver + HUD
            var poseManager = new GameObject("PoseManager");
            var receiver    = poseManager.AddComponent<UdpAngleReceiver>();
            var manager     = poseManager.AddComponent<MultiAvatarManager>();
            var hud         = poseManager.AddComponent<PoseDebugUI>();
            receiver.listenPort     = 9000;
            manager.receiver        = receiver;
            manager.avatarMediaPipe = controllers[0];
            manager.avatarMoveNet   = controllers[1];
            manager.avatarFusion    = controllers[2];
            manager.avatarGAN       = controllers[3];
            hud.receiver            = receiver;   // wire directly — no manual drag needed

            // Step 5: camera
            var cam = Camera.main;
            if (cam != null)
            {
                cam.transform.position = new Vector3(0f, 1.5f, -6f);
                cam.transform.rotation = Quaternion.Euler(5f, 0f, 0f);
            }

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log("[SceneBuilder] Done. Save (Ctrl+S) then press Play.");

            string boneReport = string.Join("\n", controllers.Select((c, i) =>
                $"  {GO_NAMES[i]}: upper={c?.upperArmBone?.name ?? "MISSING"}, " +
                $"lower={c?.lowerArmBone?.name ?? "MISSING"}"));

            EditorUtility.DisplayDialog("SceneBuilder — Done",
                "Scene built!\n\nBone assignment:\n" + boneReport +
                "\n\nIf any show MISSING, select that avatar → FBX Import Settings → " +
                "Animation Type = Humanoid, then re-run.\n\nSave (Ctrl+S) then press Play.", "OK");
        }

        // ─────────────────────────────────────────────────────────────────────
        // MENU 2: Diagnose why avatars are not moving
        // ─────────────────────────────────────────────────────────────────────
        [MenuItem("PoseTrack/Diagnose Scene")]
        static void Diagnose()
        {
            var sb = new System.Text.StringBuilder();

            // Check UdpAngleReceiver count
            var receivers = Object.FindObjectsOfType<UdpAngleReceiver>();
            sb.AppendLine($"UdpAngleReceiver count: {receivers.Length}  (should be 1)");
            foreach (var r in receivers)
                sb.AppendLine($"  - on '{r.gameObject.name}' port={r.listenPort}");

            sb.AppendLine();

            // Check MultiAvatarManager
            var managers = Object.FindObjectsOfType<MultiAvatarManager>();
            sb.AppendLine($"MultiAvatarManager count: {managers.Length}  (should be 1)");
            foreach (var m in managers)
            {
                sb.AppendLine($"  receiver linked: {m.receiver != null}");
                sb.AppendLine($"  avatarMP  linked: {m.avatarMediaPipe != null}");
                sb.AppendLine($"  avatarMV  linked: {m.avatarMoveNet   != null}");
                sb.AppendLine($"  avatarFU  linked: {m.avatarFusion    != null}");
                sb.AppendLine($"  avatarGAN linked: {m.avatarGAN       != null}");
            }

            sb.AppendLine();

            // Check ArmAngleController bones
            var controllers = Object.FindObjectsOfType<ArmAngleController>();
            sb.AppendLine($"ArmAngleController count: {controllers.Length}  (should be 4)");
            foreach (var c in controllers)
            {
                sb.AppendLine($"  '{c.gameObject.name}': " +
                              $"upper={c.upperArmBone?.name ?? "NULL"}, " +
                              $"lower={c.lowerArmBone?.name ?? "NULL"}");
            }

            Debug.Log("[SceneBuilder Diagnose]\n" + sb);
            EditorUtility.DisplayDialog("PoseTrack Scene Diagnosis", sb.ToString(), "OK");
        }

        // ─────────────────────────────────────────────────────────────────────
        // MENU 3: Undo only — restore scene from Ctrl+Z if CleanScene wiped too much
        // ─────────────────────────────────────────────────────────────────────
        [MenuItem("PoseTrack/Undo Last Build (Ctrl+Z)")]
        static void UndoBuild()
        {
            Undo.PerformUndo();
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log("[SceneBuilder] Undo performed.");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helpers
        // ─────────────────────────────────────────────────────────────────────

        // Only removes PoseManager, PoseReceiver, and label objects — never avatar humanoids
        static void RemoveOurManagerObjects()
        {
            foreach (var name in new[] { "PoseManager", "PoseReceiver" })
            {
                var go = GameObject.Find(name);
                if (go != null) Undo.DestroyObjectImmediate(go);
            }
            foreach (var name in new[] { "Label_MediaPipe", "Label_MoveNet",
                                         "Label_DeepFusionPose", "Label_GAN" })
            {
                var go = GameObject.Find(name);
                if (go != null) Undo.DestroyObjectImmediate(go);
            }
        }

        // Find all Humanoid-rigged GameObjects in the scene
        static List<GameObject> FindHumanoidsInScene()
        {
            var results = new List<GameObject>();
            foreach (var animator in Object.FindObjectsOfType<Animator>())
            {
                // isHuman requires the avatar to be configured as Humanoid in FBX import
                if (animator.isHuman && animator.gameObject.scene.IsValid())
                    results.Add(animator.gameObject);
            }
            return results;
        }

        static void CreateFloatingLabel(string text, Vector3 worldPos)
        {
            var go     = new GameObject($"Label_{text.Split(' ')[0]}");
            go.transform.position   = worldPos;
            go.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);

            var canvas = go.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.worldCamera = Camera.main;
            go.AddComponent<CanvasScaler>();
            go.GetComponent<RectTransform>().sizeDelta = new Vector2(2f, 0.4f);

            var textGo = new GameObject("Text");
            textGo.transform.SetParent(go.transform, false);
            textGo.AddComponent<RectTransform>().sizeDelta = new Vector2(200f, 40f);

            var t = textGo.AddComponent<Text>();
            t.text      = text;
            t.fontSize  = 18;
            t.alignment = TextAnchor.MiddleCenter;
            t.color     = Color.white;
            t.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        }
    }
}
#endif
