// SceneBuilder.cs  — Run once from the Unity menu to build the 4-avatar scene.
//
// Usage:
//   1. Copy this file into Assets/Scripts/Editor/
//   2. In Unity: top menu → PoseTrack → Build 4-Avatar Scene
//   3. Avatars are positioned side-by-side with name labels above each
//   4. Delete this file after running (it is only needed once)
//
// What it creates:
//   • PoseManager  (UdpAngleReceiver + MultiAvatarManager)
//   • 4 × X Bot instances (one per stream: MP / MV / FU / GR)
//   • Name labels above each avatar (using UI.Text)
//   • Camera positioned to see all 4

#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEngine.UI;

namespace PoseTrackReceiver
{
    public static class SceneBuilder
    {
        // X Bot FBX must be at this path relative to Assets/
        const string XBOT_PATH = "Assets/X Bot.fbx";

        // Avatar positions — spread 2 units apart on X axis
        static readonly Vector3[] POSITIONS = {
            new Vector3(-3f, 0f, 0f),
            new Vector3(-1f, 0f, 0f),
            new Vector3( 1f, 0f, 0f),
            new Vector3( 3f, 0f, 0f),
        };

        static readonly string[] LABELS = {
            "MediaPipe (Raw)",
            "MoveNet (Raw)",
            "DeepFusionPose",
            "GAN Refined",
        };

        static readonly string[] GO_NAMES = {
            "Avatar_MediaPipe",
            "Avatar_MoveNet",
            "Avatar_Fusion",
            "Avatar_GAN",
        };

        [MenuItem("PoseTrack/Build 4-Avatar Scene")]
        static void Build()
        {
            // ── Load X Bot prefab ────────────────────────────────────────────
            var xbotPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(XBOT_PATH);
            if (xbotPrefab == null)
            {
                EditorUtility.DisplayDialog("SceneBuilder",
                    $"Could not find X Bot at: {XBOT_PATH}\n" +
                    "Make sure 'X Bot.fbx' is in your Assets folder.", "OK");
                return;
            }

            // ── Create PoseManager ───────────────────────────────────────────
            var poseManager = new GameObject("PoseManager");
            var receiver    = poseManager.AddComponent<UdpAngleReceiver>();
            var manager     = poseManager.AddComponent<MultiAvatarManager>();
            receiver.listenPort = 9000;

            // ── Create 4 Avatars ─────────────────────────────────────────────
            var controllers = new ArmAngleController[4];

            for (int i = 0; i < 4; i++)
            {
                // Instantiate X Bot
                var avatar = (GameObject)PrefabUtility.InstantiatePrefab(xbotPrefab);
                avatar.name = GO_NAMES[i];
                avatar.transform.position = POSITIONS[i];

                // Add ArmAngleController as child of avatar root
                var ctrl = avatar.AddComponent<ArmAngleController>();

                // Auto-find humanoid bones
                var animator = avatar.GetComponent<Animator>();
                if (animator != null && animator.isHuman)
                {
                    ctrl.upperArmBone = animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
                    ctrl.lowerArmBone = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
                }
                else
                {
                    Debug.LogWarning($"[SceneBuilder] Avatar {GO_NAMES[i]} has no Humanoid Animator — assign bones manually.");
                }

                controllers[i] = ctrl;

                // ── Add floating text label above avatar ─────────────────────
                CreateFloatingLabel(LABELS[i], POSITIONS[i] + new Vector3(0f, 2.4f, 0f));
            }

            // ── Wire MultiAvatarManager ──────────────────────────────────────
            manager.receiver        = receiver;
            manager.avatarMediaPipe = controllers[0];
            manager.avatarMoveNet   = controllers[1];
            manager.avatarFusion    = controllers[2];
            manager.avatarGAN       = controllers[3];

            // ── Reposition Main Camera to see all 4 avatars ──────────────────
            var cam = Camera.main;
            if (cam != null)
            {
                cam.transform.position = new Vector3(0f, 1.5f, -6f);
                cam.transform.rotation = Quaternion.Euler(5f, 0f, 0f);
            }

            // ── Mark scene dirty ─────────────────────────────────────────────
            UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(
                UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene());

            Debug.Log("[SceneBuilder] 4-avatar scene built. Save the scene (Ctrl+S).");
            EditorUtility.DisplayDialog("SceneBuilder",
                "4-avatar scene built successfully!\n\n" +
                "• PoseManager (UdpAngleReceiver + MultiAvatarManager)\n" +
                "• Avatar_MediaPipe, Avatar_MoveNet, Avatar_Fusion, Avatar_GAN\n\n" +
                "Press Play, then run mock_streamer.py to test.", "OK");
        }

        // Creates a world-space 3D text label (Billboard Text)
        static void CreateFloatingLabel(string text, Vector3 worldPos)
        {
            // Use a world-space canvas per label
            var go        = new GameObject($"Label_{text.Split(' ')[0]}");
            go.transform.position = worldPos;

            var canvas    = go.AddComponent<Canvas>();
            canvas.renderMode      = RenderMode.WorldSpace;
            canvas.worldCamera     = Camera.main;

            go.AddComponent<CanvasScaler>();
            go.GetComponent<RectTransform>().sizeDelta = new Vector2(2f, 0.4f);
            go.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);

            var textGo    = new GameObject("Text");
            textGo.transform.SetParent(go.transform, false);

            var rt        = textGo.AddComponent<RectTransform>();
            rt.sizeDelta  = new Vector2(200f, 40f);

            var uiText    = textGo.AddComponent<Text>();
            uiText.text      = text;
            uiText.fontSize  = 18;
            uiText.alignment = TextAnchor.MiddleCenter;
            uiText.color     = Color.white;
            uiText.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        }
    }
}
#endif
