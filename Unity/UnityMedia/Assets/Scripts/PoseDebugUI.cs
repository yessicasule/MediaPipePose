using System.Text;
using UnityEngine;
using UnityEngine.UI;

namespace PoseTrackReceiver
{
    /// <summary>
    /// Places a world-space info panel directly below each avatar.
    /// Panels are parented to the avatar so they move with it.
    /// </summary>
    public class PoseDebugUI : MonoBehaviour
    {
        public UdpAngleReceiver receiver;
        [Range(1f, 20f)] public float refreshHz = 10f;

        // World-space panel config
        const float PANEL_W     = 1.60f;   // world units wide
        const float PANEL_H     = 1.20f;   // world units tall
        const float PANEL_SCALE = 0.005f;  // canvas scale (world unit per pixel)
        const float PANEL_Y     = -1.10f;  // below avatar root
        const float PANEL_Z     = 0.05f;   // slightly in front

        Text[]  _labels = new Text[4];
        float   _nextRefresh;
        float   _lastDataTime;
        int     _totalPackets;

        static readonly string[] NAMES  = { "MediaPipe", "MoveNet", "Fusion", "GAN" };
        static readonly Color[]  COLORS = {
            new Color(0.35f, 0.80f, 1.00f),   // cyan
            new Color(1.00f, 0.72f, 0.20f),   // amber
            new Color(0.30f, 1.00f, 0.50f),   // green
            new Color(1.00f, 0.40f, 0.80f),   // pink
        };

        readonly StringBuilder _sb = new(256);

        // ── Lifecycle ─────────────────────────────────────────────────────────
        void Awake()
        {
            if (receiver == null)
                receiver = FindFirstObjectByType<UdpAngleReceiver>();

            // Remove stray UI objects from old setup
            foreach (var n in new[] { "New Text", "PoseHUD_Canvas" })
            { var g = GameObject.Find(n); if (g) Destroy(g); }

            AttachPanelsToAvatars();
        }

        void Update()
        {
            if (receiver == null) return;
            if (receiver.HasData) { _totalPackets++; _lastDataTime = Time.unscaledTime; }
            if (Time.unscaledTime < _nextRefresh) return;
            _nextRefresh = Time.unscaledTime + 1f / Mathf.Max(refreshHz, 1f);
            RefreshAll();
        }

        // ── Find the 4 avatar roots via MultiAvatarManager ────────────────────
        void AttachPanelsToAvatars()
        {
            var mgr = FindFirstObjectByType<MultiAvatarManager>();
            if (mgr == null)
            {
                Debug.LogWarning("[PoseDebugUI] MultiAvatarManager not found — panels not created.");
                return;
            }

            Transform[] roots = {
                mgr.avatarMediaPipe != null ? mgr.avatarMediaPipe.transform : null,
                mgr.avatarMoveNet   != null ? mgr.avatarMoveNet.transform   : null,
                mgr.avatarFusion    != null ? mgr.avatarFusion.transform    : null,
                mgr.avatarGAN       != null ? mgr.avatarGAN.transform       : null,
            };

            for (int i = 0; i < 4; i++)
            {
                if (roots[i] == null)
                {
                    Debug.LogWarning($"[PoseDebugUI] Avatar slot {i} ({NAMES[i]}) is null — skipping panel.");
                    continue;
                }
                _labels[i] = CreateWorldPanel(roots[i], i);
            }
        }

        // ── Create one world-space panel parented to an avatar ────────────────
        Text CreateWorldPanel(Transform avatarRoot, int idx)
        {
            // Canvas GO — child of the avatar root
            var cvGO   = new GameObject($"InfoPanel_{NAMES[idx]}");
            cvGO.transform.SetParent(avatarRoot, false);
            cvGO.transform.localPosition = new Vector3(0f, PANEL_Y, PANEL_Z);
            cvGO.transform.localRotation = Quaternion.identity;
            cvGO.transform.localScale    = Vector3.one * PANEL_SCALE;

            var cv            = cvGO.AddComponent<Canvas>();
            cv.renderMode     = RenderMode.WorldSpace;
            cv.sortingOrder   = 10;
            cvGO.AddComponent<CanvasScaler>();

            // Size the canvas RectTransform
            var cvRT       = cvGO.GetComponent<RectTransform>();
            cvRT.sizeDelta = new Vector2(PANEL_W / PANEL_SCALE, PANEL_H / PANEL_SCALE);  // pixels

            // Background panel
            var bgGO  = new GameObject("BG");
            bgGO.transform.SetParent(cvGO.transform, false);
            var bgRT          = bgGO.AddComponent<RectTransform>();
            bgRT.anchorMin    = Vector2.zero;
            bgRT.anchorMax    = Vector2.one;
            bgRT.offsetMin    = Vector2.zero;
            bgRT.offsetMax    = Vector2.zero;
            bgGO.AddComponent<Image>().color = new Color(0.05f, 0.05f, 0.08f, 0.88f);

            // Coloured top accent bar (30px)
            var barGO         = new GameObject("AccentBar");
            barGO.transform.SetParent(cvGO.transform, false);
            var barRT         = barGO.AddComponent<RectTransform>();
            barRT.anchorMin   = new Vector2(0f, 1f);
            barRT.anchorMax   = new Vector2(1f, 1f);
            barRT.pivot       = new Vector2(0.5f, 1f);
            barRT.offsetMin   = new Vector2(0f, -30f);
            barRT.offsetMax   = new Vector2(0f,   0f);
            barGO.AddComponent<Image>().color = COLORS[idx];

            // Name label inside accent bar
            var nameGO         = new GameObject("StreamName");
            nameGO.transform.SetParent(cvGO.transform, false);
            var nameRT         = nameGO.AddComponent<RectTransform>();
            nameRT.anchorMin   = new Vector2(0f, 1f);
            nameRT.anchorMax   = new Vector2(1f, 1f);
            nameRT.pivot       = new Vector2(0.5f, 1f);
            nameRT.offsetMin   = new Vector2(8f,  -30f);
            nameRT.offsetMax   = new Vector2(-8f,  -2f);
            var nameText       = nameGO.AddComponent<Text>();
            nameText.text      = NAMES[idx];
            nameText.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            nameText.fontSize  = 22;
            nameText.fontStyle = FontStyle.Bold;
            nameText.color     = new Color(0.05f, 0.05f, 0.05f);
            nameText.alignment = TextAnchor.MiddleCenter;

            // Body text — fills panel below the accent bar
            var textGO         = new GameObject("DataText");
            textGO.transform.SetParent(cvGO.transform, false);
            var textRT         = textGO.AddComponent<RectTransform>();
            textRT.anchorMin   = new Vector2(0f, 0f);
            textRT.anchorMax   = new Vector2(1f, 1f);
            textRT.offsetMin   = new Vector2(14f,  10f);
            textRT.offsetMax   = new Vector2(-14f, -36f);  // 36 = below accent bar

            var txt                    = textGO.AddComponent<Text>();
            txt.font                   = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            txt.fontSize               = 20;
            txt.lineSpacing            = 1.25f;
            txt.color                  = Color.white;
            txt.alignment              = TextAnchor.UpperLeft;
            txt.supportRichText        = true;
            txt.horizontalOverflow     = HorizontalWrapMode.Overflow;
            txt.verticalOverflow       = VerticalWrapMode.Overflow;

            return txt;
        }

        // ── Refresh ───────────────────────────────────────────────────────────
        void RefreshAll()
        {
            bool waiting = receiver == null || !receiver.HasData;
            float  age   = Time.unscaledTime - _lastDataTime;
            string warn  = (!waiting && age > 0.5f) ? "  <color=#f55>⚠</color>" : "";

            if (waiting)
            {
                for (int i = 0; i < 4; i++)
                    if (_labels[i]) _labels[i].text = "<color=#555>Waiting for data…</color>";
                return;
            }

            Write(0, receiver.Latest_MP, age, warn, false);
            Write(1, receiver.Latest_MV, age, warn, false);
            Write(2, receiver.Latest_FU, age, warn, true);
            Write(3, receiver.Latest_GR, age, warn, false);
        }

        void Write(int i, UdpAngleReceiver.ArmAngles a,
                   float age, string warn, bool showConf)
        {
            if (!_labels[i]) return;
            _sb.Clear();

            string row(string label, float val) =>
                $"<color=#999>{label,-6}</color>  <b>{val,6:F1}°</b>\n";

            _sb.Append(row("Pitch",  a.shoulderPitch));
            _sb.Append(row("Roll",   a.shoulderRoll));
            _sb.Append(row("Yaw",    a.shoulderYaw));
            _sb.Append(row("Elbow",  a.elbowFlex));

            if (showConf)
            {
                float  c  = 1f / (1f + a.uncertainty);
                string cc = c > 0.85f ? "55ee77" : c > 0.6f ? "ffcc33" : "ff5555";
                _sb.Append($"<color=#999>{"Conf",-6}</color>  <color=#{cc}><b>{c:P0}</b></color>\n");
            }

            _sb.Append($"\n<color=#444>pkts {_totalPackets}   {age:F2}s{warn}</color>");
            _labels[i].text = _sb.ToString();
        }
    }
}
