// PoseDebugUI — displays incoming joint angles as an on-screen overlay.
//
// RULES for the scene update thread:
//   - Debug.Log / Debug.LogWarning must NEVER appear in Update(), LateUpdate(),
//     or FixedUpdate(). Every Console.WriteLine triggers a full editor console
//     repaint which stalls the render thread.
//   - This component refreshes its UI text at a low fixed rate (_refreshHz).
//     The animation thread (ArmAngleController) is never touched from here.
//   - Use this component for any diagnostics; never add Debug.Log calls to
//     ArmAngleController or UdpAngleReceiver.
//
// Setup:
//   1. Create a UI Canvas (Screen Space – Overlay, render mode).
//   2. Add a child Text or TextMeshProUGUI object.
//   3. Attach this component to the Canvas (or any GameObject).
//   4. Drag the Text object into the _label field.
//   5. Drag the same GameObject that holds UdpAngleReceiver into _receiverObj.

using System.Text;
using UnityEngine;
using UnityEngine.UI;

#if TMP_PRESENT
using TMPro;
#endif

namespace PoseTrackReceiver
{
    public class PoseDebugUI : MonoBehaviour
    {
        [Header("References")]
        [Tooltip("GameObject that carries the UdpAngleReceiver component.")]
        public GameObject receiverObj;

        [Tooltip("UI Text component (legacy Text or leave null if using TMP below).")]
        public Text labelText;

#if TMP_PRESENT
        [Tooltip("TextMeshProUGUI component — preferred over legacy Text.")]
        public TextMeshProUGUI labelTmp;
#endif

        [Header("Display")]
        [Range(1f, 10f)]
        [Tooltip("How many times per second to refresh the UI text (default 4).")]
        public float refreshHz = 4f;

        [Tooltip("Show raw packet counter and data-age warning.")]
        public bool showDiagnostics = true;

        // ------------------------------------------------------------------ //

        UdpAngleReceiver _receiver;
        float _nextRefresh;
        int   _packetCount;
        float _lastDataTime;

        readonly StringBuilder _sb = new(256);

        void Awake()
        {
            if (receiverObj != null)
                _receiver = receiverObj.GetComponent<UdpAngleReceiver>();

            if (_receiver == null)
                _receiver = FindObjectOfType<UdpAngleReceiver>();
        }

        void Update()
        {
            // Count incoming packets without logging
            if (_receiver != null && _receiver.HasData)
            {
                _packetCount++;
                _lastDataTime = Time.unscaledTime;
            }

            // Throttle the string rebuild — cheap but avoids per-frame allocs
            if (Time.unscaledTime < _nextRefresh) return;
            _nextRefresh = Time.unscaledTime + 1f / Mathf.Max(refreshHz, 0.5f);

            RefreshLabel();
        }

        void RefreshLabel()
        {
            _sb.Clear();

            if (_receiver == null || !_receiver.HasData)
            {
                _sb.AppendLine("Waiting for pose data...");
            }
            else
            {
                var a = _receiver.Latest;
                _sb.AppendLine("Joint Angles (filtered)");
                _sb.AppendLine("─────────────────────");
                _sb.AppendFormat("Shoulder pitch : {0,7:F1}°\n", a.shoulderPitch);
                _sb.AppendFormat("Shoulder yaw   : {0,7:F1}°\n", a.shoulderYaw);
                _sb.AppendFormat("Shoulder roll  : {0,7:F1}°\n", a.shoulderRoll);
                _sb.AppendFormat("Elbow flex     : {0,7:F1}°\n", a.elbowFlex);

                if (showDiagnostics)
                {
                    float age = Time.unscaledTime - _lastDataTime;
                    _sb.AppendLine("─────────────────────");
                    _sb.AppendFormat("Packets recv : {0}\n", _packetCount);
                    _sb.AppendFormat("Data age     : {0:F2} s", age);
                    if (age > 0.5f)
                        _sb.Append("  ⚠ stale");
                }
            }

            string text = _sb.ToString();

#if TMP_PRESENT
            if (labelTmp != null) { labelTmp.text = text; return; }
#endif
            if (labelText != null) labelText.text = text;
        }
    }
}
