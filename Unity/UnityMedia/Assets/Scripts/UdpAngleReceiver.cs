// UdpAngleReceiver.cs
// ===================
// Receives arm joint angle packets from the MonoArm Python pipeline over UDP.
//
// Packet Format
// -------------
//   S,<shoulder_flex>,<shoulder_abd>,<shoulder_rot>,<elbow_flex>\n
//
//   All values are in degrees. The 'S' prefix denotes a single-arm pose.
//
// Threading Model
// ---------------
//   A background thread calls UdpClient.Receive() (blocking). When a packet
//   arrives it is parsed into a pending ArmAngles struct under a lock.
//   On the Unity main thread (Update()), the pending struct is copied into
//   the public LatestAngles property under the same lock. This two-stage
//   approach keeps the main thread non-blocking and avoids race conditions.
//
// Socket Options
// --------------
//   ReuseAddress is set so the OS releases the port immediately when the
//   receiver is destroyed, preventing "address already in use" errors
//   during rapid Enter/Exit Play mode cycles in the Unity Editor.

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace MonoArm
{
    /// <summary>
    /// Anatomically labelled arm joint angles received from the Python pipeline.
    /// All values are in degrees.
    /// </summary>
    public struct ArmAngles
    {
        /// <summary>Shoulder flexion (+) / extension (−) in degrees.</summary>
        public float shoulderFlexion;

        /// <summary>Shoulder abduction (+) / adduction (−) in degrees.</summary>
        public float shoulderAbduction;

        /// <summary>
        /// Shoulder internal (+) / external (−) rotation in degrees.
        /// Reliable only when elbow is flexed ≥ 25°.
        /// </summary>
        public float shoulderRotation;

        /// <summary>Elbow flexion in degrees. 0° = fully extended.</summary>
        public float elbowFlexion;

        public override string ToString() =>
            $"Flex:{shoulderFlexion:F1}° Abd:{shoulderAbduction:F1}° " +
            $"Rot:{shoulderRotation:F1}° Elb:{elbowFlexion:F1}°";
    }

    /// <summary>
    /// Threaded UDP receiver that delivers ArmAngles to the Unity main thread.
    /// Attach to any persistent GameObject (e.g. a PoseManager empty object).
    /// Only one instance may bind the configured port at a time.
    /// </summary>
    public class UdpAngleReceiver : MonoBehaviour
    {
        [Header("Network")]
        [Tooltip("UDP port to listen on. Must match Python stream_hz port (default 9000).")]
        public int listenPort = 9000;

        // ── Public state ────────────────────────────────────────────────────
        /// <summary>Latest validated angles, updated each Unity frame.</summary>
        public ArmAngles LatestAngles { get; private set; }

        /// <summary>True once the first valid packet has been received.</summary>
        public bool HasData { get; private set; }

        /// <summary>Total number of valid packets received in this session.</summary>
        public int PacketCount { get; private set; }

        /// <summary>Elapsed seconds since the last valid packet arrived.</summary>
        public float TimeSinceLastPacket { get; private set; }

        // ── Private state ───────────────────────────────────────────────────
        UdpClient     _client;
        Thread        _thread;
        volatile bool _running;

        readonly object _lock = new();
        ArmAngles _pending;
        bool      _pendingReady;
        float     _lastPacketTime;

        // Singleton guard — one receiver per scene
        static UdpAngleReceiver _instance;

        // ── Unity lifecycle ─────────────────────────────────────────────────

        void Awake()
        {
            if (_instance != null && _instance != this)
            {
                Debug.LogWarning(
                    $"[UdpAngleReceiver] Duplicate on '{gameObject.name}' destroyed. " +
                    $"Only one receiver may bind port {listenPort}.");
                Destroy(this);
                return;
            }
            _instance = this;
            StartReceiver();
        }

        void OnDestroy()
        {
            StopReceiver();
            if (_instance == this) _instance = null;
        }

        void Update()
        {
            TimeSinceLastPacket = Time.unscaledTime - _lastPacketTime;

            lock (_lock)
            {
                if (!_pendingReady) return;
                LatestAngles   = _pending;
                _pendingReady  = false;
                HasData        = true;
                PacketCount++;
                _lastPacketTime = Time.unscaledTime;
            }
        }

        // ── Socket management ───────────────────────────────────────────────

        void StartReceiver()
        {
            try
            {
                var sock = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                sock.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
                sock.Bind(new IPEndPoint(IPAddress.Any, listenPort));
                _client  = new UdpClient { Client = sock };
                _running = true;
                _thread  = new Thread(ReceiveLoop) { IsBackground = true, Name = "UdpReceiver" };
                _thread.Start();
                Debug.Log($"[UdpAngleReceiver] Listening on UDP port {listenPort}");
            }
            catch (SocketException ex)
            {
                Debug.LogError(
                    $"[UdpAngleReceiver] Cannot bind port {listenPort}: {ex.Message}\n" +
                    "Ensure no other script has UdpAngleReceiver and the port is not in use.");
            }
        }

        void StopReceiver()
        {
            _running = false;
            try { _client?.Close(); } catch { }
            _client = null;
            _thread?.Join(500);
            _thread = null;
        }

        // ── Background receive loop ─────────────────────────────────────────

        void ReceiveLoop()
        {
            var ep = new IPEndPoint(IPAddress.Any, listenPort);
            while (_running)
            {
                try
                {
                    byte[] data = _client.Receive(ref ep);
                    string line = Encoding.UTF8.GetString(data).Trim();

                    if (string.IsNullOrEmpty(line)) continue;
                    if (!line.StartsWith("S,"))       continue;

                    // Parse: S,flex,abd,rot,elbow
                    if (!TryParsePacket(line.Substring(2), out ArmAngles angles)) continue;

                    lock (_lock)
                    {
                        _pending      = angles;
                        _pendingReady = true;
                    }
                }
                catch (SocketException)  { /* socket closed — exit loop */ break; }
                catch (ObjectDisposedException) { break; }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[UdpAngleReceiver] Parse error: {ex.Message}");
                }
            }
        }

        // ── Packet parser ───────────────────────────────────────────────────

        static bool TryParsePacket(string body, out ArmAngles a)
        {
            a = default;
            var  parts = body.Split(',');
            if   (parts.Length < 4) return false;

            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var fl  = System.Globalization.NumberStyles.Float;

            if (!float.TryParse(parts[0], fl, inv, out float flex))  return false;
            if (!float.TryParse(parts[1], fl, inv, out float abd))   return false;
            if (!float.TryParse(parts[2], fl, inv, out float rot))   return false;
            if (!float.TryParse(parts[3], fl, inv, out float elbow)) return false;

            a = new ArmAngles
            {
                shoulderFlexion   = flex,
                shoulderAbduction = abd,
                shoulderRotation  = rot,
                elbowFlexion      = elbow,
            };
            return true;
        }
    }
}
