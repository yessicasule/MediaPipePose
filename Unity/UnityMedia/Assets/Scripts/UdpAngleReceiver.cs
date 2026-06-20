// UdpAngleReceiver.cs — Phase 8 update: receives 4-stream multi-avatar packets.
//
// Packet prefixes:
//   MP,<pitch>,<roll>,<yaw>,<elbow>
//   MV,<pitch>,<roll>,<yaw>,<elbow>
//   FU,<pitch>,<roll>,<yaw>,<elbow>,<uncertainty>
//   GR,<pitch>,<roll>,<yaw>,<elbow>
//
// Legacy S, prefix still supported for backward compatibility.

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace PoseTrackReceiver
{
    public class UdpAngleReceiver : MonoBehaviour
    {
        [Header("Network")]
        public int listenPort = 9000;

        public struct ArmAngles
        {
            public float shoulderPitch;
            public float shoulderYaw;
            public float shoulderRoll;
            public float elbowFlex;
            public float uncertainty;   // only populated for FU prefix
        }

        // Latest data per source
        public ArmAngles Latest_MP { get; private set; }
        public ArmAngles Latest_MV { get; private set; }
        public ArmAngles Latest_FU { get; private set; }
        public ArmAngles Latest_GR { get; private set; }
        public ArmAngles Latest    { get; private set; }  // legacy

        public bool HasData { get; private set; }

        UdpClient     _client;
        Thread        _thread;
        volatile bool _running;

        readonly object _lock = new();
        ArmAngles _pendingMP, _pendingMV, _pendingFU, _pendingGR, _pendingLegacy;
        bool      _readyMP,   _readyMV,   _readyFU,   _readyGR,   _readyLegacy;

        // Only one instance may bind the port at a time
        static UdpAngleReceiver _instance;

        void Awake()
        {
            if (_instance != null && _instance != this)
            {
                Debug.LogWarning($"[UdpAngleReceiver] Duplicate on '{gameObject.name}' destroyed. " +
                                 $"Only one receiver can bind port {listenPort}.");
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

        // Socket is managed by Awake/OnDestroy — not by enable/disable
        void OnEnable()  { }
        void OnDisable() { }

        void StartReceiver()
        {
            try
            {
                // ReuseAddress releases the port immediately on Stop,
                // preventing "address in use" on rapid Play/Stop cycles in the Editor.
                var sock = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                sock.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
                sock.Bind(new IPEndPoint(IPAddress.Any, listenPort));
                _client  = new UdpClient { Client = sock };
                _running = true;
                _thread  = new Thread(Receive) { IsBackground = true };
                _thread.Start();
                Debug.Log($"[UdpAngleReceiver] Listening on UDP port {listenPort}");
            }
            catch (SocketException ex)
            {
                Debug.LogError($"[UdpAngleReceiver] Cannot bind port {listenPort}: {ex.Message}\n" +
                               "Ensure no other GameObject has UdpAngleReceiver and no other " +
                               "app is using this port.");
            }
        }

        void StopReceiver()
        {
            _running = false;
            _client?.Close();
            _client = null;
            _thread?.Join(500);
            _thread = null;
        }

        void Update()
        {
            lock (_lock)
            {
                if (_readyMP)    { Latest_MP = _pendingMP;       _readyMP    = false; HasData = true; }
                if (_readyMV)    { Latest_MV = _pendingMV;       _readyMV    = false; HasData = true; }
                if (_readyFU)    { Latest_FU = _pendingFU;       _readyFU    = false; HasData = true; }
                if (_readyGR)    { Latest_GR = _pendingGR;       _readyGR    = false; HasData = true; }
                if (_readyLegacy){ Latest     = _pendingLegacy;  _readyLegacy= false; HasData = true; }
            }
        }

        void Receive()
        {
            var ep = new IPEndPoint(IPAddress.Any, listenPort);
            while (_running)
            {
                try
                {
                    byte[] data = _client.Receive(ref ep);
                    string line = Encoding.UTF8.GetString(data).Trim();
                    if (string.IsNullOrEmpty(line)) continue;

                    string prefix = line.Length >= 2 ? line.Substring(0, 2) : "";
                    string body   = line.Length >  3 ? line.Substring(3)    : "";

                    lock (_lock)
                    {
                        switch (prefix)
                        {
                            case "MP": if (TryParse4(body, out ArmAngles mp)) { _pendingMP = mp; _readyMP = true; } break;
                            case "MV": if (TryParse4(body, out ArmAngles mv)) { _pendingMV = mv; _readyMV = true; } break;
                            case "FU": if (TryParse5(body, out ArmAngles fu)) { _pendingFU = fu; _readyFU = true; } break;
                            case "GR": if (TryParse4(body, out ArmAngles gr)) { _pendingGR = gr; _readyGR = true; } break;
                            default:
                                if (line.StartsWith("S,") && TryParse4(line.Substring(2), out ArmAngles leg))
                                { _pendingLegacy = leg; _readyLegacy = true; }
                                break;
                        }
                    }
                }
                catch (SocketException) { }
                catch (ObjectDisposedException) { break; }
            }
        }

        static bool TryParse4(string body, out ArmAngles a)
        {
            a = default;
            var p = body.Split(',');
            if (p.Length < 4) return false;
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var fl  = System.Globalization.NumberStyles.Float;
            if (!float.TryParse(p[0], fl, inv, out float v0)) return false;
            if (!float.TryParse(p[1], fl, inv, out float v1)) return false;
            if (!float.TryParse(p[2], fl, inv, out float v2)) return false;
            if (!float.TryParse(p[3], fl, inv, out float v3)) return false;
            a = new ArmAngles { shoulderPitch = v0, shoulderRoll = v1, shoulderYaw = v2, elbowFlex = v3 };
            return true;
        }

        static bool TryParse5(string body, out ArmAngles a)
        {
            if (!TryParse4(body, out a)) return false;
            var p = body.Split(',');
            if (p.Length >= 5 && float.TryParse(p[4], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float unc))
                a.uncertainty = unc;
            return true;
        }
    }
}
