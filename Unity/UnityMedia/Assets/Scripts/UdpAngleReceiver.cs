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

        UdpClient    _client;
        Thread       _thread;
        volatile bool _running;

        readonly object  _lock = new();
        ArmAngles _pendingMP, _pendingMV, _pendingFU, _pendingGR, _pendingLegacy;
        bool      _readyMP,   _readyMV,   _readyFU,   _readyGR,   _readyLegacy;

        void OnEnable()
        {
            _client  = new UdpClient(listenPort);
            _running = true;
            _thread  = new Thread(Receive) { IsBackground = true };
            _thread.Start();
        }

        void OnDisable()
        {
            _running = false;
            _client?.Close();
            _thread?.Join(500);
        }

        void Update()
        {
            lock (_lock)
            {
                if (_readyMP)    { Latest_MP  = _pendingMP;  _readyMP    = false; HasData = true; }
                if (_readyMV)    { Latest_MV  = _pendingMV;  _readyMV    = false; HasData = true; }
                if (_readyFU)    { Latest_FU  = _pendingFU;  _readyFU    = false; HasData = true; }
                if (_readyGR)    { Latest_GR  = _pendingGR;  _readyGR    = false; HasData = true; }
                if (_readyLegacy){ Latest      = _pendingLegacy; _readyLegacy = false; HasData = true; }
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
                    string body   = line.Length > 3   ? line.Substring(3)   : "";

                    lock (_lock)
                    {
                        switch (prefix)
                        {
                            case "MP":
                                if (TryParse4(body, out ArmAngles mp))  { _pendingMP  = mp; _readyMP  = true; }
                                break;
                            case "MV":
                                if (TryParse4(body, out ArmAngles mv))  { _pendingMV  = mv; _readyMV  = true; }
                                break;
                            case "FU":
                                if (TryParse5(body, out ArmAngles fu))  { _pendingFU  = fu; _readyFU  = true; }
                                break;
                            case "GR":
                                if (TryParse4(body, out ArmAngles gr))  { _pendingGR  = gr; _readyGR  = true; }
                                break;
                            default:
                                // Legacy "S," format
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
            string[] p = body.Split(',');
            if (p.Length < 4) return false;
            if (!float.TryParse(p[0], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float v0)) return false;
            if (!float.TryParse(p[1], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float v1)) return false;
            if (!float.TryParse(p[2], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float v2)) return false;
            if (!float.TryParse(p[3], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float v3)) return false;
            a = new ArmAngles { shoulderPitch = v0, shoulderRoll = v1, shoulderYaw = v2, elbowFlex = v3 };
            return true;
        }

        static bool TryParse5(string body, out ArmAngles a)
        {
            a = default;
            if (!TryParse4(body, out a)) return false;
            string[] p = body.Split(',');
            if (p.Length >= 5 && float.TryParse(p[4], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float unc))
                a.uncertainty = unc;
            return true;
        }
    }
}
