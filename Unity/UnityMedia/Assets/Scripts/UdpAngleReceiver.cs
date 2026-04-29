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
        }

        public ArmAngles Latest { get; private set; }
        public bool HasData    { get; private set; }

        UdpClient    _client;
        Thread       _thread;
        volatile bool _running;
        readonly object _lock = new();
        ArmAngles    _pending;
        bool         _pendingReady;

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
                if (!_pendingReady) return;
                Latest       = _pending;
                HasData      = true;
                _pendingReady = false;
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
                    if (TryParse(line, out ArmAngles a))
                    {
                        lock (_lock) { _pending = a; _pendingReady = true; }
                    }
                }
                catch (SocketException) { }
                catch (ObjectDisposedException) { break; }
            }
        }

        static bool TryParse(string line, out ArmAngles a)
        {
            a = default;
            if (!line.StartsWith("S,")) return false;
            string[] parts = line.Split(',');
            if (parts.Length < 5) return false;
            if (!float.TryParse(parts[1], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float sp)) return false;
            if (!float.TryParse(parts[2], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float sy)) return false;
            if (!float.TryParse(parts[3], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float sr)) return false;
            if (!float.TryParse(parts[4], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float ef)) return false;
            a = new ArmAngles { shoulderPitch = sp, shoulderYaw = sy, shoulderRoll = sr, elbowFlex = ef };
            return true;
        }
    }
}
