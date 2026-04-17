## Unity setup (receive angles from Python)

This project streams **elbow / shoulder angles** from the Python vision app to Unity over **UDP**.

### 1) Create a Unity project

- Use Unity **2022 LTS** or newer.
- Create a 3D project.

### 2) Add the receiver script

Create `Assets/Scripts/UdpAngleReceiver.cs` with the code below.

```csharp
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

[Serializable]
public class AnglePacket {
    public double t;
    public float left_elbow_deg;
    public float right_elbow_deg;
    public float left_shoulder_elev_deg;
    public float right_shoulder_elev_deg;
}

public class UdpAngleReceiver : MonoBehaviour
{
    [Header("UDP")]
    public int port = 5005;

    [Header("Debug")]
    public bool logPackets = false;

    [Header("Target transforms (optional)")]
    public Transform leftForearm;
    public Transform rightForearm;

    private UdpClient _udp;
    private IPEndPoint _ep;
    private AnglePacket _latest;

    void Start()
    {
        _ep = new IPEndPoint(IPAddress.Any, port);
        _udp = new UdpClient(port);
        _udp.BeginReceive(OnReceive, null);
    }

    void OnDestroy()
    {
        try { _udp?.Close(); } catch { }
    }

    void OnReceive(IAsyncResult ar)
    {
        try
        {
            byte[] data = _udp.EndReceive(ar, ref _ep);
            string json = Encoding.UTF8.GetString(data);
            _latest = JsonUtility.FromJson<AnglePacket>(json);
            if (logPackets) Debug.Log(json);
        }
        catch { }
        finally
        {
            try { _udp.BeginReceive(OnReceive, null); } catch { }
        }
    }

    void Update()
    {
        if (_latest == null) return;

        // Minimal example: rotate forearms around local X based on elbow flexion.
        if (leftForearm != null)
            leftForearm.localRotation = Quaternion.Euler(_latest.left_elbow_deg, 0f, 0f);
        if (rightForearm != null)
            rightForearm.localRotation = Quaternion.Euler(_latest.right_elbow_deg, 0f, 0f);
    }
}
```

### 3) Add it to the scene

- Create an empty GameObject named `AngleReceiver`
- Add the `UdpAngleReceiver` component
- Set `port = 5005` (must match Python)
- (Optional) Assign `leftForearm/rightForearm` transforms.

### 4) Run the Python sender

From `PoseTrack/`:

```bash
python stream_to_unity.py --framework mediapipe --port 5005
```

Press `q` in the OpenCV window to quit.

