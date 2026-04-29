using UnityEngine;

namespace PoseTrackReceiver
{
    public class AngleSmoother
    {
        readonly float _alpha;
        float _value;
        bool  _init;

        public AngleSmoother(float smoothing = 0.15f) => _alpha = Mathf.Clamp01(smoothing);

        public float Update(float raw)
        {
            if (!_init) { _value = raw; _init = true; return raw; }
            _value = Mathf.LerpAngle(_value, raw, _alpha);
            return _value;
        }

        public void Reset() => _init = false;
    }
}
