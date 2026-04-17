import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float = 0.0

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        return math.sqrt(self.dot(self))


def angle_deg(a: Vec3, b: Vec3, c: Vec3) -> float:
    """
    Angle ABC in degrees using vectors BA and BC.
    Returns NaN if vectors are degenerate.
    """
    ba = a - b
    bc = c - b
    nba = ba.norm()
    nbc = bc.norm()
    if nba == 0 or nbc == 0:
        return float("nan")
    cosv = max(-1.0, min(1.0, ba.dot(bc) / (nba * nbc)))
    return math.degrees(math.acos(cosv))


def elbow_flexion_deg(shoulder: Vec3, elbow: Vec3, wrist: Vec3) -> float:
    # 180 when straight, smaller when flexed.
    return angle_deg(shoulder, elbow, wrist)


def shoulder_elevation_deg(hip: Vec3, shoulder: Vec3, elbow: Vec3) -> float:
    """
    Simplified shoulder elevation: angle between torso vector (shoulder-hip) and upper-arm vector (elbow-shoulder).
    0 means arm aligned with torso downward/upward depending on coordinate system; treat as relative measure.
    """
    torso = hip - shoulder
    upper = elbow - shoulder
    nba = torso.norm()
    nbc = upper.norm()
    if nba == 0 or nbc == 0:
        return float("nan")
    cosv = max(-1.0, min(1.0, torso.dot(upper) / (nba * nbc)))
    return math.degrees(math.acos(cosv))

