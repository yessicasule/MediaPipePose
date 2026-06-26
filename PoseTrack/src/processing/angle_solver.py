"""
angle_solver.py — Anatomically Consistent Arm Joint Angle Estimation
=====================================================================

Computes four joint angles from MediaPipe pseudo-3D landmarks using
an ISB-inspired hybrid coordinate frame approach:

    1. Shoulder Flexion-Extension  [degrees]
    2. Shoulder Abduction-Adduction [degrees]
    3. Shoulder Internal-External Rotation [degrees]  (forearm proxy)
    4. Elbow Flexion               [degrees]

Mathematical Framework
----------------------
All shoulder angles are decomposed using a ZXY Euler sequence applied to
the upper-arm orientation vector expressed in the torso coordinate frame.
This avoids gimbal lock at the anatomical rest position (arm hanging down),
which occurs under ISB's recommended YXY sequence.

Coordinate Frame Convention (see coordinate_frame.py):
    X — lateral (pointing to user's left)
    Y — superior (pointing up along spine)
    Z — anterior (pointing out of the chest toward the camera)

Sign Conventions:
    Flexion   (+) : arm forward in the sagittal plane
    Extension (−) : arm backward
    Abduction (+) : arm raised away from the midline (out to the side)
    Adduction (−) : arm crossing toward the midline
    Int. Rot. (+) : palm rotating inward (pronation direction)
    Ext. Rot. (−) : palm rotating outward (supination direction)
    Elbow Flex (+) : forearm bending toward the upper arm; 0° = fully extended

Shoulder Angle Derivation
--------------------------
Let v = [vx, vy, vz]^T be the unit upper-arm vector in the torso frame.

    Flexion-Extension:
        θ_flex = atan2(−vz, −vy)                 [rad]

        When the arm hangs down: vy ≈ −1, vz ≈ 0  →  θ_flex ≈ 0°
        When the arm points forward: vz ≈ +1        →  θ_flex ≈ +90°

    Abduction-Adduction:
        θ_abd = arcsin(vx / ‖v‖)                 [rad]

        When the arm hangs down: vx ≈ 0            →  θ_abd ≈ 0°
        When arm is fully abducted: vx ≈ +1        →  θ_abd ≈ +90°

    Internal-External Rotation (forearm proxy):
        Computed from the component of the forearm vector that is
        perpendicular to the upper-arm axis. Only reliable when the
        elbow is flexed ≥ ROTATION_RELIABLE_THRESHOLD degrees.

        Let u = unit upper-arm vector (torso frame).
        Let f = unit forearm vector (torso frame).
        Project f onto the plane perpendicular to u:
            f_perp = f − (f·u)u
        Build two reference vectors in that plane:
            e1 = unit(cross(u, Y))  where Y = [0,1,0]^T in torso frame
            e2 = unit(cross(u, e1))
        θ_rot = atan2(f_perp · e1,  f_perp · e2)

Elbow Flexion Derivation
--------------------------
    θ_elbow = 180° − arccos(û_arm · û_forearm)

    where û_arm is the unit vector from shoulder to elbow (direction of
    upper arm), and û_forearm is the unit vector from elbow to wrist.
    Result: 0° = fully extended, ~150° = fully flexed.

MediaPipe Right Arm Landmarks:
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW    = 14
    RIGHT_WRIST    = 16

References:
    Wu et al. (2005). ISB recommendation on definitions of joint coordinate
    systems. J Biomechanics 38(5):981-992.
    Biryukova et al. (2000). Kinematics of human arm reconstructed from
    spatial tracking. J Biomechanics 33(8):985-995.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass

from .coordinate_frame import TorsoFrame, build_torso_frame, world_to_torso, _to_array, _normalize

# ---------------------------------------------------------------------------
# MediaPipe landmark indices — RIGHT arm
# ---------------------------------------------------------------------------
LM_RIGHT_SHOULDER = 12
LM_RIGHT_ELBOW    = 14
LM_RIGHT_WRIST    = 16
LM_LEFT_SHOULDER  = 11

# Minimum elbow flexion (degrees) for rotation estimate to be reliable.
# Below this, the forearm is nearly collinear with the upper arm, making
# rotation unobservable from the forearm orientation.
ROTATION_RELIABLE_THRESHOLD = 25.0


@dataclass
class ArmAngles:
    """
    Complete set of estimated arm joint angles.

    Attributes
    ----------
    shoulder_flexion : float
        Shoulder flexion-extension in degrees.
        Positive = flexion (forward), negative = extension (backward).
    shoulder_abduction : float
        Shoulder abduction-adduction in degrees.
        Positive = abduction (away from body), negative = adduction.
    shoulder_rotation : float
        Shoulder internal-external rotation in degrees.
        Estimated via forearm proxy; reliable only when elbow_flexion >= 25°.
        Positive = internal rotation, negative = external rotation.
    elbow_flexion : float
        Elbow flexion in degrees. 0° = fully extended arm.
    rotation_reliable : bool
        True when elbow_flexion is sufficient for rotation estimation.
    """
    shoulder_flexion: float
    shoulder_abduction: float
    shoulder_rotation: float
    elbow_flexion: float
    rotation_reliable: bool


def _compute_shoulder_angles(
    v_upper_arm_torso: np.ndarray,
) -> tuple[float, float]:
    """
    Compute shoulder flexion and abduction from the upper-arm vector
    expressed in the torso coordinate frame, using ZXY Euler decomposition.

    Parameters
    ----------
    v_upper_arm_torso : np.ndarray, shape (3,)
        Unit vector from shoulder to elbow expressed in torso frame.
        Components: [x_lateral, y_superior, z_anterior].

    Returns
    -------
    flexion_deg : float
        Shoulder flexion-extension angle in degrees.
    abduction_deg : float
        Shoulder abduction-adduction angle in degrees.
    """
    vx = v_upper_arm_torso[0]  # lateral component
    vy = v_upper_arm_torso[1]  # superior component
    vz = v_upper_arm_torso[2]  # anterior component

    # Flexion-Extension
    # θ_flex = atan2(−vz, −vy)
    # Derivation: in the sagittal plane (Y-Z plane of torso frame),
    # the rest position (arm down) corresponds to vy = −1, vz = 0,
    # giving θ = atan2(0, +1) = 0°.  Arm forward: vy→0, vz→+1,
    # giving θ = atan2(−1, 0) = +90°.
    flexion_rad = math.atan2(-vz, -vy)
    flexion_deg = math.degrees(flexion_rad)

    # Abduction-Adduction
    # θ_abd = arcsin(−vx)  for the RIGHT arm.
    #
    # Derivation: the torso X-axis points to the user's LEFT
    # (constructed as left_shoulder − right_shoulder, see coordinate_frame.py).
    # Therefore, raising the RIGHT arm outward (to the right / away from midline)
    # moves the elbow in the NEGATIVE-X direction of the torso frame (vx < 0).
    # We negate to obtain the anatomical sign convention where abduction is +.
    vx_clamped   = float(np.clip(-vx, -1.0, 1.0))   # note negation for right arm
    abduction_deg = math.degrees(math.asin(vx_clamped))

    return flexion_deg, abduction_deg


def _compute_shoulder_rotation(
    v_upper_arm_torso: np.ndarray,
    v_forearm_torso: np.ndarray,
    elbow_flexion_deg: float,
) -> tuple[float, bool]:
    """
    Estimate shoulder internal-external rotation using the forearm as a proxy.

    The forearm vector, projected onto the plane perpendicular to the upper
    arm, encodes the axial rotation of the whole limb about the humeral axis.
    This is only geometrically observable when the elbow is flexed.

    Parameters
    ----------
    v_upper_arm_torso : np.ndarray, shape (3,)
        Unit upper-arm vector in torso frame.
    v_forearm_torso : np.ndarray, shape (3,)
        Unit forearm vector in torso frame (elbow → wrist direction).
    elbow_flexion_deg : float
        Current elbow flexion angle (degrees).

    Returns
    -------
    rotation_deg : float
        Estimated rotation angle in degrees; 0 when unreliable.
    reliable : bool
        Whether the estimate is geometrically reliable.
    """
    reliable = elbow_flexion_deg >= ROTATION_RELIABLE_THRESHOLD

    if not reliable:
        return 0.0, False

    u = _normalize(v_upper_arm_torso)

    # Project forearm onto the plane perpendicular to the upper arm
    # f_perp = f − (f·u)u
    f_perp = v_forearm_torso - np.dot(v_forearm_torso, u) * u
    f_perp_norm = np.linalg.norm(f_perp)

    if f_perp_norm < 1e-9:
        return 0.0, False

    f_perp = f_perp / f_perp_norm

    # Build an orthonormal basis (e1, e2) for the perpendicular plane.
    # e1 is perpendicular to u and as close as possible to the global
    # lateral axis [1, 0, 0] (or [0, 0, 1] if u is nearly lateral).
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, reference)) > 0.9:
        reference = np.array([0.0, 0.0, 1.0])

    e1 = np.cross(u, reference)
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-9:
        return 0.0, False
    e1 = e1 / e1_norm
    e2 = np.cross(u, e1)

    # atan2 gives signed rotation angle in the perpendicular plane
    rotation_deg = math.degrees(math.atan2(
        float(np.dot(f_perp, e1)),
        float(np.dot(f_perp, e2)),
    ))

    return rotation_deg, True


def _compute_elbow_flexion(
    v_upper_arm_world: np.ndarray,
    v_forearm_world: np.ndarray,
) -> float:
    """
    Compute elbow flexion angle from upper-arm and forearm direction vectors.

    θ_elbow = arccos(−û_upper · û_forearm)

    where û_upper = shoulder→elbow  and  û_forearm = elbow→wrist.

    Geometry: when the arm is fully extended, û_upper and û_forearm are
    parallel and co-directional, so their dot product = +1 and the angle
    between them = 0°. To obtain the *included* angle at the elbow joint
    (the supplementary angle), we negate û_upper before taking the dot product:

        θ_elbow = arccos(−û_upper · û_forearm)

    This gives 0° for a straight arm and ~150° for a maximally flexed elbow.

    Parameters
    ----------
    v_upper_arm_world : np.ndarray, shape (3,)
        Vector from shoulder to elbow (direction of upper arm).
    v_forearm_world : np.ndarray, shape (3,)
        Vector from elbow to wrist (direction of forearm).

    Returns
    -------
    float
        Elbow flexion in degrees. 0° = fully extended.
    """
    u_arm    = _normalize(v_upper_arm_world)
    u_fore   = _normalize(v_forearm_world)
    # The included angle at the elbow = 180° − the angle between the two
    # bone direction vectors (both measured from proximal to distal end).
    # When the arm is fully extended, u_arm and u_fore are co-directional
    # (dot = +1, angle = 0°) so elbow flexion = 180° − 0° ... wait, that is wrong.
    #
    # Correct geometry:
    #   - u_arm  = shoulder → elbow   (upper arm direction, proximal to distal)
    #   - u_fore = elbow   → wrist    (forearm direction, proximal to distal)
    # Fully extended: u_arm and u_fore are PARALLEL (+same direction).
    #   dot(u_arm, u_fore) = +1   →   angle_between = 0°
    #   elbow_flexion = 0°  ← CORRECT: arm straight = 0° flexion.
    # At 90° bend: u_arm and u_fore are perpendicular.
    #   dot(u_arm, u_fore) = 0   →   angle_between = 90°  ← CORRECT.
    # Fully flexed (arm folded back): u_arm and u_fore are anti-parallel.
    #   dot(u_arm, u_fore) = -1  →   angle_between = 180°  ← max flexion.
    #
    # Therefore: elbow_flexion = arccos(u_arm · u_fore) directly.
    cos_ang = float(np.clip(np.dot(u_arm, u_fore), -1.0, 1.0))
    return max(0.0, math.degrees(math.acos(cos_ang)))


def compute_arm_angles(landmarks) -> ArmAngles | None:
    """
    Compute all four arm joint angles from a MediaPipe landmark list.

    This is the main entry point for the angle computation pipeline.

    Parameters
    ----------
    landmarks : list
        MediaPipe 33-landmark list with pseudo-3D (.x, .y, .z) coordinates.

    Returns
    -------
    ArmAngles or None
        Returns None if required landmarks are missing or the torso frame
        cannot be constructed (e.g., full-body occlusion).

    Pipeline
    --------
    1. Build torso coordinate frame from hip/shoulder landmarks.
    2. Extract right-arm segment vectors (shoulder→elbow, elbow→wrist).
    3. Transform arm vectors into the torso frame.
    4. Decompose shoulder orientation using ZXY Euler angles.
    5. Estimate shoulder rotation from forearm proxy.
    6. Compute elbow flexion from segment dot product.
    """
    # Step 1: Build torso frame
    frame = build_torso_frame(landmarks)
    if frame is None:
        return None

    # Step 2: Extract world-space arm segment vectors
    try:
        p_shoulder = _to_array(landmarks[LM_RIGHT_SHOULDER])
        p_elbow    = _to_array(landmarks[LM_RIGHT_ELBOW])
        p_wrist    = _to_array(landmarks[LM_RIGHT_WRIST])
    except (IndexError, TypeError):
        return None

    v_upper_arm_world = p_elbow - p_shoulder   # shoulder → elbow
    v_forearm_world   = p_wrist - p_elbow       # elbow → wrist

    if np.linalg.norm(v_upper_arm_world) < 1e-9:
        return None

    # Step 3: Transform to torso frame
    v_upper_arm_torso = world_to_torso(v_upper_arm_world, frame)
    v_forearm_torso   = world_to_torso(v_forearm_world,   frame)

    # Normalise (needed for arcsin / atan2)
    v_upper_arm_unit  = _normalize(v_upper_arm_torso)

    # Step 4: Shoulder flexion and abduction (ZXY decomposition)
    flexion, abduction = _compute_shoulder_angles(v_upper_arm_unit)

    # Step 5: Elbow flexion
    elbow = _compute_elbow_flexion(v_upper_arm_world, v_forearm_world)

    # Step 6: Shoulder rotation (forearm proxy)
    rotation, rot_reliable = _compute_shoulder_rotation(
        v_upper_arm_unit,
        _normalize(v_forearm_torso) if np.linalg.norm(v_forearm_torso) > 1e-9
        else v_forearm_torso,
        elbow,
    )

    return ArmAngles(
        shoulder_flexion=flexion,
        shoulder_abduction=abduction,
        shoulder_rotation=rotation,
        elbow_flexion=elbow,
        rotation_reliable=rot_reliable,
    )
