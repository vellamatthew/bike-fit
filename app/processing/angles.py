import numpy as np

# YOLO11 / COCO keypoint indices
KP = {
    "nose": 0,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}

# Minimum pixel distance from origin to treat a keypoint as detected
MIN_COORD = 5.0
# Minimum visibility score to consider a keypoint valid (0.0 - 1.0)
MIN_VISIBILITY = 0.5

# Keypoints to check for side visibility
SIDE_KEYPOINTS = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]


def _angle(a, b, c) -> float:
    """Angle in degrees at joint b, formed by the a-b-c triplet."""
    ba = np.array(a[:2], dtype=float) - np.array(b[:2], dtype=float)
    bc = np.array(c[:2], dtype=float) - np.array(b[:2], dtype=float)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-6:
        return 0.0
    cos_a = np.dot(ba, bc) / norm
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _kp(kps: np.ndarray, name: str):
    return kps[KP[name]]


def _detected(pt) -> bool:
    """Check if keypoint is detected based on coordinates and visibility (if available)."""
    if len(pt) < 2:
        return False
    # Check coordinate validity
    if float(pt[0]) <= MIN_COORD or float(pt[1]) <= MIN_COORD:
        return False
    # Check visibility if available (3rd dimension)
    if len(pt) >= 3 and float(pt[2]) < MIN_VISIBILITY:
        return False
    return True


def detect_side(kps: np.ndarray) -> str:
    """
    Detect which side of the body is more visible based on visibility scores.

    Args:
        kps: np.ndarray of shape (17, 3) with [x, y, visibility]

    Returns:
        "left", "right", or None if neither side is sufficiently visible
    """
    # If no visibility data, default to right
    if kps.shape[1] < 3:
        return "right"

    left_visibility = []
    right_visibility = []

    for kp_name in SIDE_KEYPOINTS:
        left_kp = _kp(kps, f"left_{kp_name}")
        right_kp = _kp(kps, f"right_{kp_name}")

        # Collect visibility scores
        if len(left_kp) >= 3:
            left_visibility.append(float(left_kp[2]))
        if len(right_kp) >= 3:
            right_visibility.append(float(right_kp[2]))

    # Calculate average visibility for each side
    avg_left = np.mean(left_visibility) if left_visibility else 0.0
    avg_right = np.mean(right_visibility) if right_visibility else 0.0

    # Need at least some visibility to make a determination
    if avg_left < MIN_VISIBILITY and avg_right < MIN_VISIBILITY:
        return None

    # Return the side with higher average visibility
    return "left" if avg_left > avg_right else "right"


def compute_angles_for_side(kps: np.ndarray, side: str) -> dict:
    """
    Compute cycling-relevant joint angles for a specific side.

    Args:
        kps: np.ndarray of shape (17, 2) with [x, y] coordinates,
             or (17, 3) with [x, y, visibility] if available.
        side: "left" or "right" - which side to compute angles for

    Returns:
        dict with knee_flexion, knee_extension, hip_flexion, elbow_flexion, back_angle (floats in degrees or None)
    """
    if side not in ["left", "right"]:
        return {
            "knee_flexion": None,
            "knee_extension": None,
            "hip_flexion": None,
            "elbow_flexion": None,
            "back_angle": None
        }

    angles = {}

    # Knee flexion angle (hip-knee-ankle)
    hip, knee, ankle = _kp(kps, f"{side}_hip"), _kp(kps, f"{side}_knee"), _kp(kps, f"{side}_ankle")
    if _detected(hip) and _detected(knee) and _detected(ankle):
        knee_flexion_angle = _angle(hip, knee, ankle)
        angles["knee_flexion"] = knee_flexion_angle
        # Knee extension is the complement of knee flexion (180° - flexion)
        # At BDC, target knee extension is 35-40°, meaning knee flexion is 140-145°
        angles["knee_extension"] = 180.0 - knee_flexion_angle
    else:
        angles["knee_flexion"] = None
        angles["knee_extension"] = None

    # Hip flexion angle (shoulder-hip-knee)
    shoulder, hip, knee = _kp(kps, f"{side}_shoulder"), _kp(kps, f"{side}_hip"), _kp(kps, f"{side}_knee")
    if _detected(shoulder) and _detected(hip) and _detected(knee):
        angles["hip_flexion"] = _angle(shoulder, hip, knee)
    else:
        angles["hip_flexion"] = None

    # Elbow flexion angle (shoulder-elbow-wrist)
    shoulder, elbow, wrist = _kp(kps, f"{side}_shoulder"), _kp(kps, f"{side}_elbow"), _kp(kps, f"{side}_wrist")
    if _detected(shoulder) and _detected(elbow) and _detected(wrist):
        angles["elbow_flexion"] = _angle(shoulder, elbow, wrist)
    else:
        angles["elbow_flexion"] = None

    # Back angle (torso angle from horizontal)
    hip, shoulder = _kp(kps, f"{side}_hip"), _kp(kps, f"{side}_shoulder")
    if _detected(hip) and _detected(shoulder):
        # Angle of torso relative to horizontal
        vec = np.array(shoulder[:2], dtype=float) - np.array(hip[:2], dtype=float)
        angles["back_angle"] = float(np.degrees(np.arctan2(abs(vec[1]), abs(vec[0]))))
    else:
        angles["back_angle"] = None

    return angles


def compute_angles(kps: np.ndarray) -> dict:
    """
    Compute cycling-relevant joint angles from keypoints array.

    Args:
        kps: np.ndarray of shape (17, 2) with [x, y] coordinates,
             or (17, 3) with [x, y, visibility] if available.

    Returns:
        dict with keys:
            - angles: dict with knee_flexion, hip_flexion, elbow_flexion, back_angle (floats in degrees or None)
            - side: "left", "right", or None indicating detected visible side
    """
    # Detect which side is more visible
    side = detect_side(kps)

    if side is None:
        # No side visible, return all None
        return {
            "angles": {
                "knee_flexion": None,
                "knee_extension": None,
                "hip_flexion": None,
                "elbow_flexion": None,
                "back_angle": None
            },
            "side": None
        }

    # Compute angles for detected side
    angles = compute_angles_for_side(kps, side)
    return {"angles": angles, "side": side}
