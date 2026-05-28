import cv2
import numpy as np

# COCO skeleton connections (pairs of keypoint indices)
# Left side bones
SKELETON_LEFT = [
    (5, 7), (7, 9),    # left arm
    (5, 11),           # left torso
    (11, 13), (13, 15),# left leg
]

# Right side bones
SKELETON_RIGHT = [
    (6, 8), (8, 10),   # right arm
    (6, 12),           # right torso
    (12, 14), (14, 16),# right leg
]

# Full skeleton (when side is None or for backward compatibility)
SKELETON = [
    (5, 6),            # shoulders
    (11, 12),          # hips
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (5, 11), (6, 12),  # torso sides
    (11, 13), (13, 15),# left leg
    (12, 14), (14, 16),# right leg
]

JOINT_COLOR  = (0, 255, 120)    # green joints
BONE_COLOR   = (255, 200, 0)    # yellow bones
ANGLE_COLOR  = (255, 255, 255)  # white text


def draw_skeleton(frame: np.ndarray, kps: np.ndarray, angles: dict = None, side: str = None) -> np.ndarray:
    """
    Draw keypoints and skeleton on frame.

    Args:
        frame: Input image (BGR)
        kps: (17, 2) array of pixel coordinates
        angles: Optional dict with angle values (not used anymore, kept for compatibility)
        side: "left", "right", or None. If specified, only draws that side + center bones.

    Returns:
        Annotated copy of frame
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Determine which skeleton to draw
    if side == "left":
        skeleton_to_draw = SKELETON_LEFT
    elif side == "right":
        skeleton_to_draw = SKELETON_RIGHT
    else:
        # Draw full skeleton if side not specified
        skeleton_to_draw = SKELETON

    # Draw bones
    for i, j in skeleton_to_draw:
        p1, p2 = kps[i], kps[j]
        if p1[0] > 5 and p1[1] > 5 and p2[0] > 5 and p2[1] > 5:
            cv2.line(out,
                     (int(p1[0]), int(p1[1])),
                     (int(p2[0]), int(p2[1])),
                     BONE_COLOR, 4, cv2.LINE_AA)

    # Draw joints (only for the selected side)
    if side == "left":
        # Left side only: left shoulder, left hip, and left arm/leg joints
        keypoint_indices = [5, 11, 7, 9, 13, 15]
    elif side == "right":
        # Right side only: right shoulder, right hip, and right arm/leg joints
        keypoint_indices = [6, 12, 8, 10, 14, 16]
    else:
        # Draw all keypoints
        keypoint_indices = range(len(kps))

    for idx in keypoint_indices:
        pt = kps[idx]
        if pt[0] > 5 and pt[1] > 5:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 8, JOINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(out, (int(pt[0]), int(pt[1])), 8, (0, 0, 0), 2, cv2.LINE_AA)

    return out
