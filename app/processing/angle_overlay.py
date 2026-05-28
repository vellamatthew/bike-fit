"""
Draw angle measurements and annotations on skeleton frames for bike fitting visualization.
"""
import cv2
import numpy as np
from processing.annotate import draw_skeleton


# Color scheme based on assessment status
COLORS = {
    'good': (0, 200, 0),      # Green
    'marginal': (0, 165, 255),  # Orange
    'poor': (0, 0, 255),        # Red
    'unknown': (180, 180, 180)  # Gray
}


def annotate_frame_with_angles(
    frame: np.ndarray,
    keypoints: np.ndarray,
    angles: dict,
    assessments: dict = None,
    side: str = None,
    use_assessment_values: bool = True
) -> np.ndarray:
    """
    Draw skeleton and angle overlays with color coding based on fit assessment.

    Args:
        frame: BGR image
        keypoints: (17, 2) array of pixel coordinates
        angles: Dict with angle values (e.g., {'knee_flexion': 110, 'hip_flexion': 58, ...})
        assessments: Optional dict from fit_assessment (for color coding)
        side: "left", "right", or None
        use_assessment_values: When True, prefer assessed summary values for labels.
            When False, use this frame's raw angle values while still using assessments
            only for color coding.

    Returns:
        Annotated frame with skeleton + angle labels
    """
    # Draw skeleton first
    annotated = draw_skeleton(frame, keypoints, angles, side)

    # Add angle annotations
    annotated = _draw_angle_labels(
        annotated, keypoints, angles, assessments, side, use_assessment_values
    )

    return annotated


def _draw_angle_labels(
    frame: np.ndarray,
    keypoints: np.ndarray,
    angles: dict,
    assessments: dict,
    side: str,
    use_assessment_values: bool
) -> np.ndarray:
    """Draw angle values as text labels near joints."""
    out = frame.copy()

    # Determine which side to annotate
    if side == "left":
        knee_idx, hip_idx, elbow_idx = 13, 11, 7
        shoulder_idx = 5
    elif side == "right":
        knee_idx, hip_idx, elbow_idx = 14, 12, 8
        shoulder_idx = 6
    else:
        # Auto: use right as default
        knee_idx, hip_idx, elbow_idx = 14, 12, 8
        shoulder_idx = 6

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    bg_thickness = 6

    overlay_metrics = _build_overlay_metrics(angles, assessments, use_assessment_values)

    # Knee extension label
    knee_pos = keypoints[knee_idx].astype(int)
    if knee_pos[0] > 5 and knee_pos[1] > 5:
        knee_ext = overlay_metrics.get('knee_extension_bdc')

        if knee_ext is not None:
            # Determine color based on assessment
            color = _get_color_for_angle('knee_extension_bdc', assessments)
            label = f"{knee_ext:.0f}"

            # Draw with background for readability
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = knee_pos[0] + 10
            text_y = knee_pos[1] - 10

            # Background rectangle
            cv2.rectangle(out,
                         (text_x - 4, text_y - text_size[1] - 4),
                         (text_x + text_size[0] + 4, text_y + 4),
                         (0, 0, 0), -1)

            # Text
            cv2.putText(out, label, (text_x, text_y), font, font_scale, color, thickness)

    # Hip angle
    hip_pos = keypoints[hip_idx].astype(int)
    if hip_pos[0] > 5 and hip_pos[1] > 5:
        hip_angle = overlay_metrics.get('hip_angle_bdc')

        if hip_angle is not None:
            color = _get_color_for_angle('hip_angle_bdc', assessments)
            label = f"{hip_angle:.0f}"

            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = hip_pos[0] - text_size[0] - 15
            text_y = hip_pos[1]

            cv2.rectangle(out,
                         (text_x - 4, text_y - text_size[1] - 4),
                         (text_x + text_size[0] + 4, text_y + 4),
                         (0, 0, 0), -1)

            cv2.putText(out, label, (text_x, text_y), font, font_scale, color, thickness)

    # Elbow angle
    elbow_pos = keypoints[elbow_idx].astype(int)
    if elbow_pos[0] > 5 and elbow_pos[1] > 5:
        elbow_flex = overlay_metrics.get('elbow_flexion')

        if elbow_flex is not None:
            color = _get_color_for_angle('elbow_flexion', assessments)
            label = f"{elbow_flex:.0f}"

            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = elbow_pos[0] + 10
            text_y = elbow_pos[1]

            cv2.rectangle(out,
                         (text_x - 4, text_y - text_size[1] - 4),
                         (text_x + text_size[0] + 4, text_y + 4),
                         (0, 0, 0), -1)

            cv2.putText(out, label, (text_x, text_y), font, font_scale, color, thickness)

    # Back angle (near shoulder)
    shoulder_pos = keypoints[shoulder_idx].astype(int)
    if shoulder_pos[0] > 5 and shoulder_pos[1] > 5:
        back_angle = overlay_metrics.get('back_angle')

        if back_angle is not None:
            color = _get_color_for_angle('back_angle', assessments)
            label = f"{back_angle:.0f}"

            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = shoulder_pos[0] - text_size[0] - 15
            text_y = shoulder_pos[1] - 15

            cv2.rectangle(out,
                         (text_x - 4, text_y - text_size[1] - 4),
                         (text_x + text_size[0] + 4, text_y + 4),
                         (0, 0, 0), -1)

            cv2.putText(out, label, (text_x, text_y), font, font_scale, color, thickness)

    return out


def _build_overlay_metrics(
    angles: dict,
    assessments: dict | None,
    use_assessment_values: bool
) -> dict:
    """Prefer assessed display values so the preview matches the fit summary."""
    if not assessments or not use_assessment_values:
        return {
            'knee_extension_bdc': angles.get('knee_extension'),
            'hip_angle_bdc': angles.get('hip_flexion'),
            'elbow_flexion': angles.get('elbow_flexion'),
            'back_angle': angles.get('back_angle'),
        }

    return {
        'knee_extension_bdc': _assessment_mean(assessments, 'knee_extension_bdc', angles.get('knee_extension')),
        'hip_angle_bdc': _assessment_mean(assessments, 'hip_angle_bdc', angles.get('hip_flexion')),
        'elbow_flexion': _assessment_mean(assessments, 'elbow_flexion', angles.get('elbow_flexion')),
        'back_angle': _assessment_mean(assessments, 'back_angle', angles.get('back_angle')),
    }


def _assessment_mean(assessments: dict, key: str, fallback: float | None) -> float | None:
    """Get the assessed mean value for an angle, falling back to the raw frame value."""
    if not assessments:
        return fallback

    assessment = assessments.get(key)
    if not assessment:
        return fallback

    measured = assessment.get('measured_mean')
    return measured if measured is not None else fallback


def _get_color_for_angle(angle_name: str, assessments: dict) -> tuple:
    """Get color based on assessment status."""
    if assessments is None:
        return COLORS['unknown']

    # Map angle names to assessment keys
    angle_map = {
        'knee_flexion': 'knee_flexion_bdc',
        'knee_extension': 'knee_extension_bdc',
        'hip_flexion': 'hip_angle_bdc',
        'elbow_flexion': 'elbow_flexion',
        'back_angle': 'back_angle'
    }

    assessment_key = angle_map.get(angle_name, angle_name)

    if assessment_key not in assessments:
        return COLORS['unknown']

    assessment = assessments[assessment_key]
    status = assessment.get('status', 'unknown')

    if status == 'in_range':
        return COLORS['good']
    elif assessment.get('severity'):
        severity_str = assessment['severity'].value if hasattr(assessment['severity'], 'value') else str(assessment['severity'])
        if severity_str == 'marginal':
            return COLORS['marginal']
        elif severity_str == 'poor':
            return COLORS['poor']

    return COLORS['unknown']
