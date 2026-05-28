"""
Compute real-world body segment measurements from pose keypoints and scale factor.

This module converts pixel-based keypoint positions to real-world measurements
(in millimeters) using a calibrated scale factor from wheel detection.
"""

import numpy as np
from typing import Dict, Optional, List


# COCO keypoint indices (from angles.py)
COCO_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}


def compute_segment_lengths(
    keypoints: np.ndarray,
    scale_factor: float,
    side: str = 'right'
) -> Dict[str, Optional[float]]:
    """
    Compute real-world body segment lengths from keypoints.

    Args:
        keypoints: (17, 2) array of pixel coordinates [x, y] for each COCO keypoint
        scale_factor: Calibration scale factor (mm per pixel)
        side: Which side of body to measure ('left', 'right', or 'auto')

    Returns:
        Dictionary with segment lengths in millimeters:
            - inseam: ankle to hip
            - torso: hip to shoulder
            - upper_arm: shoulder to elbow
            - forearm: elbow to wrist
            - arm_reach: shoulder to wrist (total)
            - thigh: hip to knee
            - shin: knee to ankle
            - leg_length: hip to ankle (total)
    """
    measurements = {}

    # Determine which side's keypoints to use
    if side == 'left':
        ankle_idx = COCO_KEYPOINTS['left_ankle']
        knee_idx = COCO_KEYPOINTS['left_knee']
        hip_idx = COCO_KEYPOINTS['left_hip']
        shoulder_idx = COCO_KEYPOINTS['left_shoulder']
        elbow_idx = COCO_KEYPOINTS['left_elbow']
        wrist_idx = COCO_KEYPOINTS['left_wrist']
    else:  # 'right' or 'auto'
        ankle_idx = COCO_KEYPOINTS['right_ankle']
        knee_idx = COCO_KEYPOINTS['right_knee']
        hip_idx = COCO_KEYPOINTS['right_hip']
        shoulder_idx = COCO_KEYPOINTS['right_shoulder']
        elbow_idx = COCO_KEYPOINTS['right_elbow']
        wrist_idx = COCO_KEYPOINTS['right_wrist']

    # Helper function to compute distance
    def distance_mm(idx1: int, idx2: int) -> Optional[float]:
        """Compute distance between two keypoints in mm."""
        p1 = keypoints[idx1]
        p2 = keypoints[idx2]

        # Check if keypoints are valid (not at origin or very close to edges)
        if np.all(p1 < 5) or np.all(p2 < 5):
            return None

        pixel_dist = np.linalg.norm(p2 - p1)
        return pixel_dist * scale_factor

    # Leg segments
    measurements['inseam'] = distance_mm(ankle_idx, hip_idx)
    measurements['thigh'] = distance_mm(hip_idx, knee_idx)
    measurements['shin'] = distance_mm(knee_idx, ankle_idx)
    measurements['leg_length'] = measurements['inseam']  # Same as inseam

    # Torso
    measurements['torso'] = distance_mm(hip_idx, shoulder_idx)

    # Arm segments
    measurements['upper_arm'] = distance_mm(shoulder_idx, elbow_idx)
    measurements['forearm'] = distance_mm(elbow_idx, wrist_idx)
    measurements['arm_reach'] = distance_mm(shoulder_idx, wrist_idx)

    return measurements


def estimate_inseam(
    keypoints: np.ndarray,
    scale_factor: float,
    side: str = 'right'
) -> Optional[float]:
    """
    Estimate inseam length (ankle to hip) in millimeters.

    This is the most critical measurement for saddle height recommendations.
    Standard saddle height is typically 88-90% of inseam for road cycling.

    Args:
        keypoints: (17, 2) array of pixel coordinates
        scale_factor: Calibration scale factor (mm per pixel)
        side: Which side to measure

    Returns:
        Inseam length in mm, or None if keypoints invalid
    """
    measurements = compute_segment_lengths(keypoints, scale_factor, side)
    return measurements.get('inseam')


def estimate_torso_length(
    keypoints: np.ndarray,
    scale_factor: float,
    side: str = 'right'
) -> Optional[float]:
    """
    Estimate torso length (hip to shoulder) in millimeters.

    Used for stack/reach analysis and handlebar positioning.

    Args:
        keypoints: (17, 2) array of pixel coordinates
        scale_factor: Calibration scale factor (mm per pixel)
        side: Which side to measure

    Returns:
        Torso length in mm, or None if keypoints invalid
    """
    measurements = compute_segment_lengths(keypoints, scale_factor, side)
    return measurements.get('torso')


def estimate_arm_reach(
    keypoints: np.ndarray,
    scale_factor: float,
    side: str = 'right'
) -> Optional[float]:
    """
    Estimate arm reach (shoulder to wrist) in millimeters.

    Used for handlebar reach and drop analysis.

    Args:
        keypoints: (17, 2) array of pixel coordinates
        scale_factor: Calibration scale factor (mm per pixel)
        side: Which side to measure

    Returns:
        Arm reach in mm, or None if keypoints invalid
    """
    measurements = compute_segment_lengths(keypoints, scale_factor, side)
    return measurements.get('arm_reach')


def compute_leg_extension_at_bdc(
    keypoints: np.ndarray,
    scale_factor: float,
    side: str = 'right'
) -> Optional[float]:
    """
    Compute full leg extension at bottom dead center (BDC).

    This is the actual measured leg length from hip to pedal when pedal is
    at the bottom of the stroke. Compared to inseam to determine saddle height.

    Typically:
    - Optimal leg extension at BDC: 88-90% of inseam
    - Too high (>95%): knee over-extended, risk of injury
    - Too low (<85%): inefficient power transfer

    Args:
        keypoints: (17, 2) array of pixel coordinates at BDC
        scale_factor: Calibration scale factor (mm per pixel)
        side: Which side to measure

    Returns:
        Leg extension in mm, or None if keypoints invalid
    """
    # Leg extension at BDC is the same as inseam measurement at that instant
    return estimate_inseam(keypoints, scale_factor, side)


def compute_measurements_for_frame(
    frame_data: dict,
    scale_factor: float,
    side: Optional[str] = None
) -> Dict[str, Optional[float]]:
    """
    Compute all body measurements for a single frame of angle data.

    Args:
        frame_data: Dictionary from angle_data with 'keypoints' and 'detected_side'
        scale_factor: Calibration scale factor (mm per pixel)
        side: Optional explicit side override ('left' or 'right')

    Returns:
        Dictionary with all measurements in mm
    """
    keypoints_list = frame_data.get('keypoints')
    if keypoints_list is None:
        return {}

    keypoints = np.array(keypoints_list)
    measurement_side = side or frame_data.get('detected_side', 'right')

    return compute_segment_lengths(keypoints, scale_factor, measurement_side)


def compute_average_measurements(
    angle_data: List[dict],
    scale_factor: float,
    frame_indices: Optional[List[int]] = None,
    side: Optional[str] = None
) -> Dict[str, Optional[float]]:
    """
    Compute average body measurements across multiple frames.

    Useful for getting stable measurements across a pedal stroke or multiple strokes.

    Args:
        angle_data: List of frame dictionaries from video processing
        scale_factor: Calibration scale factor (mm per pixel)
        frame_indices: Optional list of specific frame indices to average.
                      If None, uses all frames with valid keypoints.
        side: Optional explicit side override ('left' or 'right')

    Returns:
        Dictionary with average measurements in mm
    """
    if frame_indices is None:
        frame_indices = range(len(angle_data))

    # Collect measurements from each frame
    all_measurements = {
        'inseam': [],
        'torso': [],
        'arm_reach': [],
        'thigh': [],
        'shin': [],
        'upper_arm': [],
        'forearm': [],
        'leg_length': []
    }

    for idx in frame_indices:
        if idx >= len(angle_data):
            continue

        frame = angle_data[idx]
        measurements = compute_measurements_for_frame(frame, scale_factor, side=side)

        for key, value in measurements.items():
            if value is not None and not np.isnan(value):
                all_measurements[key].append(value)

    # Compute averages
    avg_measurements = {}
    for key, values in all_measurements.items():
        if len(values) > 0:
            avg_measurements[key] = np.mean(values)
            avg_measurements[f'{key}_std'] = np.std(values)
            avg_measurements[f'{key}_min'] = np.min(values)
            avg_measurements[f'{key}_max'] = np.max(values)
        else:
            avg_measurements[key] = None
            avg_measurements[f'{key}_std'] = None
            avg_measurements[f'{key}_min'] = None
            avg_measurements[f'{key}_max'] = None

    return avg_measurements


def estimate_saddle_height_from_inseam(
    inseam_mm: float,
    method: str = 'hamley'
) -> Dict[str, float]:
    """
    Estimate optimal saddle height from inseam measurement.

    Args:
        inseam_mm: Measured inseam length in millimeters
        method: Method to use for estimation
            - 'hamley': 88.3% of inseam (most common)
            - 'lemond': 88.3% of inseam (same as Hamley)
            - 'hinault': 89.3% of inseam
            - 'conservative': 87-88% range
            - 'aggressive': 89-90% range

    Returns:
        Dictionary with:
            - recommended: Single recommended height
            - min: Lower bound of range
            - max: Upper bound of range
            - method: Method used
    """
    if method == 'hamley' or method == 'lemond':
        recommended = inseam_mm * 0.883
        min_height = inseam_mm * 0.875
        max_height = inseam_mm * 0.890
    elif method == 'hinault':
        recommended = inseam_mm * 0.893
        min_height = inseam_mm * 0.885
        max_height = inseam_mm * 0.900
    elif method == 'conservative':
        recommended = inseam_mm * 0.875
        min_height = inseam_mm * 0.870
        max_height = inseam_mm * 0.880
    elif method == 'aggressive':
        recommended = inseam_mm * 0.895
        min_height = inseam_mm * 0.890
        max_height = inseam_mm * 0.900
    else:
        # Default to Hamley
        recommended = inseam_mm * 0.883
        min_height = inseam_mm * 0.875
        max_height = inseam_mm * 0.890

    return {
        'recommended': recommended,
        'min': min_height,
        'max': max_height,
        'method': method,
        'percentage': (recommended / inseam_mm) * 100 if inseam_mm > 0 else 0
    }


def compare_measured_to_recommended_saddle_height(
    measured_leg_extension_mm: float,
    inseam_mm: float,
    method: str = 'hamley'
) -> Dict[str, float]:
    """
    Compare actual measured leg extension to recommended saddle height.

    Args:
        measured_leg_extension_mm: Measured leg extension at BDC
        inseam_mm: Measured inseam
        method: Saddle height estimation method

    Returns:
        Dictionary with:
            - measured: Actual leg extension
            - recommended: Recommended saddle height
            - difference: Measured - recommended (positive = too high)
            - percentage: Current height as % of inseam
            - status: 'too_high', 'optimal', or 'too_low'
    """
    recommended = estimate_saddle_height_from_inseam(inseam_mm, method)

    difference = measured_leg_extension_mm - recommended['recommended']
    percentage = (measured_leg_extension_mm / inseam_mm * 100) if inseam_mm > 0 else 0

    # Determine status
    if measured_leg_extension_mm < recommended['min']:
        status = 'too_low'
    elif measured_leg_extension_mm > recommended['max']:
        status = 'too_high'
    else:
        status = 'optimal'

    return {
        'measured': measured_leg_extension_mm,
        'recommended': recommended['recommended'],
        'min': recommended['min'],
        'max': recommended['max'],
        'difference': difference,
        'percentage': percentage,
        'status': status
    }
