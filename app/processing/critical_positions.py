"""
Extract and analyze angles at critical pedal positions (TDC and BDC).

TDC (Top Dead Centre): Pedal at highest position, maximum knee flexion
BDC (Bottom Dead Centre): Pedal at lowest position, maximum knee extension
"""
import numpy as np
from typing import Dict, List, Optional


def extract_angles_at_positions(
    angle_data: List[dict],
    tdc_frames: List[int],
    bdc_frames: List[int],
    side: str = "auto"
) -> Dict[str, Dict[str, any]]:
    """
    Extract angle measurements at critical pedal positions (TDC and BDC).

    Args:
        angle_data: List of per-frame angle records from VideoWorker
        tdc_frames: Frame indices where pedal is at Top Dead Centre (max knee flexion)
        bdc_frames: Frame indices where pedal is at Bottom Dead Centre (max knee extension)
        side: Which side to analyze ("auto", "left", or "right")

    Returns:
        Dictionary with structure:
        {
            'knee_extension_bdc': {
                'values': [angle1, angle2, ...],  # One per BDC frame
                'mean': float,
                'std': float,
                'min': float,
                'max': float
            },
            'knee_flexion_tdc': { ... },
            'hip_flexion_tdc': { ... },
            'elbow_flexion': { ... },  # averaged across all frames
            'back_angle': { ... }      # averaged across all frames
        }
    """
    # Determine angle key prefix based on side
    if side == "auto":
        prefix = ""
    elif side == "left":
        prefix = "left_"
    elif side == "right":
        prefix = "right_"
    else:
        prefix = ""

    results = {}

    # Knee extension at BDC
    knee_ext_bdc = []
    for frame_idx in bdc_frames:
        if 0 <= frame_idx < len(angle_data):
            val = angle_data[frame_idx].get(f"{prefix}knee_extension")
            if val is not None:
                knee_ext_bdc.append(val)

    results['knee_extension_bdc'] = _compute_stats(knee_ext_bdc)

    # Hip angle (minimum across entire pedal cycle)
    # The minimum hip angle (most closed/acute) typically occurs at TDC, but we check all frames
    hip_angle_min = []
    for rec in angle_data:
        val = rec.get(f"{prefix}hip_flexion")
        if val is not None:
            hip_angle_min.append(val)

    # Get the minimum value across all frames as the representative measurement
    if hip_angle_min:
        min_val = min(hip_angle_min)
        results['hip_angle_bdc'] = {
            'values': [min_val],  # Single representative value
            'mean': min_val,
            'std': 0.0,  # No variance for a single measurement
            'min': min_val,
            'max': min_val,
            'count': 1
        }
    else:
        results['hip_angle_bdc'] = _compute_stats([])

    # Elbow flexion (average across all frames)
    elbow_flex = [
        rec.get(f"{prefix}elbow_flexion")
        for rec in angle_data
        if rec.get(f"{prefix}elbow_flexion") is not None
    ]
    results['elbow_flexion'] = _compute_stats(elbow_flex)

    # Back angle (average across all frames)
    back_angle = [
        rec.get(f"{prefix}back_angle")
        for rec in angle_data
        if rec.get(f"{prefix}back_angle") is not None
    ]
    results['back_angle'] = _compute_stats(back_angle)

    return results


def _compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """
    Compute statistics for a list of angle values.

    Args:
        values: List of angle measurements

    Returns:
        Dictionary with values, mean, std, min, max
    """
    if not values or len(values) == 0:
        return {
            'values': [],
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'count': 0
        }

    values_array = np.array(values)
    return {
        'values': values,
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'count': len(values)
    }


def get_frames_at_positions(
    angle_data: List[dict],
    tdc_frames: List[int],
    bdc_frames: List[int]
) -> Dict[str, List[dict]]:
    """
    Get complete angle records for frames at TDC and BDC positions.

    Args:
        angle_data: List of per-frame angle records
        tdc_frames: Frame indices at Top Dead Centre
        bdc_frames: Frame indices at Bottom Dead Centre

    Returns:
        {
            'tdc_records': [record1, record2, ...],
            'bdc_records': [record1, record2, ...]
        }
    """
    tdc_records = [
        angle_data[idx] for idx in tdc_frames
        if 0 <= idx < len(angle_data)
    ]

    bdc_records = [
        angle_data[idx] for idx in bdc_frames
        if 0 <= idx < len(angle_data)
    ]

    return {
        'tdc_records': tdc_records,
        'bdc_records': bdc_records
    }


def select_representative_frames(
    angle_data: List[dict],
    tdc_frames: List[int],
    bdc_frames: List[int],
    side: str = "auto"
) -> Dict[str, any]:
    """
    Select representative frames (median angle) from TDC and BDC positions.

    Args:
        angle_data: List of per-frame angle records
        tdc_frames: Frame indices at Top Dead Centre
        bdc_frames: Frame indices at Bottom Dead Centre
        side: Which side to analyze ("auto", "left", or "right")

    Returns:
        {
            'tdc_representative': {
                'frame_idx': int,
                'angles': dict,
                'keypoints': list
            },
            'bdc_representative': {...},
            'consistency_metrics': {
                'knee_extension_bdc': {
                    'values': [...],
                    'mean': float,
                    'std': float,
                    'min': float,
                    'max': float,
                    'is_consistent': bool
                },
                ...
            }
        }
    """
    # Determine angle key prefix
    prefix = "" if side == "auto" else f"{side}_"

    # Extract angles at positions
    critical_angles = extract_angles_at_positions(angle_data, tdc_frames, bdc_frames, side)

    # The current UI only displays the BDC representative frame.
    tdc_rep = None

    # Find representative BDC frame (median knee extension)
    bdc_rep = None
    if bdc_frames and critical_angles['knee_extension_bdc']['values']:
        values = critical_angles['knee_extension_bdc']['values']
        median_angle = float(np.median(values))

        best_idx = None
        best_diff = float('inf')
        for frame_idx in bdc_frames:
            if 0 <= frame_idx < len(angle_data):
                angle_val = angle_data[frame_idx].get(f"{prefix}knee_extension")
                if angle_val is not None:
                    diff = abs(angle_val - median_angle)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = frame_idx

        if best_idx is not None:
            bdc_rep = {
                'frame_idx': best_idx,
                'angles': {k: angle_data[best_idx].get(f"{prefix}{k}")
                          for k in ['knee_flexion', 'knee_extension', 'hip_flexion', 'elbow_flexion', 'back_angle']},
                'keypoints': angle_data[best_idx].get('keypoints')
            }

    # Calculate consistency metrics
    consistency_threshold_good = 3.0  # degrees
    consistency_threshold_fair = 5.0  # degrees

    consistency_metrics = {}
    for angle_name, stats in critical_angles.items():
        if stats['std'] is not None:
            if stats['std'] < consistency_threshold_good:
                consistency = 'good'
                is_consistent = True
            elif stats['std'] < consistency_threshold_fair:
                consistency = 'fair'
                is_consistent = True
            else:
                consistency = 'poor'
                is_consistent = False

            consistency_metrics[angle_name] = {
                **stats,
                'consistency': consistency,
                'is_consistent': is_consistent
            }
        else:
            consistency_metrics[angle_name] = {
                **stats,
                'consistency': 'unknown',
                'is_consistent': True
            }

    return {
        'tdc_representative': tdc_rep,
        'bdc_representative': bdc_rep,
        'consistency_metrics': consistency_metrics
    }
