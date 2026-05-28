"""
Bicycle fit assessment based on measured joint angles.

Compares measured angles against discipline-specific target ranges and generates
fit recommendations based on deviations.
"""
from typing import Dict, List, Tuple
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP


class Discipline(Enum):
    """Cycling disciplines with different fit requirements."""
    ROAD = "road"
    MOUNTAIN = "mountain"
    TIME_TRIAL = "time_trial"
    TRIATHLON = "triathlon"


class DeviationSeverity(Enum):
    """Severity levels for angle deviations."""
    GOOD = "good"           # Within target range
    MARGINAL = "marginal"   # Slightly outside target (< 5° deviation)
    POOR = "poor"           # Significantly outside target (≥ 5° deviation)


# Target angle ranges for each discipline (in degrees)
# Source: Burt (2014) - Bike Fit: Optimise your bike position for high performance and injury avoidance
# All measurements at BDC (Bottom Dead Centre) unless marked as Static
TARGET_RANGES = {
    Discipline.ROAD: {
        'knee_extension_bdc': (35, 40),      # BDC - Knee extension
        'hip_angle_bdc': (55, 65),           # BDC - Minimum hip angle
        'elbow_flexion': (150, 170),         # Static - Upper body
        'back_angle': (45, 45),              # Static - Single target value
    },
    Discipline.MOUNTAIN: {
        'knee_extension_bdc': (35, 40),
        'hip_angle_bdc': (60, 80),
        'elbow_flexion': (150, 170),
        'back_angle': (50, 50),              # Static - Single target value
    },
    Discipline.TIME_TRIAL: {
        'knee_extension_bdc': (37, 42),
        'hip_angle_bdc': (35, 45),           # More aggressive position
        'elbow_flexion': (90, 100),          # Tighter elbow on aerobars
        'back_angle': (20, 20),              # Static - Single target value
    },
    Discipline.TRIATHLON: {
        'knee_extension_bdc': (37, 42),
        'hip_angle_bdc': (45, 55),           # Slightly less aggressive than TT
        'elbow_flexion': (90, 100),
        'back_angle': (25, 25),              # Static - Single target value
    },
}


def _round_angle_value(value: float | None) -> float | None:
    """Round angle values to whole degrees using conventional half-up rounding."""
    if value is None:
        return None
    return float(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def assess_angle(
    measured_stats: Dict[str, any],
    target_range: Tuple[float, float],
    angle_name: str
) -> Dict[str, any]:
    """
    Assess angle measurements (with statistics) against target range.

    Args:
        measured_stats: Statistics dict with 'mean', 'std', 'min', 'max', 'count', 'values'
        target_range: (min, max) target range in degrees
        angle_name: Name of the angle being assessed

    Returns:
        {
            'angle_name': str,
            'measured_mean': float or None,
            'measured_range': (float, float) or None,
            'std': float or None,
            'count': int,
            'consistency': str,  # 'good', 'fair', 'poor', 'unknown'
            'consistency_warning': str or None,
            'target_min': float,
            'target_max': float,
            'deviation': float or None,
            'severity': DeviationSeverity,
            'status': str
        }
    """
    target_min, target_max = target_range
    measured = _round_angle_value(measured_stats.get('mean'))
    measured_min = _round_angle_value(measured_stats.get('min'))
    measured_max = _round_angle_value(measured_stats.get('max'))

    if measured is None:
        return {
            'angle_name': angle_name,
            'measured_mean': None,
            'measured_range': None,
            'std': None,
            'count': 0,
            'consistency': 'unknown',
            'consistency_warning': None,
            'target_min': target_min,
            'target_max': target_max,
            'deviation': None,
            'severity': DeviationSeverity.POOR,
            'status': 'not_detected'
        }

    # Determine deviation based on mean
    if measured < target_min:
        deviation = measured - target_min  # Negative
        status = 'too_low'
    elif measured > target_max:
        deviation = measured - target_max  # Positive
        status = 'too_high'
    else:
        deviation = 0.0
        status = 'in_range'

    # Determine severity based on deviation
    if status == 'in_range':
        severity = DeviationSeverity.GOOD
    elif abs(deviation) < 5.0:
        severity = DeviationSeverity.MARGINAL
    else:
        severity = DeviationSeverity.POOR

    # Consistency assessment
    std = measured_stats.get('std')
    consistency = 'unknown'
    consistency_warning = None

    if std is not None:
        if std < 3.0:
            consistency = 'good'
        elif std < 5.0:
            consistency = 'fair'
            consistency_warning = f"Moderate variability (σ={std:.1f}°)"
        else:
            consistency = 'poor'
            consistency_warning = f"High variability (σ={std:.1f}°) - inconsistent pedal strokes"

    return {
        'angle_name': angle_name,
        'measured_mean': measured,
        'measured_range': (measured_min, measured_max),
        'std': std,
        'count': measured_stats.get('count', 0),
        'consistency': consistency,
        'consistency_warning': consistency_warning,
        'target_min': target_min,
        'target_max': target_max,
        'deviation': deviation,
        'severity': severity,
        'status': status
    }


def assess_fit(
    critical_angles: Dict[str, Dict[str, any]],
    discipline: Discipline = Discipline.ROAD
) -> Dict[str, any]:
    """
    Perform complete fit assessment for all measured angles.

    Args:
        critical_angles: Output from critical_positions.extract_angles_at_positions()
        discipline: Cycling discipline for target ranges

    Returns:
        {
            'discipline': Discipline,
            'assessments': {
                'knee_extension_bdc': {...},
                'knee_flexion_tdc': {...},
                # ...
            },
            'summary': {
                'total_angles': int,
                'in_range': int,
                'marginal': int,
                'poor': int,
                'not_detected': int
            }
        }
    """
    target_ranges = TARGET_RANGES[discipline]
    assessments = {}

    # Assess each angle
    for angle_name, angle_stats in critical_angles.items():
        if angle_name in target_ranges:
            target_range = target_ranges[angle_name]
            assessments[angle_name] = assess_angle(angle_stats, target_range, angle_name)

    # Generate summary
    summary = {
        'total_angles': len(assessments),
        'in_range': 0,
        'marginal': 0,
        'poor': 0,
        'not_detected': 0
    }

    for assessment in assessments.values():
        if assessment['status'] == 'not_detected':
            summary['not_detected'] += 1
        elif assessment['severity'] == DeviationSeverity.GOOD:
            summary['in_range'] += 1
        elif assessment['severity'] == DeviationSeverity.MARGINAL:
            summary['marginal'] += 1
        elif assessment['severity'] == DeviationSeverity.POOR:
            summary['poor'] += 1

    return {
        'discipline': discipline,
        'assessments': assessments,
        'summary': summary
    }


def generate_recommendations(assessment_result: Dict[str, any]) -> List[Dict[str, str]]:
    """
    Generate prioritized bike adjustment recommendations based on fit assessment.

    Recommendations follow traditional fitting priority:
    1. Saddle height (affects knee extension/flexion)
    2. Saddle fore-aft (affects knee-pedal relationship)
    3. Handlebar position (affects upper body angles)

    Args:
        assessment_result: Output from assess_fit()

    Returns:
        List of recommendations, each with:
        {
            'priority': int (1=highest),
            'category': str ('saddle_height', 'handlebars', etc.),
            'action': str (description of adjustment),
            'reason': str (which angles are affected),
            'severity': DeviationSeverity
        }
    """
    assessments = assessment_result['assessments']
    recommendations = []

    # Priority 1: Saddle Height
    # Affects: knee_extension_bdc
    knee_ext = assessments.get('knee_extension_bdc', {})

    # Knee extension recommendations (PRIMARY metric for saddle height)
    knee_ext_status = knee_ext.get('status')

    if knee_ext_status == 'too_low':  # Knee too bent at BDC (< 35°)
        recommendations.append({
            'priority': 1,
            'category': 'saddle_height',
            'action': f"Raise saddle (knee extension at BDC is {knee_ext['measured_mean']:.1f}°, target: {knee_ext['target_min']}-{knee_ext['target_max']}°)",
            'reason': 'Knee too bent at bottom of pedal stroke',
            'severity': knee_ext['severity']
        })
    elif knee_ext_status == 'too_high':  # Knee too straight at BDC (> 40°)
        recommendations.append({
            'priority': 1,
            'category': 'saddle_height',
            'action': f"Lower saddle (knee extension at BDC is {knee_ext['measured_mean']:.1f}°, target: {knee_ext['target_min']}-{knee_ext['target_max']}°)",
            'reason': 'Knee too straight at bottom of pedal stroke',
            'severity': knee_ext['severity']
        })

    # Priority 2: Handlebar Position
    # Affects: hip_angle_bdc, back_angle, elbow_flexion
    hip_angle = assessments.get('hip_angle_bdc', {})
    back = assessments.get('back_angle', {})
    elbow = assessments.get('elbow_flexion', {})

    if hip_angle.get('status') == 'too_low':  # Hip too closed at BDC
        recommendations.append({
            'priority': 2,
            'category': 'handlebars',
            'action': f"Raise handlebars or shorten stem (hip angle at BDC is {hip_angle['measured_mean']:.1f}°, target: {hip_angle['target_min']}-{hip_angle['target_max']}°)",
            'reason': 'Hip angle too closed (cramped position)',
            'severity': hip_angle['severity']
        })
    elif hip_angle.get('status') == 'too_high':  # Hip too open at BDC
        recommendations.append({
            'priority': 2,
            'category': 'handlebars',
            'action': f"Lower handlebars or lengthen stem (hip angle at BDC is {hip_angle['measured_mean']:.1f}°, target: {hip_angle['target_min']}-{hip_angle['target_max']}°)",
            'reason': 'Hip angle too open (too upright)',
            'severity': hip_angle['severity']
        })

    if back.get('status') == 'too_low':  # Back too horizontal/aggressive
        recommendations.append({
            'priority': 2,
            'category': 'handlebars',
            'action': f"Raise handlebars or shorten stem (back angle is {back['measured_mean']:.1f}°, target: {back['target_min']}-{back['target_max']}°)",
            'reason': 'Back position too low/aggressive',
            'severity': back['severity']
        })
    elif back.get('status') == 'too_high':  # Back too upright
        recommendations.append({
            'priority': 2,
            'category': 'handlebars',
            'action': f"Lower handlebars or lengthen stem (back angle is {back['measured_mean']:.1f}°, target: {back['target_min']}-{back['target_max']}°)",
            'reason': 'Back position too upright',
            'severity': back['severity']
        })

    if elbow.get('status') == 'too_high':  # Arms too straight
        recommendations.append({
            'priority': 2,
            'category': 'handlebars',
            'action': f"Shorten stem or raise handlebars (elbow angle is {elbow['measured_mean']:.1f}°, target: {elbow['target_min']}-{elbow['target_max']}°)",
            'reason': 'Arms too straight (insufficient shock absorption)',
            'severity': elbow['severity']
        })
    elif elbow.get('status') == 'too_low':  # Arms too bent
        recommendations.append({
            'priority': 2,
            'category': 'handlebars',
            'action': f"Lengthen stem or lower handlebars (elbow angle is {elbow['measured_mean']:.1f}°, target: {elbow['target_min']}-{elbow['target_max']}°)",
            'reason': 'Arms too bent (cramped upper body)',
            'severity': elbow['severity']
        })

    # Sort by priority and severity
    recommendations.sort(key=lambda x: (x['priority'], x['severity'].value))

    # If no issues found
    if not recommendations:
        recommendations.append({
            'priority': 0,
            'category': 'none',
            'action': 'No adjustments needed - all angles within target ranges',
            'reason': 'Bike fit looks good!',
            'severity': DeviationSeverity.GOOD
        })

    return recommendations
