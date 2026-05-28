# metrics.py

import numpy as np
from .data_loading import get_image_dimensions, calculate_head_size, detect_visible_side

PCKH_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

JOINT_ANGLES = {
    'knee':  ('hip',      'knee',  'ankle'),
    'hip':   ('shoulder', 'hip',   'knee'),
    'ankle': ('knee',     'ankle', 'foot_index'),
    'elbow': ('shoulder', 'elbow', 'wrist'),
}


# Internal: compute angle at vertex B given three (x, y) points A, B, C

def _compute_angle(a, b, c):
    """
    Compute the angle in degrees at point B in the triangle A-B-C.

    Args:
        a, b, c: (x, y) tuples

    Returns:
        Angle in degrees [0, 180], or None if vectors are degenerate.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(180.0 - np.degrees(np.arccos(cos_angle)))


# Internal: accumulate raw counts for a single video

def _accumulate_video_counts(video_keypoints, video_head_bboxes, predictions, keypoint_mapping):
    """
    Accumulate raw detection and PCKh correctness counts, and joint angle
    errors, for one video.

    Missing keypoints (pred is None, or keypoint confidence == 0.0) are
    excluded from PCKh numerator and denominator, but counted against DR.
    Frames where any of the three keypoints for an angle is occluded in GT
    or missing in the prediction are excluded from angle MAE.

    Args:
        video_keypoints:   dict {frame_num -> {cvat_name -> {x, y, occluded}}}
        video_head_bboxes: dict {frame_num -> head_bbox}
        predictions:       dict {frame_num -> wrapper | None}  (full dataset)
        keypoint_mapping:  dict {cvat_name -> keypoint_index}

    Returns:
        counts:       dict {part_name -> {detected, total, pckh_correct}}
        angle_errors: dict {angle_name -> list of float (degrees)}
        visible_side: str ('left' or 'right')
    """
    counts = {}
    angle_errors = {name: [] for name in JOINT_ANGLES}

    visible_side = detect_visible_side(video_keypoints)

    # Only evaluate keypoints on the camera-facing side that this model supports
    visible_kp_mapping = {
        cvat_name: kp_idx
        for cvat_name, kp_idx in keypoint_mapping.items()
        if cvat_name.split('_')[0] == visible_side
    }

    if not visible_kp_mapping:
        return counts, angle_errors, visible_side

    # Cache image dimensions once per video (all frames share same resolution)
    sample_frame = next(iter(video_keypoints))
    img_width, img_height = get_image_dimensions(sample_frame)

    # Pre-build angle keypoint mappings for this visible side and model.
    # Each entry: angle_name -> (cvat_proximal, cvat_joint, cvat_distal, idx_p, idx_j, idx_d)
    # Skip angles where any keypoint is absent from this model's mapping.
    active_angles = {}
    for angle_name, (proximal, joint, distal) in JOINT_ANGLES.items():
        cvat_p = f'{visible_side}_{proximal}'
        cvat_j = f'{visible_side}_{joint}'
        cvat_d = f'{visible_side}_{distal}'

        if cvat_p in keypoint_mapping and cvat_j in keypoint_mapping and cvat_d in keypoint_mapping:
            active_angles[angle_name] = (
                cvat_p, cvat_j, cvat_d,
                keypoint_mapping[cvat_p],
                keypoint_mapping[cvat_j],
                keypoint_mapping[cvat_d],
            )

    for frame_num, gt_frame in video_keypoints.items():
        head_bbox = video_head_bboxes.get(frame_num)
        if head_bbox is None:
            continue

        head_size = calculate_head_size(head_bbox)
        if head_size == 0:
            continue

        pred = predictions.get(frame_num)

        # PCKh / Detection Rate
        for cvat_name, kp_idx in visible_kp_mapping.items():
            if cvat_name not in gt_frame:
                continue

            part_name = cvat_name.split('_', 1)[1]  # e.g. 'left_knee' -> 'knee'

            if part_name not in counts:
                counts[part_name] = {
                    'detected':     0,
                    'total':        0,
                    'pckh_correct': {t: 0 for t in PCKH_THRESHOLDS},
                }

            counts[part_name]['total'] += 1

            if pred is None:
                continue

            kp = pred[kp_idx]

            if kp.confidence == 0.0:
                continue

            counts[part_name]['detected'] += 1

            pred_x = kp.x * img_width
            pred_y = kp.y * img_height
            gt_kp  = gt_frame[cvat_name]

            distance   = np.sqrt((pred_x - gt_kp['x']) ** 2 + (pred_y - gt_kp['y']) ** 2)
            normalised = distance / head_size

            for threshold in PCKH_THRESHOLDS:
                if normalised <= threshold:
                    counts[part_name]['pckh_correct'][threshold] += 1

        # Joint Angle ME
        if pred is None:
            continue

        for angle_name, (cvat_p, cvat_j, cvat_d, idx_p, idx_j, idx_d) in active_angles.items():
            # All three GT keypoints must exist and be visible
            if cvat_p not in gt_frame or cvat_j not in gt_frame or cvat_d not in gt_frame:
                continue
            if gt_frame[cvat_p]['occluded'] or gt_frame[cvat_j]['occluded'] or gt_frame[cvat_d]['occluded']:
                continue

            # All three predicted keypoints must have been detected
            kp_p = pred[idx_p]
            kp_j = pred[idx_j]
            kp_d = pred[idx_d]
            if kp_p.confidence == 0.0 or kp_j.confidence == 0.0 or kp_d.confidence == 0.0:
                continue

            # GT angle (pixel coordinates)
            gt_angle = _compute_angle(
                (gt_frame[cvat_p]['x'], gt_frame[cvat_p]['y']),
                (gt_frame[cvat_j]['x'], gt_frame[cvat_j]['y']),
                (gt_frame[cvat_d]['x'], gt_frame[cvat_d]['y']),
            )

            # Predicted angle (normalised → pixel coordinates)
            pred_angle = _compute_angle(
                (kp_p.x * img_width, kp_p.y * img_height),
                (kp_j.x * img_width, kp_j.y * img_height),
                (kp_d.x * img_width, kp_d.y * img_height),
            )

            if gt_angle is None or pred_angle is None:
                continue
            
            angle_errors[angle_name].append(gt_angle - pred_angle)

    return counts, angle_errors, visible_side


# Internal: compute metrics for a single video from its raw counts

def _compute_video_metrics(counts, angle_errors):
    """
    Convert raw counts and angle errors for one video into DR, PCKh, AUC,
    and Joint Angle MAE.

    Returns dict with keys: detection_rate, pckh, auc, angle_mae, n_frames
    Returns empty dict if counts is empty.
    """
    if not counts:
        return {}

    parts = sorted(counts.keys())

    # Detection Rate
    dr_per_part = {}
    for part in parts:
        c = counts[part]
        dr_per_part[part] = c['detected'] / c['total'] if c['total'] > 0 else 0.0

    # PCKh per threshold
    pckh_per_threshold = {}
    for t in PCKH_THRESHOLDS:
        per_part = {}
        for part in parts:
            c = counts[part]
            if c['detected'] > 0:
                per_part[part] = c['pckh_correct'][t] / c['detected'] * 100.0
            else:
                per_part[part] = 0.0
        pckh_per_threshold[t] = per_part

    # AUC
    auc_per_part = {}
    for part in parts:
        pckh_curve = [pckh_per_threshold[t][part] for t in PCKH_THRESHOLDS]
        auc_per_part[part] = _compute_auc(pckh_curve)

    # Joint Angle Errors
    angle_mae, angle_rmse, angle_me = {}, {}, {}
    for angle_name, errors in angle_errors.items():
        if errors:
            errs = np.array(errors)
            angle_mae[angle_name]  = float(np.mean(np.abs(errs)))
            angle_rmse[angle_name] = float(np.sqrt(np.mean(errs ** 2)))
            angle_me[angle_name]   = float(np.mean(errs))

        # Angles with no valid frames are omitted (model lacks the keypoints)

    # n_frames (detected and total, per part)
    n_frames_per_part = {
        part: {
            'detected': counts[part]['detected'],
            'total':    counts[part]['total'],
        }
        for part in parts
    }

    return {
        'detection_rate': dr_per_part,
        'pckh':           pckh_per_threshold,
        'auc':            auc_per_part,
        'angle_mae':      angle_mae,
        'angle_rmse':     angle_rmse,
        'angle_me':       angle_me,
        'n_frames':       n_frames_per_part,
    }


# Internal: AUC from per-threshold PCKh values

def _compute_auc(pckh_at_thresholds):
    """
    Area under the PCKh curve via trapezoid rule, normalised to [0, 1] range,
    returned on [0, 100] scale to match PCKh%.

    Args:
        pckh_at_thresholds: list of PCKh% values aligned to PCKH_THRESHOLDS
    """
    thresholds = PCKH_THRESHOLDS
    auc = np.trapz(pckh_at_thresholds, thresholds) / (thresholds[-1] - thresholds[0])
    return float(auc)


# Internal: aggregate per-video metrics into overall mean/std

def _aggregate_across_videos(per_video_metrics):
    """
    Aggregate per-video metrics into overall mean +/- std.

    For each metric and each part, collects per-video values then computes
    mean and std. Parts missing from a video are skipped for that video.

    Args:
        per_video_metrics: dict {video_id -> video_metrics_dict}

    Returns:
        aggregated metrics dict
    """
    # Collect all parts that appear across any video
    all_parts = set()
    for vm in per_video_metrics.values():
        if vm:
            all_parts.update(vm.get('detection_rate', {}).keys())
    parts = sorted(all_parts)

    def _mean_std(values):
        values = [v for v in values if v is not None]
        if not values:
            return {'mean': None, 'std': None}
        return {
            'mean': float(np.mean(values)),
            'std':  float(np.std(values)),
        }

    # Detection Rate
    dr_per_part = {}
    for part in parts:
        values = [
            vm['detection_rate'][part]
            for vm in per_video_metrics.values()
            if vm and part in vm.get('detection_rate', {})
        ]
        dr_per_part[part] = _mean_std(values)

    dr_means = [dr_per_part[p]['mean'] for p in parts if dr_per_part[p]['mean'] is not None]
    dr_overall = _mean_std(dr_means)

    # PCKh per threshold
    pckh_per_threshold = {}
    for t in PCKH_THRESHOLDS:
        per_part = {}
        for part in parts:
            values = [
                vm['pckh'][t][part]
                for vm in per_video_metrics.values()
                if vm and part in vm.get('pckh', {}).get(t, {})
            ]
            per_part[part] = _mean_std(values)

        part_means = [per_part[p]['mean'] for p in parts if per_part[p]['mean'] is not None]
        pckh_per_threshold[t] = {
            'mean':         float(np.mean(part_means)) if part_means else None,
            'std':          float(np.std(part_means))  if part_means else None,
            'per_keypoint': per_part,
        }

    # AUC
    auc_per_part = {}
    for part in parts:
        values = [
            vm['auc'][part]
            for vm in per_video_metrics.values()
            if vm and part in vm.get('auc', {})
        ]
        auc_per_part[part] = _mean_std(values)

    auc_means = [auc_per_part[p]['mean'] for p in parts if auc_per_part[p]['mean'] is not None]
    auc_overall = _mean_std(auc_means)

    # Joint Angle MAE / RMSE / ME
    all_angle_names = set()
    for vm in per_video_metrics.values():
        if vm:
            all_angle_names.update(vm.get('angle_mae', {}).keys())

    def _agg_angle_metric(metric_key):
        agg = {}
        for angle_name in sorted(all_angle_names):
            values = [
                vm[metric_key][angle_name]
                for vm in per_video_metrics.values()
                if vm and angle_name in vm.get(metric_key, {})
            ]
            agg[angle_name] = _mean_std(values)
        joint_means = [agg[a]['mean'] for a in agg if agg[a]['mean'] is not None]
        overall = _mean_std(joint_means)
        return agg, overall

    angle_mae_agg,  angle_mae_overall  = _agg_angle_metric('angle_mae')
    angle_rmse_agg, angle_rmse_overall = _agg_angle_metric('angle_rmse')
    angle_me_agg,   angle_me_overall   = _agg_angle_metric('angle_me')

    # n_frames (summed across videos, per part)
    n_frames_per_part = {}
    for part in parts:
        detected = sum(
            vm['n_frames'][part]['detected']
            for vm in per_video_metrics.values()
            if vm and part in vm.get('n_frames', {})
        )
        total = sum(
            vm['n_frames'][part]['total']
            for vm in per_video_metrics.values()
            if vm and part in vm.get('n_frames', {})
        )
        n_frames_per_part[part] = {'detected': detected, 'total': total}

    return {
        'detection_rate': {
            'overall':      dr_overall,
            'per_keypoint': dr_per_part,
        },
        'pckh': pckh_per_threshold,
        'auc': {
            'overall':      auc_overall,
            'per_keypoint': auc_per_part,
        },
        'angle_mae': {
            'overall':   angle_mae_overall,
            'per_joint': angle_mae_agg,
        },
        'angle_rmse': {
            'overall':   angle_rmse_overall,
            'per_joint': angle_rmse_agg,
        },
        'angle_me': {
            'overall':   angle_me_overall,
            'per_joint': angle_me_agg,
        },
        'n_frames': {
            'per_keypoint': n_frames_per_part,
        },
    }

# Public entry point

def evaluate_model(videos, predictions, keypoint_mapping):
    """
    Evaluate a model's serialized predictions against ground truth.

    Args:
        videos:           dict from split_frames_into_videos()
        predictions:      dict {frame_num -> SerializedKeypointWrapper | None}
                          as returned by load_predictions()
        keypoint_mapping: dict {cvat_name -> keypoint_index}
                          as returned by load_predictions()

    Returns:
        dict with keys:
            'aggregated' — mean/std metrics across all videos
            'per_video'  — per-video breakdown {video_id -> metrics}
    """
    per_video_metrics = {}

    for video_id, video_data in videos.items():
        counts, angle_errors, _ = _accumulate_video_counts(
            video_keypoints   = video_data['keypoints'],
            video_head_bboxes = video_data['head_bboxes'],
            predictions       = predictions,
            keypoint_mapping  = keypoint_mapping,
        )
        per_video_metrics[video_id] = _compute_video_metrics(counts, angle_errors)

    aggregated = _aggregate_across_videos(per_video_metrics)

    return {
        'aggregated': aggregated,
        'per_video':  per_video_metrics,
    }
