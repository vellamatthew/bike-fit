"""
Perspective correction using wheel detection and homography optimization.

This module detects bike wheels using instance segmentation, fits ellipses to them,
and optimizes a homography transformation to correct perspective distortion.
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from typing import Optional, Tuple, Dict, List


class PerspectiveCorrectionError(Exception):
    """Raised when perspective correction fails."""
    pass


IDENTITY_HOMOGRAPHY_PARAMS = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
FREE_HOMOGRAPHY_INDICES = [0, 1, 3, 4, 6, 7]  # h1, h2, h4, h5, h7, h8


def build_constrained_homography(free_params: np.ndarray) -> np.ndarray:
    """Build a homography with translation terms fixed at zero."""
    h = IDENTITY_HOMOGRAPHY_PARAMS.copy()
    for idx, value in zip(FREE_HOMOGRAPHY_INDICES, free_params):
        h[idx] = value

    return np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ])


def load_wheel_segmentation_model():
    """
    Load the YOLO wheel segmentation model.

    Returns:
        YOLO model instance

    Raises:
        FileNotFoundError: If model file not found
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics package required for wheel detection")

    # Look for wheel.pt in project root
    project_root = Path.cwd()
    model_path = project_root / "wheel.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Wheel segmentation model not found at {model_path}\n"
            "Please place your trained wheel segmentation model as 'wheel.pt' in the project root."
        )

    model = YOLO(model_path)
    return model


def sample_ellipse_points(ellipse: Tuple, n_points: int = 100) -> np.ndarray:
    """
    Sample points uniformly around an ellipse.

    Args:
        ellipse: OpenCV ellipse tuple ((cx, cy), (width, height), angle)
        n_points: Number of points to sample

    Returns:
        Array of shape (n_points, 2) with sampled points
    """
    center, axes, angle = ellipse
    cx, cy = center
    a, b = axes[0] / 2, axes[1] / 2  # semi-axes
    angle_rad = np.deg2rad(angle)

    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Points on axis-aligned ellipse
    x = a * np.cos(t)
    y = b * np.sin(t)

    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = cx + x * cos_a - y * sin_a
    y_rot = cy + x * sin_a + y * cos_a

    return np.column_stack([x_rot, y_rot])


def detect_wheels(model, frame: np.ndarray, verbose: bool = False) -> Optional[List[np.ndarray]]:
    """
    Detect wheel masks in a frame using instance segmentation.

    Args:
        model: YOLO segmentation model
        frame: Input frame (BGR)
        verbose: Print detection info

    Returns:
        List of masks (numpy arrays) if at least 2 wheel candidates are detected, None otherwise
    """
    results = model(frame, verbose=False)
    result = results[0]

    if result.masks is None:
        if verbose:
            print("[Perspective] No wheels detected in frame")
        return None

    masks = result.masks.data.cpu().numpy()

    if len(masks) < 2:
        if verbose:
            print(f"[Perspective] Only {len(masks)} wheel(s) detected, need 2")
        return None

    if len(masks) > 2:
        if verbose:
            print(f"[Perspective] {len(masks)} wheel candidates detected")

    return list(masks)


def ellipse_center_distance(ellipse_a: Tuple, ellipse_b: Tuple) -> float:
    """Return Euclidean distance between two ellipse centers."""
    center_a = np.array(ellipse_a[0], dtype=float)
    center_b = np.array(ellipse_b[0], dtype=float)
    return float(np.linalg.norm(center_a - center_b))


def average_ellipse_diameter(ellipse: Tuple) -> float:
    """Return the average fitted diameter for an ellipse."""
    return float((ellipse[1][0] + ellipse[1][1]) / 2)


def select_distinct_wheel_ellipses(
    candidates: List[Tuple[Tuple, float]],
    verbose: bool = False,
    log_prefix: str = "[Perspective]"
) -> Optional[List[Tuple]]:
    """
    Select two plausible wheel ellipses while rejecting duplicate detections.

    The wheel model can occasionally emit multiple instance masks for the same
    physical wheel. Choose the largest-area pair whose centers are far enough
    apart to represent two different wheels.
    """
    if len(candidates) < 2:
        return None

    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    best_rejected_distance = None

    for i in range(len(ranked) - 1):
        for j in range(i + 1, len(ranked)):
            ellipse_a, area_a = ranked[i]
            ellipse_b, area_b = ranked[j]
            center_distance = ellipse_center_distance(ellipse_a, ellipse_b)
            min_diameter = min(
                average_ellipse_diameter(ellipse_a),
                average_ellipse_diameter(ellipse_b)
            )
            min_center_distance = max(25.0, min_diameter * 0.35)

            if center_distance >= min_center_distance:
                selected = sorted([ellipse_a, ellipse_b], key=lambda e: e[0][0])
                if verbose:
                    print(
                        f"{log_prefix} Selected 2 distinct wheels "
                        f"(center distance={center_distance:.1f}px, "
                        f"areas={area_a:.0f}/{area_b:.0f})"
                    )
                return selected

            if best_rejected_distance is None or center_distance > best_rejected_distance:
                best_rejected_distance = center_distance

    if verbose:
        distance_text = (
            f"{best_rejected_distance:.1f}px" if best_rejected_distance is not None else "n/a"
        )
        print(
            f"{log_prefix} Wheel candidates appear to be duplicate detections "
            f"(largest center distance={distance_text})"
        )

    return None


def fit_ellipses_to_masks(
    masks: List[np.ndarray],
    img_shape: Tuple[int, int, int],
    verbose: bool = False
) -> Optional[List[Tuple]]:
    """
    Fit ellipses to wheel masks with quality checks.

    Args:
        masks: List of wheel masks
        img_shape: Shape of original image (H, W, C)
        verbose: Print fitting info

    Returns:
        List of ellipse tuples if successful, None if fitting fails quality checks
    """
    candidates = []
    h, w = img_shape[:2]

    for i, mask in enumerate(masks):
        # Resize mask to image dimensions
        mask_resized = cv2.resize(mask, (w, h))
        mask_uint8 = (mask_resized * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            if verbose:
                print(f"[Perspective] Wheel {i+1}: No contours found")
            return None

        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) < 5:
            if verbose:
                print(f"[Perspective] Wheel {i+1}: Not enough points for ellipse fitting")
            return None

        # Fit ellipse
        ellipse = cv2.fitEllipse(largest_contour)
        center, axes, angle = ellipse

        # Quality checks
        # 1. Check if wheel touches edge
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        touches_edge = (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)

        if touches_edge:
            if verbose:
                print(f"[Perspective] Wheel {i+1}: Touches image edge - unreliable")
            return None

        # 2. Check ellipse fit quality
        ellipse_points = sample_ellipse_points(ellipse, n_points=100)
        distances = cdist(largest_contour.reshape(-1, 2), ellipse_points).min(axis=1)
        mean_error = np.mean(distances)

        if mean_error > 10.0:  # Allow small segmentation imperfections.
            if verbose:
                print(f"[Perspective] Wheel {i+1}: Poor ellipse fit (error={mean_error:.1f}px)")
            return None

        if verbose:
            print(f"[Perspective] Wheel {i+1}: center=({center[0]:.1f},{center[1]:.1f}), "
                  f"axes=({axes[0]:.1f},{axes[1]:.1f}), angle={angle:.1f}°, error={mean_error:.1f}px")

        candidates.append((ellipse, float(mask.sum())))

    ellipses = select_distinct_wheel_ellipses(candidates, verbose=verbose)

    return ellipses


def transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography to 2D points.

    Args:
        points: Array of shape (N, 2)
        H: 3x3 homography matrix

    Returns:
        Transformed points of shape (N, 2)
    """
    n = len(points)
    pts_homog = np.column_stack([points, np.ones(n)])
    transformed = (H @ pts_homog.T).T
    # Convert from homogeneous coordinates
    transformed = transformed[:, :2] / (transformed[:, 2:] + 1e-10)
    return transformed


def homography_output_size(H: np.ndarray, img_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int]]:
    """
    Return the full-frame output canvas size required by a homography.

    The optimizer works from wheel ellipses, so a low wheel loss can still imply
    extreme transformed frame corners. This helper lets callers reject those
    unsafe transforms before OpenCV attempts a huge allocation.
    """
    h, w = img_shape[:2]
    corners = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype=np.float32)

    transformed_corners = transform_points(corners, H)
    if not np.all(np.isfinite(transformed_corners)):
        return None

    min_x, min_y = transformed_corners.min(axis=0)
    max_x, max_y = transformed_corners.max(axis=0)
    out_w = int(np.ceil(max_x - min_x))
    out_h = int(np.ceil(max_y - min_y))

    if out_w <= 0 or out_h <= 0:
        return None

    return out_w, out_h


def is_homography_output_reasonable(
    H: np.ndarray,
    img_shape: Tuple[int, int, int],
    max_area_multiplier: float = 8.0,
    max_dimension_multiplier: float = 4.0
) -> bool:
    """Return True when a homography will not create an unsafe output canvas."""
    output_size = homography_output_size(H, img_shape)
    if output_size is None:
        return False

    out_w, out_h = output_size
    h, w = img_shape[:2]
    max_area = max(1, int(w * h * max_area_multiplier))
    max_dimension = max(1, int(max(w, h) * max_dimension_multiplier))

    return (out_w * out_h <= max_area) and (out_w <= max_dimension) and (out_h <= max_dimension)


def serialize_ellipses(ellipses: Optional[List[Tuple]]) -> Optional[List]:
    """Convert OpenCV ellipse tuples into JSON-safe nested lists."""
    if ellipses is None:
        return None

    serialized = []
    for center, axes, angle in ellipses:
        serialized.append([
            [float(center[0]), float(center[1])],
            [float(axes[0]), float(axes[1])],
            float(angle),
        ])
    return serialized


def deserialize_ellipses(ellipses: Optional[List]) -> Optional[List[Tuple]]:
    """Convert serialized ellipse data back into OpenCV ellipse tuples."""
    if ellipses is None:
        return None

    return [
        (
            (float(item[0][0]), float(item[0][1])),
            (float(item[1][0]), float(item[1][1])),
            float(item[2]),
        )
        for item in ellipses
    ]


def fit_ellipse_to_points(points: np.ndarray) -> Optional[Tuple]:
    """
    Fit ellipse to points using OpenCV.

    Args:
        points: Array of shape (N, 2)

    Returns:
        Ellipse tuple or None if fitting fails
    """
    points_int = points.astype(np.float32).reshape(-1, 1, 2)
    if len(points) >= 5:
        try:
            ellipse = cv2.fitEllipse(points_int)
            return ellipse
        except:
            return None
    return None


def ellipse_metrics(ellipse: Tuple) -> Dict:
    """
    Extract metrics from an ellipse.

    Args:
        ellipse: OpenCV ellipse tuple

    Returns:
        Dictionary with center, radius, eccentricity, aspect_ratio, axes
    """
    center, axes, angle = ellipse
    a, b = axes[0] / 2, axes[1] / 2  # semi-axes

    # Ensure a >= b for consistent eccentricity calculation
    if a < b:
        a, b = b, a

    radius = (a + b) / 2
    eccentricity = np.sqrt(1 - (b / a) ** 2) if a > 0 else 0
    aspect_ratio = b / a if a > 0 else 1

    return {
        'center': np.array(center),
        'radius': radius,
        'eccentricity': eccentricity,
        'aspect_ratio': aspect_ratio,
        'axes': (a, b)
    }


def perspective_correction_loss(
    free_params: np.ndarray,
    ellipses: List[Tuple],
    img_shape: Tuple[int, int, int],
    return_details: bool = False
):
    """
    Loss function for perspective correction optimization.

    Goals:
    1. Both wheels should be circular (aspect_ratio → 1)
    2. Both wheels should have equal radii
    3. Wheel centers should be horizontally aligned (same y-coordinate)

    Args:
        free_params: Six free homography parameters; translation is fixed at zero
        ellipses: List of 2 ellipse tuples
        img_shape: Image shape (H, W, C)
        return_details: If True, return detailed metrics

    Returns:
        Loss value (float) or details dictionary
    """
    H = build_constrained_homography(free_params)

    # Reject degenerate homographies
    det = np.linalg.det(H)
    if det < 1e-8 or det > 1e8:
        return 1e10 if not return_details else {'total_loss': 1e10}

    metrics_list = []

    for ellipse in ellipses:
        points = sample_ellipse_points(ellipse, n_points=100)
        transformed_pts = transform_points(points, H)

        # Reject if points explode to unreasonable coordinates
        if np.any(np.abs(transformed_pts) > img_shape[1] * 10):
            return 1e10 if not return_details else {'total_loss': 1e10}

        new_ellipse = fit_ellipse_to_points(transformed_pts)
        if new_ellipse is None:
            return 1e10 if not return_details else {'total_loss': 1e10}

        metrics_list.append(ellipse_metrics(new_ellipse))

    m1, m2 = metrics_list

    # Loss components
    circularity_loss = (1 - m1['aspect_ratio'])**2 + (1 - m2['aspect_ratio'])**2

    mean_radius = (m1['radius'] + m2['radius']) / 2
    radius_loss = ((m1['radius'] - m2['radius']) / (mean_radius + 1e-6))**2

    y_diff = np.abs(m1['center'][1] - m2['center'][1])
    alignment_loss = (y_diff / (mean_radius + 1e-6))**2

    total_loss = 0.333 * circularity_loss + 0.599 * radius_loss + 0.068 * alignment_loss

    if return_details:
        return {
            'total_loss': total_loss,
            'circularity_loss': circularity_loss,
            'radius_loss': radius_loss,
            'alignment_loss': alignment_loss,
            'metrics': metrics_list,
            'H': H
        }

    return total_loss


def optimize_homography(
    ellipses: List[Tuple],
    img_shape: Tuple[int, int, int],
    verbose: bool = False,
    validate_output_size: bool = True
) -> Optional[np.ndarray]:
    """
    Optimize homography to correct perspective distortion.

    Args:
        ellipses: List of 2 ellipse tuples
        img_shape: Image shape (H, W, C)
        verbose: Print optimization progress

    Returns:
        3x3 homography matrix if successful, None if optimization fails
    """
    # Start from identity matrix, optimising only non-translation terms.
    h_init = IDENTITY_HOMOGRAPHY_PARAMS[FREE_HOMOGRAPHY_INDICES]

    if verbose:
        print("[Perspective] Running Powell optimization...")

    try:
        result = minimize(
            perspective_correction_loss,
            h_init,
            args=(ellipses, img_shape),
            method='Powell',
            options={'maxiter': 5000, 'disp': verbose, 'ftol': 1e-10}
        )

        if not result.success:
            if verbose:
                print(f"[Perspective] Optimization failed: {result.message}")
            return None

        # Check if final loss is reasonable
        if result.fun > 1.0:  # Arbitrary threshold
            if verbose:
                print(f"[Perspective] Optimization loss too high: {result.fun:.6f}")
            return None

        H_optimal = build_constrained_homography(result.x)

        if validate_output_size and not is_homography_output_reasonable(H_optimal, img_shape):
            if verbose:
                output_size = homography_output_size(H_optimal, img_shape)
                print(f"[Perspective] Homography output too large: {output_size}")
            return None

        if verbose:
            print(f"[Perspective] Optimization successful! Loss: {result.fun:.6f}")
            print(f"[Perspective] Homography matrix:\n{H_optimal}")

        return H_optimal

    except Exception as e:
        if verbose:
            print(f"[Perspective] Optimization error: {e}")
        return None


def apply_perspective_correction(
    frame: np.ndarray,
    H: np.ndarray,
    fixed_output: bool = False,
    normalize_wheelbase: bool = False,
    wheel_ellipses: Optional[List[Tuple]] = None
) -> np.ndarray:
    """
    Apply homography transformation to correct perspective.

    Args:
        frame: Input frame (BGR)
        H: 3x3 homography matrix

    Returns:
        Corrected frame
    """
    h, w = frame.shape[:2]

    use_fixed_output = fixed_output or normalize_wheelbase

    if not use_fixed_output and not is_homography_output_reasonable(H, frame.shape):
        return frame

    # Calculate output bounds by transforming corners
    corners = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype=np.float32)

    transformed_corners = transform_points(corners, H)
    if not np.all(np.isfinite(transformed_corners)):
        return frame

    min_x, min_y = transformed_corners.min(axis=0)
    max_x, max_y = transformed_corners.max(axis=0)

    if use_fixed_output:
        out_w, out_h = w, h
        if wheel_ellipses:
            wheel_centers = np.array([ellipse[0] for ellipse in wheel_ellipses], dtype=np.float64)
            transformed_wheels = transform_points(wheel_centers, H)
            transformed_anchor = transformed_wheels.mean(axis=0)
            # Put the bike slightly low in the viewport to leave room for the rider.
            target_anchor = np.array([w / 2, h * 0.66])
        else:
            transformed_wheels = None
            source_center = np.array([[w / 2, h / 2]], dtype=np.float64)
            transformed_anchor = transform_points(source_center, H)[0]
            target_anchor = np.array([w / 2, h / 2])

        if not np.all(np.isfinite(transformed_anchor)):
            return frame

        scale = 1.0
        if transformed_wheels is not None and transformed_wheels.shape == (2, 2):
            wheelbase = float(np.linalg.norm(transformed_wheels[1] - transformed_wheels[0]))
            if np.isfinite(wheelbase) and wheelbase > 1:
                target_wheelbase = w * 0.30
                scale = target_wheelbase / wheelbase
                if normalize_wheelbase:
                    scale = min(max(scale, 0.25), 4.0)
                else:
                    scale = min(1.0, scale)
                    scale = max(scale, 0.35)

        T_to_origin = np.array([
            [1, 0, -transformed_anchor[0]],
            [0, 1, -transformed_anchor[1]],
            [0, 0, 1]
        ])
        S = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ])
        T_to_target = np.array([
            [1, 0, target_anchor[0]],
            [0, 1, target_anchor[1]],
            [0, 0, 1]
        ])
        H_shift = T_to_target @ S @ T_to_origin
    else:
        padding = 0
        out_w = int(max_x - min_x) + 2 * padding
        out_h = int(max_y - min_y) + 2 * padding
        if out_w <= 0 or out_h <= 0:
            return frame

        H_shift = np.array([
            [1, 0, -min_x + padding],
            [0, 1, -min_y + padding],
            [0, 0, 1]
        ])

    H_adjusted = H_shift @ H

    # Warp the frame
    corrected = cv2.warpPerspective(
        frame, H_adjusted, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)  # black borders
    )

    return corrected


def normalize_frame_for_video(
    frame: np.ndarray,
    target_size: Tuple[int, int],
    border_value: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize a frame into a fixed video size while preserving aspect ratio.

    Args:
        frame: Input BGR frame
        target_size: Output size as (width, height)
        border_value: Padding color

    Returns:
        Frame with exactly target_size dimensions.
    """
    target_w, target_h = target_size
    h, w = frame.shape[:2]

    if w == target_w and h == target_h:
        return frame

    if w <= 0 or h <= 0 or target_w <= 0 or target_h <= 0:
        return frame

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, frame.shape[2]), border_value, dtype=frame.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def find_wheels_for_confirmation(
    video_path: str,
    wheel_model,
    max_frames_to_try: int = 100,
    verbose: bool = False,
    status_callback=None
) -> Optional[Tuple[np.ndarray, List[Tuple], int]]:
    """
    Find a suitable frame with detected wheels for user confirmation.

    Tries frames sequentially until finding one where:
    - Exactly 2 wheels are detected
    - Ellipse fitting succeeds with good quality

    Args:
        video_path: Path to video file
        wheel_model: YOLO wheel segmentation model
        max_frames_to_try: Maximum number of frames to attempt
        verbose: Print progress
        status_callback: Optional callback function to report status updates

    Returns:
        Tuple of (frame, ellipses, frame_idx) if successful, None if all frames fail
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_try = min(max_frames_to_try, total_frames)

    if verbose:
        print(f"[Perspective] Searching for suitable frame with wheels ({frames_to_try} frames)...")
    if status_callback:
        status_callback(f"Searching {frames_to_try} frames for wheels...")

    wheels_detected_count = 0
    ellipse_fit_failed_count = 0

    for frame_idx in range(frames_to_try):
        ret, frame = cap.read()
        if not ret:
            break

        if verbose and frame_idx % 10 == 0:
            print(f"[Perspective] Trying frame {frame_idx}/{frames_to_try}...")
        if status_callback and frame_idx % 10 == 0:
            status_callback(f"Checking frame {frame_idx}/{frames_to_try}...")

        # Detect wheels
        masks = detect_wheels(wheel_model, frame, verbose=verbose)
        if masks is None:
            continue

        wheels_detected_count += 1
        if verbose:
            print(f"[Perspective] Frame {frame_idx}: Detected {len(masks)} wheels, attempting ellipse fit...")

        # Fit ellipses
        ellipses = fit_ellipses_to_masks(masks, frame.shape, verbose=verbose)
        if ellipses is None:
            ellipse_fit_failed_count += 1
            if verbose:
                print(f"[Perspective] Frame {frame_idx}: Ellipse fitting failed")
            continue

        # Found suitable frame!
        if verbose:
            print(f"[Perspective] ✓ Found suitable frame {frame_idx} with {len(ellipses)} wheels")
        if status_callback:
            status_callback(f"Found suitable wheels in frame {frame_idx}")
        cap.release()
        return (frame.copy(), ellipses, frame_idx)

    cap.release()

    # Provide diagnostic information
    if verbose:
        print(f"[Perspective] ✗ Could not find suitable frame in {frames_to_try} frames")
        print(f"[Perspective]   Frames with wheels detected: {wheels_detected_count}")
        print(f"[Perspective]   Frames where ellipse fit failed: {ellipse_fit_failed_count}")
        print(f"[Perspective]   Possible issues:")
        print(f"[Perspective]     - No wheels visible in video")
        print(f"[Perspective]     - Wheels partially cut off by frame edges")
        print(f"[Perspective]     - Wheel segmentation model not detecting wheels")

    if status_callback:
        status_callback(
            f"Could not find suitable wheels ({wheels_detected_count} detections, "
            f"{ellipse_fit_failed_count} fit failures in {frames_to_try} frames)"
        )

    return None


def compute_homography_from_video(
    video_path: str,
    wheel_model,
    max_frames_to_try: int = 100,
    verbose: bool = False
) -> Optional[np.ndarray]:
    """
    Compute perspective correction homography from video.

    Tries frames sequentially until finding one where:
    - Exactly 2 wheels are detected
    - Ellipse fitting succeeds with good quality
    - Homography optimization converges

    Args:
        video_path: Path to video file
        wheel_model: YOLO wheel segmentation model
        max_frames_to_try: Maximum number of frames to attempt
        verbose: Print progress

    Returns:
        3x3 homography matrix if successful, None if all frames fail
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_try = min(max_frames_to_try, total_frames)

    if verbose:
        print(f"[Perspective] Attempting to compute homography from {frames_to_try} frames...")

    for frame_idx in range(frames_to_try):
        ret, frame = cap.read()
        if not ret:
            break

        if verbose and frame_idx % 10 == 0:
            print(f"[Perspective] Trying frame {frame_idx}/{frames_to_try}...")

        # Detect wheels
        masks = detect_wheels(wheel_model, frame, verbose=False)
        if masks is None:
            continue

        # Fit ellipses
        ellipses = fit_ellipses_to_masks(masks, frame.shape, verbose=False)
        if ellipses is None:
            continue

        # Optimize homography
        H = optimize_homography(ellipses, frame.shape, verbose=False)
        if H is not None:
            if verbose:
                print(f"[Perspective] ✓ Successfully computed homography from frame {frame_idx}")
            cap.release()
            return H

    cap.release()

    if verbose:
        print(f"[Perspective] ✗ Failed to compute homography from {frames_to_try} frames")

    return None


def estimate_homography_for_frame(
    frame: np.ndarray,
    wheel_model,
    verbose: bool = False,
    validate_output_size: bool = True
) -> Optional[np.ndarray]:
    """
    Estimate a perspective-correction homography from one frame.

    Returns None when the frame is not suitable for wheel-based homography
    estimation, allowing callers to fall back to a previous valid homography.
    """
    result = estimate_homography_and_ellipses_for_frame(
        frame,
        wheel_model,
        verbose=verbose,
        validate_output_size=validate_output_size
    )
    return result[0] if result is not None else None


def estimate_homography_and_ellipses_for_frame(
    frame: np.ndarray,
    wheel_model,
    verbose: bool = False,
    validate_output_size: bool = True
) -> Optional[Tuple[np.ndarray, List[Tuple]]]:
    """Estimate a homography and return the ellipses used to compute it."""
    masks = detect_wheels(wheel_model, frame, verbose=verbose)
    if masks is None:
        return None

    ellipses = fit_ellipses_to_masks(masks, frame.shape, verbose=verbose)
    if ellipses is None:
        return None

    H = optimize_homography(
        ellipses,
        frame.shape,
        verbose=verbose,
        validate_output_size=validate_output_size
    )
    if H is None:
        return None

    return H, ellipses


def compute_homography_from_ellipses(
    ellipses: List[Tuple],
    img_shape: Tuple[int, int, int],
    verbose: bool = False,
    validate_output_size: bool = True
) -> Optional[np.ndarray]:
    """
    Compute homography from pre-detected ellipses.

    Args:
        ellipses: List of 2 ellipse tuples
        img_shape: Image shape (H, W, C)
        verbose: Print progress

    Returns:
        3x3 homography matrix if successful, None if optimization fails
    """
    if verbose:
        print("[Perspective] Computing homography from confirmed wheels...")

    H = optimize_homography(
        ellipses,
        img_shape,
        verbose=verbose,
        validate_output_size=validate_output_size
    )

    if H is not None:
        if verbose:
            print("[Perspective] ✓ Homography computed successfully")
    else:
        if verbose:
            print("[Perspective] ✗ Homography computation failed")

    return H


class PerspectiveCorrectionCache:
    """Cache for perspective correction homography matrix."""

    def __init__(self):
        self.H: Optional[np.ndarray] = None
        self.video_path: Optional[str] = None

    def get(self, video_path: str) -> Optional[np.ndarray]:
        """Get cached homography for a video path."""
        if self.video_path == video_path and self.H is not None:
            return self.H
        return None

    def set(self, video_path: str, H: np.ndarray):
        """Cache homography for a video path."""
        self.video_path = video_path
        self.H = H

    def clear(self):
        """Clear the cache."""
        self.H = None
        self.video_path = None
