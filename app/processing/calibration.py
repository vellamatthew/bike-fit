"""
Physical measurement calibration using wheel detection and known bike geometry.

This module enables conversion from pixel coordinates to real-world measurements
by detecting bike wheels and comparing the measured wheelbase to the known
wheelbase from bike geometry data.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path

from processing.perspective_correction import select_distinct_wheel_ellipses


class CalibrationError(Exception):
    """Raised when calibration fails validation checks."""
    pass


class WheelCalibration:
    """
    Calibrate pixel-to-real-world measurements using wheel detection.

    Uses the known wheelbase (distance between wheel axles) from bike geometry
    to compute a scale factor that converts pixel distances to millimeters.

    Attributes:
        video_path: Path to video file
        wheelbase_mm: Known wheelbase in millimeters from bike geometry
        scale_factor: Computed scale factor (mm per pixel) after calibration
        wheel_centers: Detected wheel center positions in pixels
        frame_used: Frame index used for calibration
    """

    def __init__(self, video_path: str, wheelbase_mm: float, homography: Optional[np.ndarray] = None):
        """
        Initialize calibration with video path and known wheelbase.

        Args:
            video_path: Path to cycling video
            wheelbase_mm: Known wheelbase from bike geometry (in mm)
            homography: Optional perspective-correction homography to apply
                before wheel detection and wheelbase measurement

        Raises:
            ValueError: If wheelbase is not in reasonable range (800-1500mm)
        """
        if not (800 <= wheelbase_mm <= 1500):
            raise ValueError(
                f"Wheelbase {wheelbase_mm}mm is outside typical range (800-1500mm). "
                f"Please verify bike geometry data."
            )

        self.video_path = video_path
        self.wheelbase_mm = wheelbase_mm
        self.homography = homography
        self.scale_factor: Optional[float] = None
        self.wheel_centers: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.frame_used: Optional[int] = None
        self._wheel_model = None

    def calibrate(
        self,
        max_frames_to_try: int = 100,
        verbose: bool = False
    ) -> Optional[float]:
        """
        Detect wheels and compute scale factor.

        This method:
        1. Loads the wheel segmentation model
        2. Searches video frames for visible wheels
        3. Fits ellipses to wheels to find centers
        4. Computes distance between wheel centers
        5. Calculates scale factor (mm/pixel)
        6. Validates the result

        Args:
            max_frames_to_try: Maximum frames to search for wheels
            verbose: Print detailed progress information

        Returns:
            Scale factor (mm per pixel) if successful, None if failed
        """
        if verbose:
            print(f"[Calibration] Starting calibration with wheelbase={self.wheelbase_mm}mm")
            if self.homography is not None:
                print("[Calibration] Using perspective-corrected frames")

        # Load wheel detection model
        try:
            self._wheel_model = self._load_wheel_model()
        except Exception as e:
            if verbose:
                print(f"[Calibration] Failed to load wheel model: {e}")
            return None

        # Find suitable frame with wheels
        result = self._find_wheels_in_video(max_frames_to_try, verbose)

        if result is None:
            if verbose:
                print(f"[Calibration] Failed to detect wheels in video")
            return None

        frame_idx, wheel_centers, wheel_ellipses = result
        self.frame_used = frame_idx
        self.wheel_centers = wheel_centers

        if verbose:
            print(f"[Calibration] Detected wheels in frame {frame_idx}")
            print(f"[Calibration]   Left wheel center: {wheel_centers[0]}")
            print(f"[Calibration]   Right wheel center: {wheel_centers[1]}")

        # Compute scale factor
        pixel_distance = np.linalg.norm(wheel_centers[1] - wheel_centers[0])
        self.scale_factor = self.wheelbase_mm / pixel_distance

        if verbose:
            print(f"[Calibration]   Pixel distance: {pixel_distance:.1f}px")
            print(f"[Calibration]   Scale factor: {self.scale_factor:.4f} mm/px")

        # Validate calibration
        try:
            self._validate_calibration(wheel_ellipses, verbose)
        except CalibrationError as e:
            if verbose:
                print(f"[Calibration] Validation failed: {e}")
            # Reset calibration
            self.scale_factor = None
            self.wheel_centers = None
            return None

        if verbose:
            print(f"[Calibration] ✓ Calibration successful!")

        return self.scale_factor

    def _load_wheel_model(self):
        """Load the YOLO wheel segmentation model."""
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

        return YOLO(model_path)

    def _find_wheels_in_video(
        self,
        max_frames: int,
        verbose: bool
    ) -> Optional[Tuple[int, Tuple[np.ndarray, np.ndarray], Tuple]]:
        """
        Search video for frame with detectable wheels.

        Returns:
            Tuple of (frame_idx, (left_center, right_center), (left_ellipse, right_ellipse))
            or None if no suitable frame found
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_try = min(max_frames, total_frames)

        if verbose:
            print(f"[Calibration] Searching {frames_to_try} frames for wheels...")

        for frame_idx in range(frames_to_try):
            ret, frame = cap.read()
            if not ret:
                break

            if verbose and frame_idx % 10 == 0:
                print(f"[Calibration] Checking frame {frame_idx}/{frames_to_try}...")

            frame = self._prepare_frame(frame)

            # Detect wheels
            result = self._detect_wheels_in_frame(frame, verbose=False)

            if result is not None:
                left_center, right_center, left_ellipse, right_ellipse = result
                cap.release()

                if verbose:
                    print(f"[Calibration] ✓ Found suitable wheels in frame {frame_idx}")

                return frame_idx, (left_center, right_center), (left_ellipse, right_ellipse)

        cap.release()
        return None

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply perspective correction before calibration when available."""
        if self.homography is None:
            return frame

        from processing.perspective_correction import apply_perspective_correction
        return apply_perspective_correction(frame, self.homography)

    def _detect_wheels_in_frame(
        self,
        frame: np.ndarray,
        verbose: bool = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple, Tuple]]:
        """
        Detect exactly 2 wheels in a frame and return their centers.

        Returns:
            Tuple of (left_center, right_center, left_ellipse, right_ellipse)
            or None if detection failed
        """
        # Run wheel detection
        results = self._wheel_model(frame, verbose=False)
        result = results[0]

        if result.masks is None or len(result.masks.data) < 2:
            return None

        masks = result.masks.data.cpu().numpy()

        # Fit ellipses to masks
        candidates = []
        h, w = frame.shape[:2]

        for mask in masks:
            # Resize mask to frame size
            mask_resized = cv2.resize(mask, (w, h))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                return None

            largest_contour = max(contours, key=cv2.contourArea)

            if len(largest_contour) < 5:
                return None

            # Check if wheel touches edge (unreliable)
            x, y, bw, bh = cv2.boundingRect(largest_contour)
            touches_edge = (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)

            if touches_edge:
                return None

            # Fit ellipse
            try:
                ellipse = cv2.fitEllipse(largest_contour)
                candidates.append((ellipse, float(mask.sum())))
            except:
                return None

        ellipses = select_distinct_wheel_ellipses(
            candidates,
            verbose=verbose,
            log_prefix="[Calibration]"
        )

        if ellipses is None:
            return None

        if len(ellipses) != 2:
            return None

        # Extract centers
        left_center = np.array(ellipses[0][0])
        right_center = np.array(ellipses[1][0])

        return left_center, right_center, ellipses[0], ellipses[1]

    def _validate_calibration(self, wheel_ellipses: Tuple, verbose: bool):
        """
        Validate the calibration by checking if wheels are similar in size.

        Args:
            wheel_ellipses: Tuple of (left_ellipse, right_ellipse)
            verbose: Print validation details

        Raises:
            CalibrationError: If validation fails
        """
        # Log computed wheel diameters before the size-ratio check.
        for i, ellipse in enumerate(wheel_ellipses):
            center, axes, angle = ellipse
            avg_diameter_px = (axes[0] + axes[1]) / 2
            computed_diameter_mm = avg_diameter_px * self.scale_factor

            if verbose:
                print(f"[Calibration] Wheel {i+1} computed diameter: {computed_diameter_mm:.1f}mm")

        # Check if wheels are similar in size (should be equal in reality)
        left_diameter_px = (wheel_ellipses[0][1][0] + wheel_ellipses[0][1][1]) / 2
        right_diameter_px = (wheel_ellipses[1][1][0] + wheel_ellipses[1][1][1]) / 2

        size_ratio = max(left_diameter_px, right_diameter_px) / min(left_diameter_px, right_diameter_px)

        if verbose:
            print(f"[Calibration] Wheel size ratio: {size_ratio:.2f}")

        # Only warn if extreme difference (more than 50% - indicates perspective issues)
        if size_ratio > 1.5:
            if verbose:
                print(f"[Calibration] ⚠ Warning: Wheels differ significantly in size (ratio: {size_ratio:.2f})")
                print(f"[Calibration] This may indicate perspective distortion, but proceeding anyway")
        else:
            if verbose:
                print(f"[Calibration] ✓ Wheel sizes are consistent (ratio: {size_ratio:.2f})")

    def pixels_to_mm(self, distance_pixels: float) -> Optional[float]:
        """
        Convert pixel distance to millimeters.

        Args:
            distance_pixels: Distance in pixels

        Returns:
            Distance in millimeters, or None if not calibrated
        """
        if self.scale_factor is None:
            return None
        return distance_pixels * self.scale_factor

    def mm_to_pixels(self, distance_mm: float) -> Optional[float]:
        """
        Convert millimeter distance to pixels.

        Args:
            distance_mm: Distance in millimeters

        Returns:
            Distance in pixels, or None if not calibrated
        """
        if self.scale_factor is None:
            return None
        return distance_mm / self.scale_factor

    def get_calibration_info(self) -> Dict:
        """
        Get calibration information as a dictionary.

        Returns:
            Dictionary with calibration details
        """
        return {
            'calibrated': self.scale_factor is not None,
            'wheelbase_mm': self.wheelbase_mm,
            'scale_factor': self.scale_factor,
            'frame_used': self.frame_used,
            'wheel_centers': self.wheel_centers
        }


def detect_wheels_for_calibration(
    video_path: str,
    homography: Optional[np.ndarray] = None,
    max_frames_to_try: int = 100,
    verbose: bool = False
) -> Optional[Tuple[int, np.ndarray, Tuple]]:
    """
    Detect wheels in a video for calibration purposes (standalone function).

    This is a convenience function that can be used independently of the
    WheelCalibration class for testing or UI preview purposes.

    Args:
        video_path: Path to video file
        max_frames_to_try: Maximum frames to search
        verbose: Print progress information

    Returns:
        Tuple of (frame_index, frame_with_wheels, (ellipse1, ellipse2))
        or None if detection failed
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        if verbose:
            print("[Calibration] ultralytics package not available")
        return None

    # Load model
    project_root = Path.cwd()
    model_path = project_root / "wheel.pt"

    if not model_path.exists():
        if verbose:
            print(f"[Calibration] Wheel model not found at {model_path}")
        return None

    model = YOLO(model_path)

    # Search video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_try = min(max_frames_to_try, total_frames)

    for frame_idx in range(frames_to_try):
        ret, frame = cap.read()
        if not ret:
            break

        if homography is not None:
            from processing.perspective_correction import apply_perspective_correction
            frame = apply_perspective_correction(frame, homography)

        # Try to detect wheels
        results = model(frame, verbose=False)
        result = results[0]

        if result.masks is None or len(result.masks.data) < 2:
            continue

        masks = result.masks.data.cpu().numpy()

        # Attempt to fit ellipses for all candidates before choosing a pair.
        h, w = frame.shape[:2]
        candidates = []

        for mask in masks:
            mask_resized = cv2.resize(mask, (w, h))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)

                if len(largest_contour) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(largest_contour)
                        candidates.append((ellipse, float(mask.sum())))
                    except:
                        pass

        ellipses = select_distinct_wheel_ellipses(
            candidates,
            verbose=verbose,
            log_prefix="[Calibration]"
        )

        if ellipses is not None:
            cap.release()
            return frame_idx, frame.copy(), tuple(ellipses)

    cap.release()
    return None
