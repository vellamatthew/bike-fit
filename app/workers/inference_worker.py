import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from typing import Optional

from inference.yolo_pose import infer
from processing.angles import compute_angles, compute_angles_for_side
from processing.annotate import draw_skeleton


class VideoWorker(QThread):
    """
    Processes a video file frame by frame.
    Emits:
        total_frames(int)  — total number of frames in the video
        progress(int)  — 0–100
        frame_ready(np.ndarray, dict)  — annotated frame + angles for display
        finished(list[dict])  — per-frame angle records for the graph
        error(str)
        angle_plot_ready(list, list): raw angles, smoothed angles for plotting
        perspective_status(str)  — status updates for perspective correction
        wheel_confirmation_request(np.ndarray, list, tuple)  — request user confirmation for wheels
    """
    total_frames = pyqtSignal(int)
    progress    = pyqtSignal(int)
    frame_ready = pyqtSignal(np.ndarray, dict)
    finished    = pyqtSignal(list)
    error       = pyqtSignal(str)
    angle_plot_ready = pyqtSignal(list, list)
    perspective_status = pyqtSignal(str)
    wheel_confirmation_request = pyqtSignal(np.ndarray, list, tuple)  # frame, ellipses, img_shape
    homography_computed = pyqtSignal(object)  # np.ndarray or None - emits the computed homography matrix
    video_exported = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        apply_perspective_correction: bool = False,
        export_video: bool = False,
        perspective_mode: str = "single",
        fixed_perspective_warp: bool = False,
        normalize_wheelbase_view: bool = False
    ):
        super().__init__()
        self.video_path = video_path
        self.apply_perspective_correction = apply_perspective_correction
        self.export_video = export_video
        self.perspective_mode = perspective_mode
        self.fixed_perspective_warp = fixed_perspective_warp
        self.normalize_wheelbase_view = normalize_wheelbase_view
        self._stop = False
        self._homography: Optional[np.ndarray] = None
        self._last_wheel_ellipses: Optional[list] = None
        self._wheel_model = None
        self._user_confirmed_correction: Optional[bool] = None
        self._confirmed_ellipses: Optional[list] = None
        self._confirmed_img_shape: Optional[tuple] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._frame_limit_start: Optional[int] = None  # For trimming long videos
        self._frame_limit_end: Optional[int] = None

    def stop(self):
        self._stop = True
        # Break any pending confirmation wait so the thread can exit promptly.
        if self._user_confirmed_correction is None:
            self._user_confirmed_correction = False

    def confirm_wheels(self, confirmed: bool, ellipses: list = None, img_shape: tuple = None):
        """Called from main thread when user confirms/rejects wheel detection."""
        self._user_confirmed_correction = confirmed
        if confirmed:
            self._confirmed_ellipses = ellipses
            self._confirmed_img_shape = img_shape

    def set_frame_limit(self, start_frame: int, end_frame: int):
        """Set frame range to process (for trimming long videos)."""
        self._frame_limit_start = start_frame
        self._frame_limit_end = end_frame
        print(f"[VideoWorker] Frame limit set: {start_frame} to {end_frame}")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"Cannot open video: {self.video_path}")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames.emit(total)
        records = []
        frame_idx = 0
        per_frame_homography = (
            self.apply_perspective_correction
            and self.perspective_mode == "per_frame"
        )
        per_frame_estimated_count = 0
        per_frame_fallback_count = 0
        per_frame_uncorrected_count = 0

        # Compute perspective correction homography if enabled
        if self.apply_perspective_correction:
            if per_frame_homography:
                self._homography = self._initialize_per_frame_perspective_correction()
            else:
                self._homography = self._compute_perspective_correction()
            if self._homography is None:
                if per_frame_homography and self._wheel_model is not None:
                    self.perspective_status.emit("Per-frame perspective correction initialized")
                else:
                    self.perspective_status.emit("Perspective correction failed - processing without correction")
            else:
                if per_frame_homography:
                    self.perspective_status.emit("Per-frame perspective correction initialized")
                else:
                    self.perspective_status.emit("Perspective correction applied")

            # Emit the computed homography (or None) to the main thread
            self.homography_computed.emit(self._homography)

        # Setup video writer if export is enabled
        output_path = None
        export_frame_size = None
        if self.export_video:
            import os
            from pathlib import Path

            # Create output directory
            output_dir = Path("output_videos")
            output_dir.mkdir(exist_ok=True)

            # Generate output filename
            input_name = Path(self.video_path).stem
            suffix = "_corrected" if self._homography is not None else "_annotated"
            output_path = output_dir / f"{input_name}{suffix}.mp4"

            print(f"[Export] Will save annotated video to: {output_path}")

        try:
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    break

                # Check if we're trimming and skip frames outside the range
                if self._frame_limit_start is not None and frame_idx < self._frame_limit_start:
                    frame_idx += 1
                    continue

                if self._frame_limit_end is not None and frame_idx >= self._frame_limit_end:
                    print(f"[VideoWorker] Reached frame limit end ({self._frame_limit_end}), stopping")
                    break

                homography_used = None
                wheel_ellipses_used = self._last_wheel_ellipses

                # Apply perspective correction if homography was computed
                if per_frame_homography and self._wheel_model is not None:
                    estimated = self._estimate_frame_homography(frame)
                    if estimated is not None:
                        estimated_homography, estimated_ellipses = estimated
                        self._homography = estimated_homography
                        self._last_wheel_ellipses = estimated_ellipses
                        homography_used = estimated_homography
                        wheel_ellipses_used = estimated_ellipses
                        per_frame_estimated_count += 1
                    elif self._homography is not None:
                        homography_used = self._homography
                        wheel_ellipses_used = self._last_wheel_ellipses
                        per_frame_fallback_count += 1
                    else:
                        per_frame_uncorrected_count += 1

                    if homography_used is not None:
                        frame = self._apply_correction(frame, homography_used, wheel_ellipses_used)

                    if frame_idx % 25 == 0:
                        self.perspective_status.emit(
                            "Per-frame correction: "
                            f"{per_frame_estimated_count} estimated, "
                            f"{per_frame_fallback_count} fallback, "
                            f"{per_frame_uncorrected_count} uncorrected"
                        )
                elif self._homography is not None:
                    homography_used = self._homography
                    frame = self._apply_correction(frame, homography_used, wheel_ellipses_used)

                result = infer(frame)
                angles = {}
                angles_left = {}
                angles_right = {}
                keypoints_data = None
                side = None
                annotated = frame.copy()

                if result["xy"] is not None:
                    xy = result["xy"]
                    kpts = result["kpts"]

                    # Auto-detect best side
                    result_dict = compute_angles(kpts)
                    angles = result_dict["angles"]
                    side = result_dict["side"]

                    # Compute angles for BOTH sides
                    angles_left = compute_angles_for_side(kpts, "left")
                    angles_right = compute_angles_for_side(kpts, "right")

                    if self.normalize_wheelbase_view:
                        annotated = frame.copy()
                    else:
                        annotated = draw_skeleton(frame, xy, angles, side)

                    # Save keypoints as list for JSON serialization
                    keypoints_data = xy.tolist()

                # Initialize video writer on first annotated frame (if export enabled)
                if self.export_video and self._video_writer is None and annotated is not None:
                    h, w = annotated.shape[:2]
                    export_frame_size = (w, h)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self._video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
                    print(f"[Export] Video writer initialized: {w}x{h} @ {fps} fps")

                # Write annotated frame to video if export enabled
                if self.export_video and self._video_writer is not None:
                    if export_frame_size is not None:
                        from processing.perspective_correction import normalize_frame_for_video
                        annotated = normalize_frame_for_video(annotated, export_frame_size)
                    self._video_writer.write(annotated)

                # Store angles with prefixes to distinguish sides
                from processing.perspective_correction import serialize_ellipses

                record = {
                    "frame": frame_idx,
                    "keypoints": keypoints_data,
                    "detected_side": side,
                    "homography": homography_used.tolist() if homography_used is not None else None,
                    "wheel_ellipses": serialize_ellipses(wheel_ellipses_used),
                }

                # Add detected side angles (default/active)
                for key, val in angles.items():
                    record[key] = val

                # Add left side angles
                for key, val in angles_left.items():
                    record[f"left_{key}"] = val

                # Add right side angles
                for key, val in angles_right.items():
                    record[f"right_{key}"] = val

                records.append(record)
                self.frame_ready.emit(annotated, angles)

                # Calculate progress based on trimmed total if applicable
                if total > 0:
                    if self._frame_limit_end is not None:
                        # When trimming, calculate progress based on the trimmed range
                        effective_total = self._frame_limit_end - (self._frame_limit_start or 0)
                        effective_current = frame_idx - (self._frame_limit_start or 0)
                        self.progress.emit(int(effective_current / effective_total * 100))
                    else:
                        # Normal progress calculation
                        self.progress.emit(int(frame_idx / total * 100))

                frame_idx += 1

        finally:
            cap.release()

            # Close video writer if it was opened
            if self._video_writer is not None:
                self._video_writer.release()
                print(f"[Export] Video saved to: {output_path}")
                self.perspective_status.emit(f"Video exported to: {output_path}")
                if output_path is not None:
                    self.video_exported.emit(str(output_path))

        # Apply Savitzky-Golay smoothing to angles
        records = self._smooth_angles(records)

        self.progress.emit(100)
        self.finished.emit(records)

    def _smooth_angles(self, records: list[dict]) -> list[dict]:
        """
        Apply Savitzky-Golay filter to smooth angle data.
        Creates smoothed copies while preserving raw angles.
        Smooths angles for both left and right sides.
        """
        angle_keys = ["knee_flexion", "knee_extension", "hip_flexion", "elbow_flexion", "back_angle"]

        # Create keys for detected side, left, and right
        all_keys = angle_keys.copy()
        for key in angle_keys:
            all_keys.append(f"left_{key}")
            all_keys.append(f"right_{key}")

        # Extract angle arrays
        angle_arrays = {key: [] for key in all_keys}
        for rec in records:
            for key in all_keys:
                angle_arrays[key].append(rec.get(key))

        # Apply Savitzky-Golay filter (window=9, polynomial=3)
        # Window of 9 frames provides good smoothing without too much lag
        window_length = 9
        polyorder = 3

        smoothed_arrays = {}
        for key, values in angle_arrays.items():
            # Convert None to NaN for processing
            values_array = np.array([v if v is not None else np.nan for v in values])

            # Only smooth if we have enough valid data points
            valid_mask = ~np.isnan(values_array)
            if np.sum(valid_mask) >= window_length:
                # Interpolate NaN values for smoothing
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    # Simple linear interpolation for gaps
                    interpolated = values_array.copy()
                    for i in range(len(values_array)):
                        if np.isnan(values_array[i]) and len(valid_indices) > 0:
                            # Use nearest valid value
                            nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - i))]
                            interpolated[i] = values_array[nearest_idx]

                    # Apply filter
                    try:
                        smoothed = savgol_filter(interpolated, window_length, polyorder)
                        # Restore NaN where original was None
                        smoothed[~valid_mask] = np.nan
                        smoothed_arrays[key] = smoothed
                    except:
                        # If filtering fails, use original
                        smoothed_arrays[key] = values_array
                else:
                    smoothed_arrays[key] = values_array
            else:
                smoothed_arrays[key] = values_array

        # Add smoothed angles to records
        for i, rec in enumerate(records):
            for key in all_keys:
                smoothed_val = smoothed_arrays[key][i]
                # Convert NaN back to None
                rec[f"{key}_smoothed"] = None if np.isnan(smoothed_val) else float(smoothed_val)

        self.angle_plot_ready.emit(angle_arrays["knee_flexion"], smoothed_arrays["knee_flexion"].tolist())

        return records

    def _compute_perspective_correction(self) -> Optional[np.ndarray]:
        """
        Compute perspective correction homography from video with user confirmation.

        Returns:
            3x3 homography matrix if successful, None otherwise
        """
        try:
            from processing.perspective_correction import (
                load_wheel_segmentation_model,
                find_wheels_for_confirmation,
                compute_homography_from_ellipses
            )

            self.perspective_status.emit("Loading wheel detection model...")

            # Load wheel segmentation model
            try:
                wheel_model = load_wheel_segmentation_model()
                self._wheel_model = wheel_model
            except FileNotFoundError as e:
                self.perspective_status.emit(f"Wheel model not found")
                return None
            except Exception as e:
                self.perspective_status.emit(f"Error loading wheel model: {e}")
                return None

            self.perspective_status.emit("Searching for wheels in video...")

            # Find a suitable frame with wheels for user confirmation
            result = find_wheels_for_confirmation(
                self.video_path,
                wheel_model,
                max_frames_to_try=100,
                verbose=True,
                status_callback=lambda msg: self.perspective_status.emit(msg)
            )

            if result is None:
                self.perspective_status.emit("Could not find suitable wheels in video")
                return None

            frame, ellipses, frame_idx = result
            img_shape = frame.shape

            self.perspective_status.emit(f"Found wheels in frame {frame_idx}. Awaiting confirmation...")

            # Emit signal to main thread requesting confirmation
            self.wheel_confirmation_request.emit(frame, ellipses, img_shape)

            # Wait for user response (with timeout)
            timeout_seconds = 60  # 1 minute timeout
            wait_interval_ms = 100
            total_wait_ms = 0

            while (
                not self._stop
                and self._user_confirmed_correction is None
                and total_wait_ms < timeout_seconds * 1000
            ):
                self.msleep(wait_interval_ms)
                total_wait_ms += wait_interval_ms

            if self._stop:
                return None

            if self._user_confirmed_correction is None:
                self.perspective_status.emit("Confirmation timeout - skipping perspective correction")
                return None

            if not self._user_confirmed_correction:
                self.perspective_status.emit("User cancelled perspective correction")
                return None

            # User confirmed - compute homography
            self.perspective_status.emit("Computing perspective correction...")

            H = compute_homography_from_ellipses(
                self._confirmed_ellipses,
                self._confirmed_img_shape,
                verbose=True,
                validate_output_size=not self.fixed_perspective_warp
            )

            if H is None:
                self.perspective_status.emit("Homography optimization failed")
                return None

            self._last_wheel_ellipses = self._confirmed_ellipses
            return H

        except ImportError as e:
            self.perspective_status.emit(f"Missing dependency for perspective correction: {e}")
            return None
        except Exception as e:
            self.perspective_status.emit(f"Perspective correction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _initialize_per_frame_perspective_correction(self) -> Optional[np.ndarray]:
        """
        Load the wheel model for automatic per-frame perspective correction.

        Unlike single-frame correction, this mode does not block on a wheel
        confirmation dialog. It will estimate a fresh homography in the main
        processing loop whenever a frame is suitable.
        """
        try:
            from processing.perspective_correction import load_wheel_segmentation_model

            self.perspective_status.emit("Loading wheel detection model...")
            try:
                self._wheel_model = load_wheel_segmentation_model()
            except FileNotFoundError:
                self.perspective_status.emit("Wheel model not found")
                return None
            except Exception as e:
                self.perspective_status.emit(f"Error loading wheel model: {e}")
                return None

            return None

        except ImportError as e:
            self.perspective_status.emit(f"Missing dependency for perspective correction: {e}")
            return None

    def _estimate_frame_homography(self, frame: np.ndarray) -> Optional[tuple[np.ndarray, list]]:
        """Estimate a fresh homography for a single frame."""
        if self._wheel_model is None:
            return None

        from processing.perspective_correction import estimate_homography_and_ellipses_for_frame
        return estimate_homography_and_ellipses_for_frame(
            frame,
            self._wheel_model,
            verbose=False,
            validate_output_size=not self.fixed_perspective_warp
        )

    def _apply_correction(
        self,
        frame: np.ndarray,
        homography: Optional[np.ndarray] = None,
        wheel_ellipses: Optional[list] = None
    ) -> np.ndarray:
        """
        Apply perspective correction to a frame.

        Args:
            frame: Input frame

        Returns:
            Corrected frame
        """
        from processing.perspective_correction import apply_perspective_correction
        H = homography if homography is not None else self._homography
        return apply_perspective_correction(
            frame,
            H,
            fixed_output=self.fixed_perspective_warp,
            normalize_wheelbase=self.normalize_wheelbase_view,
            wheel_ellipses=wheel_ellipses
        )
