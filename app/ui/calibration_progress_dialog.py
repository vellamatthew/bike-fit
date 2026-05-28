"""
Dialog showing progress during calibration wheel detection.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont


class CalibrationWorker(QThread):
    """Background thread for running calibration."""

    finished = pyqtSignal(object)  # Emits scale_factor (float or None)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, video_path: str, wheelbase_mm: float, homography_matrix=None):
        super().__init__()
        self.video_path = video_path
        self.wheelbase_mm = wheelbase_mm
        self.homography_matrix = homography_matrix

    def run(self):
        try:
            from processing.calibration import WheelCalibration
            import cv2
            import numpy as np

            self.status_update.emit("Initializing calibration...")

            # If we have a homography matrix, we need to apply it to frames before detection
            if self.homography_matrix is not None:
                self.status_update.emit("Using perspective-corrected frames...")
                print("[Calibration] Using perspective correction for calibration")

                # Read video and find wheels with correction applied
                from processing.perspective_correction import apply_perspective_correction

                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    self.error.emit("Could not open video")
                    return

                # Load wheel model
                try:
                    from ultralytics import YOLO
                    from pathlib import Path
                    model_path = Path.cwd() / "wheel.pt"
                    if not model_path.exists():
                        self.error.emit("Wheel model not found")
                        return
                    wheel_model = YOLO(model_path)
                except Exception as e:
                    self.error.emit(f"Failed to load wheel model: {e}")
                    return

                # Search for wheels in corrected frames
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames_to_try = min(100, total_frames)

                wheel_centers = None
                wheel_ellipses = None

                for frame_idx in range(frames_to_try):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Apply perspective correction
                    corrected_frame = apply_perspective_correction(frame, self.homography_matrix)

                    # Detect wheels in corrected frame
                    results = wheel_model(corrected_frame, verbose=False)
                    result = results[0]

                    if result.masks is None or len(result.masks.data) < 2:
                        continue

                    # Process masks to get ellipses
                    masks = result.masks.data.cpu().numpy()
                    if len(masks) > 2:
                        mask_areas = [mask.sum() for mask in masks]
                        largest_indices = np.argsort(mask_areas)[-2:]
                        masks = [masks[i] for i in largest_indices]

                    h, w = corrected_frame.shape[:2]
                    ellipses = []

                    for mask in masks:
                        mask_resized = cv2.resize(mask, (w, h))
                        mask_uint8 = (mask_resized * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if len(contours) == 0:
                            continue

                        largest_contour = max(contours, key=cv2.contourArea)
                        if len(largest_contour) < 5:
                            continue

                        try:
                            ellipse = cv2.fitEllipse(largest_contour)
                            ellipses.append(ellipse)
                        except:
                            continue

                    if len(ellipses) == 2:
                        # Found wheels! Extract centers
                        ellipses = sorted(ellipses, key=lambda e: e[0][0])
                        left_center = np.array(ellipses[0][0])
                        right_center = np.array(ellipses[1][0])
                        wheel_centers = (left_center, right_center)
                        wheel_ellipses = tuple(ellipses)

                        print(f"[Calibration] Found wheels in corrected frame {frame_idx}")
                        break

                cap.release()

                if wheel_centers is None:
                    self.error.emit("Could not detect wheels in corrected frames")
                    return

                # Compute scale factor from corrected frame measurements
                pixel_distance = np.linalg.norm(wheel_centers[1] - wheel_centers[0])
                scale_factor = self.wheelbase_mm / pixel_distance

                print(f"[Calibration] Corrected pixel distance: {pixel_distance:.1f}px")
                print(f"[Calibration] Scale factor with correction: {scale_factor:.4f} mm/px")

            else:
                # No perspective correction - use standard calibration
                calibration = WheelCalibration(self.video_path, self.wheelbase_mm)

                self.status_update.emit("Searching for wheels in video...")

                # Run calibration with verbose output
                scale_factor = calibration.calibrate(
                    max_frames_to_try=100,
                    verbose=True
                )

            if scale_factor is not None:
                self.status_update.emit(
                    f"✓ Calibration successful! Scale: {scale_factor:.4f} mm/px"
                )
            else:
                self.status_update.emit("✗ Could not detect wheels")

            self.finished.emit(scale_factor)

        except Exception as e:
            self.error.emit(str(e))


class CalibrationProgressDialog(QDialog):
    """Dialog showing progress during calibration."""

    def __init__(self, video_path: str, wheelbase_mm: float, homography_matrix=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrating...")
        self.setModal(True)
        self._video_path = video_path
        self._wheelbase_mm = wheelbase_mm
        self._homography_matrix = homography_matrix
        self._worker: CalibrationWorker | None = None
        self._scale_factor: float | None = None
        self._build_ui()
        self._start_calibration()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(16)

        # Title
        title = QLabel("Calibrating Physical Measurements")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #fff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Status label
        self._status_label = QLabel("Initializing...")
        self._status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        self._status_label.setMinimumHeight(40)
        layout.addWidget(self._status_label)

        # Progress bar (indeterminate)
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(0)  # Indeterminate
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 4px;
                background: #1a1a1a;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background: #0aa;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self._progress_bar)

        self.setMinimumWidth(400)
        self.setMaximumWidth(500)

    def _start_calibration(self):
        """Start calibration worker thread."""
        self._worker = CalibrationWorker(self._video_path, self._wheelbase_mm, self._homography_matrix)
        self._worker.status_update.connect(self._on_status_update)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_status_update(self, status: str):
        """Update status label."""
        self._status_label.setText(status)

    def _on_finished(self, scale_factor: float | None):
        """Handle calibration completion."""
        self._scale_factor = scale_factor

        if scale_factor is not None:
            # Let the user confirm wheel detection before keeping calibration.
            if self._show_wheel_visualization():
                self.accept()
            else:
                self._scale_factor = None
                self.reject()
        else:
            self.reject()

    def _show_wheel_visualization(self) -> bool:
        """Let the user confirm wheel detection before keeping calibration."""
        try:
            # Try to get the frame with detected wheels from the calibration
            from processing.calibration import detect_wheels_for_calibration

            result = detect_wheels_for_calibration(self._video_path, max_frames_to_try=100, verbose=False)

            if result is not None:
                frame_idx, frame, ellipses = result

                # Show wheel confirmation dialog
                from ui.wheel_confirmation_dialog import WheelConfirmationDialog
                dialog = WheelConfirmationDialog(
                    frame,
                    ellipses,
                    self,
                    title_text="Review Wheel Detection",
                    description_text=(
                        "If the wheel outlines look wrong, make sure the video shows both wheels fully, "
                        "with no occlusion, from a clear side view.\n"
                        "The yellow ellipses should be centered on the two bike wheels."
                    ),
                    cancel_label="Cancel Calibration",
                    confirm_label="Looks Good",
                )

                return dialog.exec() == dialog.DialogCode.Accepted
        except Exception as e:
            print(f"[Calibration] Could not show wheel visualization: {e}")

        return False

    def _on_error(self, error_msg: str):
        """Handle calibration error."""
        self._status_label.setText(f"Error: {error_msg}")
        self._status_label.setStyleSheet("color: #f44; font-size: 11px;")
        self._progress_bar.setVisible(False)

        # Auto-close after showing error
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, self.reject)

    def get_scale_factor(self) -> float | None:
        """Get the computed scale factor."""
        return self._scale_factor
