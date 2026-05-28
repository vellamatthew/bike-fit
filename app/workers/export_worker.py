"""Background worker for exporting annotated videos."""

import cv2
import numpy as np
import os
import time
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional


class ExportWorker(QThread):
    """
    Re-renders video with updated angle overlays in the background.

    Emits:
        progress(int) - 0-100 progress
        finished(str) - path to exported video
        error(str) - error message
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        output_path: str,
        angle_data: list,
        assessment: dict,
        side: str,
        homography_matrix: Optional[np.ndarray] = None,
        per_frame_homographies: Optional[dict[int, np.ndarray]] = None,
        per_frame_wheel_ellipses: Optional[dict[int, list]] = None,
        fixed_perspective_warp: bool = False,
        normalize_wheelbase_view: bool = False
    ):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.angle_data = angle_data
        self.assessment = assessment
        self.side = side
        self.homography_matrix = homography_matrix
        self.per_frame_homographies = per_frame_homographies or {}
        self.per_frame_wheel_ellipses = per_frame_wheel_ellipses or {}
        self.fixed_perspective_warp = fixed_perspective_warp
        self.normalize_wheelbase_view = normalize_wheelbase_view
        self._stop = False

    def stop(self):
        """Request the worker to stop."""
        self._stop = True

    def run(self):
        """Process video and export with annotations."""
        from processing.angle_overlay import annotate_frame_with_angles

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"Cannot open video: {self.video_path}")
            return

        writer = None
        temp_output_path = f"{self.output_path}.tmp.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = len(self.angle_data)
        frame_idx = 0
        output_size = None

        try:
            while not self._stop and frame_idx < total_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                # Apply perspective correction if available
                homography = self.per_frame_homographies.get(frame_idx, self.homography_matrix)
                if homography is not None:
                    from processing.perspective_correction import (
                        apply_perspective_correction,
                        deserialize_ellipses
                    )
                    frame = apply_perspective_correction(
                        frame,
                        homography,
                        fixed_output=self.fixed_perspective_warp,
                        normalize_wheelbase=self.normalize_wheelbase_view,
                        wheel_ellipses=deserialize_ellipses(
                            self.per_frame_wheel_ellipses.get(frame_idx)
                        )
                    )

                # Get angle data for this frame
                record = self.angle_data[frame_idx] if frame_idx < len(self.angle_data) else None
                annotated_frame = frame

                if record and record.get("keypoints") is not None and not self.normalize_wheelbase_view:
                    keypoints = np.array(record["keypoints"], dtype=np.float32)
                    frame_angles = self._angles_for_record(record, self.side)
                    viz_side = self.side if self.side in ("left", "right") else record.get("detected_side", "right")

                    annotated_frame = annotate_frame_with_angles(
                        frame,
                        keypoints,
                        frame_angles,
                        self.assessment["assessments"],
                        viz_side,
                        use_assessment_values=False
                    )

                # Initialize writer on first frame
                if writer is None:
                    h, w = annotated_frame.shape[:2]
                    output_size = (w, h)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (w, h))

                if output_size is not None:
                    from processing.perspective_correction import normalize_frame_for_video
                    annotated_frame = normalize_frame_for_video(annotated_frame, output_size)

                writer.write(annotated_frame)
                frame_idx += 1

                # Emit progress
                if total_frames > 0:
                    self.progress.emit(int(frame_idx / total_frames * 100))

        except Exception as e:
            self.error.emit(str(e))
            return
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        # Replace original with temp file if successful
        if not self._stop and writer is not None and os.path.exists(temp_output_path):
            # Retry logic for Windows file locking issues
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    os.replace(temp_output_path, self.output_path)
                    print("[Export] Annotated video updated with angle overlays")
                    self.finished.emit(self.output_path)
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        print(f"[Export] File locked, retrying in 0.5s... ({attempt + 1}/{max_retries})")
                        time.sleep(0.5)
                    else:
                        self.error.emit(f"Cannot replace video file (file may be open in video player): {e}")
                        # Clean up temp file
                        if os.path.exists(temp_output_path):
                            os.remove(temp_output_path)
        elif os.path.exists(temp_output_path):
            # Clean up temp file if stopped
            os.remove(temp_output_path)

    @staticmethod
    def _angles_for_record(record: dict, side: str) -> dict:
        """Extract the relevant angle set for a frame record and selected side."""
        if side == "left":
            prefix = "left_"
        elif side == "right":
            prefix = "right_"
        else:
            prefix = ""

        return {
            "knee_flexion": record.get(f"{prefix}knee_flexion"),
            "knee_extension": record.get(f"{prefix}knee_extension"),
            "hip_flexion": record.get(f"{prefix}hip_flexion"),
            "elbow_flexion": record.get(f"{prefix}elbow_flexion"),
            "back_angle": record.get(f"{prefix}back_angle"),
        }
