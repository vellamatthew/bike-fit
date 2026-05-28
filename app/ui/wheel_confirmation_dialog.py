"""
Dialog for reviewing detected wheels before continuing.
"""
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent


class WheelConfirmationDialog(QDialog):
    """Dialog showing detected wheels with fitted ellipses for user review."""

    def __init__(
        self,
        frame: np.ndarray,
        ellipses: list,
        parent=None,
        title_text: str | None = None,
        description_text: str | None = None,
        cancel_label: str | None = None,
        confirm_label: str | None = None,
        baseline_label: str | None = None,
        show_measurement_overlay: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle(title_text or "Confirm Wheel Detection")
        self.setModal(True)
        self._frame = frame
        self._ellipses = ellipses
        self._title_text = title_text or "Review Wheel Detection"
        self._description_text = description_text or (
            "If the wheel outlines look wrong, make sure the video shows both wheels fully, "
            "with no occlusion, from a clear side view.\n"
            "The yellow ellipses should be centered on the two bike wheels."
        )
        self._cancel_label = cancel_label or "Cancel"
        self._confirm_label = confirm_label or "Looks Good"
        self._baseline_label = baseline_label or ""
        self._show_measurement_overlay = show_measurement_overlay
        self._pixmap: QPixmap | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel(self._title_text)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Description
        description = QLabel(self._description_text)
        description.setStyleSheet("font-size: 12px; color: #aaa;")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)

        # Image frame
        image_frame = QFrame()
        image_frame.setStyleSheet("""
            QFrame {
                background: #1a1a1a;
                border: 2px solid #444;
                border-radius: 8px;
            }
        """)
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(8, 8, 8, 8)

        # Visualize the wheels
        vis_image = self._visualize_wheels()
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(320, 240)

        # Convert to QPixmap and scale
        self._pixmap = self._numpy_to_pixmap(vis_image)
        self._update_pixmap()

        image_layout.addWidget(self._image_label)
        layout.addWidget(image_frame)

        # Wheel info
        info_text = self._get_wheel_info()
        info_label = QLabel(info_text)
        info_label.setStyleSheet("""
            QLabel {
                background: #2a2a2a;
                border-radius: 6px;
                padding: 12px;
                font-size: 11px;
                color: #aaa;
                font-family: monospace;
            }
        """)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        if self._cancel_label is not None:
            cancel_btn = QPushButton(self._cancel_label)
            cancel_btn.setStyleSheet("""
                QPushButton {
                    background: #3a3a3a;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px 20px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background: #4a4a4a;
                }
            """)
            cancel_btn.clicked.connect(self.reject)
            button_layout.addWidget(cancel_btn)

        if self._confirm_label is not None:
            proceed_btn = QPushButton(self._confirm_label)
            proceed_btn.setStyleSheet("""
                QPushButton {
                    background: #0066cc;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px 20px;
                    font-size: 13px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #0052a3;
                }
            """)
            proceed_btn.clicked.connect(self.accept)
            button_layout.addWidget(proceed_btn)

        layout.addLayout(button_layout)

        # Set reasonable dialog size
        self.resize(900, 800)

    def resizeEvent(self, event: QResizeEvent):
        """Keep the wheel preview scaled to the current dialog size."""
        super().resizeEvent(event)
        self._update_pixmap()

    def _visualize_wheels(self) -> np.ndarray:
        """Create visualization of detected wheels with fitted ellipses."""
        vis_image = self._frame.copy()

        # Convert to RGB for display
        if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        centers = []

        # Draw wheel ellipses first.
        for ellipse in self._ellipses:
            cv2.ellipse(vis_image, ellipse, (255, 255, 0), 3)
            centers.append(tuple(int(v) for v in ellipse[0]))

        if self._show_measurement_overlay and len(centers) >= 2:
            left_center, right_center = sorted(centers[:2], key=lambda pt: pt[0])
            line_y = int((left_center[1] + right_center[1]) / 2)
            start_point = (left_center[0], line_y)
            end_point = (right_center[0], line_y)

            for center in (left_center, right_center):
                cv2.circle(vis_image, center, 6, (255, 255, 0), -1)
                cv2.circle(vis_image, center, 10, (0, 0, 0), 2)

            cv2.line(vis_image, start_point, end_point, (255, 255, 0), 3)

            tick_height = 16
            cv2.line(
                vis_image,
                (start_point[0], line_y - tick_height),
                (start_point[0], line_y + tick_height),
                (255, 255, 0),
                2,
            )
            cv2.line(
                vis_image,
                (end_point[0], line_y - tick_height),
                (end_point[0], line_y + tick_height),
                (255, 255, 0),
                2,
            )

            if self._baseline_label:
                label_text = self._baseline_label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.75
                thickness = 2
                text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                text_x = int((start_point[0] + end_point[0] - text_size[0]) / 2)
                text_y = min(vis_image.shape[0] - 20, line_y + 34)

                bg_x1 = text_x - 8
                bg_y1 = text_y - text_size[1] - 8
                bg_x2 = text_x + text_size[0] + 8
                bg_y2 = text_y + 8
                cv2.rectangle(vis_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.putText(
                    vis_image,
                    label_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 0),
                    thickness,
                )

        return vis_image

    def _get_wheel_info(self) -> str:
        """Get formatted information about detected wheels."""
        info_lines = []
        centers = []

        for i, ellipse in enumerate(self._ellipses):
            center, axes, angle = ellipse
            cx, cy = center
            width, height = axes
            centers.append((cx, cy))

            info_lines.append(
                f"Wheel {i+1}: Center=({cx:.0f}, {cy:.0f}), "
                f"Size=({width:.0f}×{height:.0f}), Angle={angle:.1f}°"
            )

        if len(centers) >= 2:
            left_center, right_center = sorted(centers[:2], key=lambda pt: pt[0])
            horizontal_px = abs(right_center[0] - left_center[0])
            info_lines.append(f"Horizontal center distance: {horizontal_px:.1f} px")

        if self._baseline_label:
            info_lines.append(self._baseline_label)

        return "\n".join(info_lines)

    @staticmethod
    def _numpy_to_pixmap(image: np.ndarray) -> QPixmap:
        """Convert numpy array (RGB) to QPixmap."""
        h, w = image.shape[:2]

        # Ensure RGB format
        if len(image.shape) == 2:
            # Grayscale
            qimg = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            # RGB
            bytes_per_line = 3 * w
            qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        return QPixmap.fromImage(qimg)

    def _update_pixmap(self):
        """Scale the stored pixmap to fit the current preview area."""
        if self._pixmap is None:
            return

        scaled_pixmap = self._pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._image_label.setPixmap(scaled_pixmap)
