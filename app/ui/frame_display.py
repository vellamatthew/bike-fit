"""
Widget for displaying a single annotated frame with a compact assessment panel.
"""
import html
import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QFrame, QToolTip, QHBoxLayout
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QResizeEvent, QCursor
from PyQt6.QtGui import QImage, QPixmap


class ClickableLabel(QLabel):
    """A QLabel that can invoke a callback when clicked."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._on_click = None
        self._hover_enabled = False
        self._instant_tooltip = ""

    def set_click_handler(self, fn):
        self._on_click = fn

    def set_hover_enabled(self, enabled: bool):
        self._hover_enabled = enabled
        self._apply_hover_style(False)

    def set_instant_tooltip(self, text: str):
        self._instant_tooltip = text or ""
        super().setToolTip("")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._on_click:
            self._on_click()
            event.accept()
            return
        super().mousePressEvent(event)

    def enterEvent(self, event):
        if self._hover_enabled:
            self._apply_hover_style(True)
            if self._instant_tooltip:
                QToolTip.showText(QCursor.pos(), self._instant_tooltip, self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply_hover_style(False)
        QToolTip.hideText()
        super().leaveEvent(event)

    def event(self, event):
        if event.type() == QEvent.Type.ToolTip:
            return True
        return super().event(event)

    def _apply_hover_style(self, hovered: bool):
        border_color = "#4a5968" if hovered else "#333"
        background = "#2a2a2a" if hovered else "#242424"
        self.setStyleSheet(f"""
            QLabel {{
                background: {background};
                border: 1px solid {border_color};
                border-radius: 4px;
                color: #666;
                font-size: 11px;
            }}
            QToolTip {{
                background: rgba(56, 56, 56, 242);
                color: #f0f0f0;
                border: 1px solid rgba(255, 255, 255, 18);
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 11px;
            }}
        """)


class FrameDisplayWidget(QWidget):
    """Display a single annotated frame with title and angle stats."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title_text = title
        self._pixmap: QPixmap | None = None
        self._on_frame_click = None
        self._frame_click_enabled = False
        self._frame_click_tooltip = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Frame image (takes all available space)
        self._frame_label = ClickableLabel("No frame")
        self._frame_label.set_hover_enabled(False)
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumSize(400, 225)  # 16:9 aspect ratio, smaller
        self._frame_label.set_click_handler(self._handle_frame_click)
        self._frame_label.setStyleSheet("""
            QLabel {
                background: #242424;
                border: 1px solid #333;
                border-radius: 10px;
                color: #666;
                font-size: 11px;
            }
        """)
        layout.addWidget(self._frame_label, stretch=1)

        self._metrics_frame = QFrame()
        self._metrics_frame.setObjectName("metricsFrame")
        self._metrics_frame.setFixedHeight(200)
        self._metrics_frame.setMinimumHeight(200)
        self._metrics_frame.setVisible(False)
        self._metrics_frame.setStyleSheet("""
            QFrame#metricsFrame {
                background: #242424;
                border: 1px solid #333;
                border-radius: 10px;
            }
        """)
        metrics_layout = QVBoxLayout(self._metrics_frame)
        metrics_layout.setContentsMargins(12, 10, 12, 10)
        metrics_layout.setSpacing(8)

        self._metrics_title = QLabel("Fit Measurements")
        self._metrics_title.setStyleSheet("color: #f0f0f0; font-size: 13px; font-weight: bold;")
        metrics_layout.addWidget(self._metrics_title)

        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(10)

        self._metrics_grid = QGridLayout()
        self._metrics_grid.setHorizontalSpacing(8)
        self._metrics_grid.setVerticalSpacing(8)
        content_row.addLayout(self._metrics_grid, stretch=3)

        self._measurements_card = QFrame()
        self._measurements_card.setStyleSheet("""
            QFrame {
                background: #2b2b2b;
                border: none;
                border-radius: 8px;
            }
        """)
        measurements_layout = QVBoxLayout(self._measurements_card)
        measurements_layout.setContentsMargins(12, 10, 12, 10)
        measurements_layout.setSpacing(8)

        self._measurements_title = QLabel("Estimated Body Measurements")
        self._measurements_title.setStyleSheet("color: #f0f0f0; font-size: 13px; font-weight: bold;")
        self._measurements_title.setWordWrap(True)
        measurements_layout.addWidget(self._measurements_title)

        self._measurements_label = QLabel("Configure your bike to get estimate body measurements here.")
        self._measurements_label.setStyleSheet("color: #7f7f7f; font-size: 11px; line-height: 1.5;")
        self._measurements_label.setWordWrap(True)
        measurements_layout.addWidget(self._measurements_label)
        measurements_layout.addStretch()
        content_row.addWidget(self._measurements_card, stretch=2)

        metrics_layout.addLayout(content_row)
        layout.addWidget(self._metrics_frame)

    def set_frame(self, frame: np.ndarray, frame_number: int, angles: dict, assessment: dict | None = None, side: str | None = None):
        """
        Display annotated frame with statistics.

        Args:
            frame: BGR numpy array
            frame_number: Frame index
            angles: Dict with angle values
            assessment: Optional fit assessment dict
            side: Optional side label
        """
        # Convert frame to QPixmap
        self._pixmap = self._frame_to_pixmap(frame)
        self._update_pixmap()
        self.set_frame_click_enabled(True, "Click to enlarge frame")
        self._metrics_frame.setVisible(assessment is not None)
        self._update_metrics(assessment)

    def clear(self):
        """Clear the displayed frame."""
        self._pixmap = None
        self._frame_label.clear()
        self._frame_label.setText("No frame")
        self.set_frame_click_enabled(False)
        self._metrics_frame.setVisible(True)
        self._show_empty_metrics()
        self.clear_measurements()

    def set_frame_click_handler(self, fn):
        """Register the callback used when the frame preview is clicked."""
        self._on_frame_click = fn

    def set_frame_click_enabled(self, enabled: bool, tooltip: str = ""):
        """Show or hide the subtle click-to-open hint above the frame."""
        self._frame_click_enabled = enabled
        self._frame_click_tooltip = tooltip if enabled else ""
        self._frame_label.set_instant_tooltip(self._frame_click_tooltip)
        self._frame_label.set_hover_enabled(enabled)
        self._frame_label.setCursor(
            Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def get_current_pixmap(self) -> QPixmap | None:
        """Return the currently displayed frame pixmap."""
        if self._pixmap is None:
            return None
        return QPixmap(self._pixmap)

    def resizeEvent(self, event: QResizeEvent):
        """Keep the displayed frame crisp when the widget is resized."""
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self):
        """Scale the stored pixmap to the current label size."""
        if self._pixmap is None:
            return

        scaled_pixmap = self._pixmap.scaled(
            self._frame_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._frame_label.setPixmap(scaled_pixmap)

    def _clear_metrics(self):
        """Remove all metric widgets from the compact info panel."""
        while self._metrics_grid.count():
            item = self._metrics_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_empty_metrics(self):
        """Render a simple empty state for the metrics area."""
        self._clear_metrics()
        empty_state = QLabel("Angle results will appear here after you process a video.")
        empty_state.setStyleSheet("color: #7f7f7f; font-size: 11px;")
        empty_state.setWordWrap(True)
        self._metrics_grid.addWidget(empty_state, 0, 0, 1, 2)

    def set_measurements(self, body_measurements: dict | None):
        """Update the estimated body measurements summary."""
        measurement_labels = [
            ("inseam", "Inseam"),
            ("torso", "Torso"),
            ("arm_reach", "Arm Reach"),
            ("thigh", "Thigh"),
            ("shin", "Shin"),
        ]

        if not body_measurements:
            self.clear_measurements()
            return

        rows = []
        for key, label in measurement_labels:
            value = body_measurements.get(key)
            if value is not None:
                rows.append(f"<b>{html.escape(label)}:</b> {value:.0f} mm")

        if rows:
            self._measurements_label.setText("<br>".join(rows))
            self._measurements_label.setStyleSheet("color: #aaa; font-size: 11px; line-height: 1.5;")
        else:
            self.clear_measurements()

    def clear_measurements(self):
        """Reset the measurements card to its empty state."""
        self._measurements_label.setText("Configure your bike to estimate body measurements here.")
        self._measurements_label.setStyleSheet("color: #7f7f7f; font-size: 11px; line-height: 1.5;")

    def _update_metrics(self, assessment: dict | None):
        """Render the compact assessment panel under the frame preview."""
        self._clear_metrics()

        if not assessment:
            self._metrics_frame.setVisible(True)
            self._show_empty_metrics()
            return

        self._metrics_frame.setVisible(True)
        assessments = assessment.get("assessments", {})
        angle_order = [
            ("knee_extension_bdc", "Knee Extension", "BDC"),
            ("hip_angle_bdc", "Hip Angle", "BDC"),
            ("elbow_flexion", "Elbow Angle", "Average"),
            ("back_angle", "Back Angle", "Average"),
        ]

        for index, (key, label, context) in enumerate(angle_order):
            data = assessments.get(key)
            if not data:
                continue

            status = data.get("status")
            if status == "in_range":
                color = "#7bd88f"
                icon = "✓"
            elif status == "not_detected":
                color = "#f06a6a"
                icon = "✗"
            else:
                severity = data.get("severity")
                severity_str = severity.value if hasattr(severity, "value") else str(severity)
                if severity_str == "marginal":
                    color = "#f3ba52"
                else:
                    color = "#f06a6a"
                icon = "•"

            measured = data.get("measured_mean")
            target_min = data.get("target_min")
            target_max = data.get("target_max")
            value_text = "Not detected"
            target_text = "No target"
            if measured is not None:
                value_text = f"{measured:.0f}°"
                if target_min is not None and target_max is not None:
                    target_text = f"Target {target_min:.0f}-{target_max:.0f}°"

            card = self._build_metric_card(label, context, icon, value_text, target_text, color)
            row = index // 2
            col = index % 2
            self._metrics_grid.addWidget(card, row, col)

    def _build_metric_card(
        self,
        label: str,
        context: str,
        icon: str,
        value_text: str,
        target_text: str,
        accent_color: str,
    ) -> QFrame:
        """Create a compact tile for a single assessed angle."""
        card = QFrame()
        safe_label = html.escape(label)
        safe_context = html.escape(context)
        safe_value = html.escape(value_text)
        safe_target = html.escape(target_text)
        safe_icon = html.escape(icon)
        card.setStyleSheet(f"""
            QFrame {{
                background: #2b2b2b;
                border: none;
                border-radius: 8px;
            }}
            QLabel#metricLabel {{
                color: #efefef;
                font-size: 10px;
                font-weight: 600;
            }}
            QLabel#metricContext {{
                color: #7f7f7f;
                font-size: 9px;
            }}
            QLabel#metricValue {{
                color: {accent_color};
                font-size: 15px;
                font-weight: 700;
            }}
            QLabel#metricTarget {{
                color: #9a9a9a;
                font-size: 9px;
            }}
            QLabel#metricStatus {{
                color: {accent_color};
                font-size: 10px;
                font-weight: 700;
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(4)

        label_widget = QLabel(safe_label)
        label_widget.setObjectName("metricLabel")
        top_row.addWidget(label_widget)

        context_widget = QLabel(safe_context)
        context_widget.setObjectName("metricContext")
        top_row.addWidget(context_widget)
        top_row.addStretch()

        status_widget = QLabel(safe_icon)
        status_widget.setObjectName("metricStatus")
        top_row.addWidget(status_widget)
        layout.addLayout(top_row)

        value_widget = QLabel(safe_value)
        value_widget.setObjectName("metricValue")
        layout.addWidget(value_widget)

        target_widget = QLabel(safe_target)
        target_widget.setObjectName("metricTarget")
        layout.addWidget(target_widget)

        return card

    @staticmethod
    def _frame_to_pixmap(frame: np.ndarray) -> QPixmap:
        """Convert BGR frame to QPixmap."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _handle_frame_click(self):
        """Open the video only when the preview is currently interactive."""
        if self._frame_click_enabled and self._on_frame_click:
            self._on_frame_click()
