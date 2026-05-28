"""Warning dialog for long videos with trim offer."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)


class LongVideoWarningDialog(QDialog):
    """
    Warning dialog shown when video processing will take longer than 2 minutes.
    Offers automatic trimming to speed up analysis.
    """

    def __init__(self, eta_seconds: float, total_frames: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Long Video Detected")
        self.setMinimumWidth(450)
        self.setModal(True)

        self.eta_seconds = eta_seconds
        self.total_frames = total_frames
        self.trim_video = False

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Warning message
        warning_label = QLabel(
            f"This video will take approximately {self._format_time(self.eta_seconds)} to process."
        )
        warning_label.setStyleSheet("color: #ddd; font-size: 13px;")
        warning_label.setWordWrap(True)
        layout.addWidget(warning_label)

        # Explanation
        explanation = QLabel(
            "For bike fit analysis, you only need 10 seconds of video. "
            "We can process just the first 10 seconds to speed this up."
        )
        explanation.setStyleSheet("color: #aaa; font-size: 12px;")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        button_layout.addStretch()

        # Continue button (no trim)
        continue_btn = QPushButton("Continue Anyway")
        continue_btn.setMinimumHeight(36)
        continue_btn.setMinimumWidth(140)
        continue_btn.clicked.connect(self._on_continue)
        continue_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                color: #dddddd;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 400;
            }
            QPushButton:hover {
                background: #333333;
                color: #ffffff;
            }
            QPushButton:pressed {
                background: #383838;
            }
        """)
        button_layout.addWidget(continue_btn)

        # Trim button (recommended)
        trim_btn = QPushButton("Trim to 10s")
        trim_btn.setMinimumHeight(36)
        trim_btn.setMinimumWidth(140)
        trim_btn.clicked.connect(self._on_trim)
        trim_btn.setDefault(True)
        trim_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                color: #dddddd;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 400;
            }
            QPushButton:hover {
                background: #333333;
                color: #ffffff;
            }
            QPushButton:pressed {
                background: #383838;
            }
        """)
        button_layout.addWidget(trim_btn)

        layout.addLayout(button_layout)

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time string."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes} minute{'s' if minutes > 1 else ''} {secs} seconds"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours} hour{'s' if hours > 1 else ''} {minutes} minutes"

    def _on_continue(self):
        """User chose to continue with full video."""
        self.trim_video = False
        self.accept()

    def _on_trim(self):
        """User chose to trim video."""
        self.trim_video = True
        self.accept()

    def should_trim(self) -> bool:
        """Returns True if user chose to trim the video."""
        return self.trim_video
