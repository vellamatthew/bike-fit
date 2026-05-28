"""Welcome dialog shown on application startup with video recording instructions."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser, QPushButton
)


class WelcomeDialog(QDialog):
    """
    Welcome screen shown on startup with app overview and usage instructions.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Bike Fit Analyser")
        self.setMinimumSize(650, 460)
        self.setModal(True)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Content area with help text
        content_browser = QTextBrowser()
        content_browser.setStyleSheet("""
            QTextBrowser {
                background: transparent;
                border: none;
                color: #ddd;
                font-size: 13px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #444;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #555;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        content_browser.setOpenExternalLinks(False)
        content_browser.setHtml(self._get_help_content())
        layout.addWidget(content_browser)

        # Next button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        next_btn = QPushButton("Get Started")
        next_btn.setMinimumHeight(36)
        next_btn.setMinimumWidth(120)
        next_btn.clicked.connect(self.accept)
        next_btn.setDefault(True)
        next_btn.setStyleSheet("""
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
        button_layout.addWidget(next_btn)

        layout.addLayout(button_layout)

    def _get_help_content(self) -> str:
        """Generate HTML content for the welcome dialog."""
        return """
        <style>
            body {
                margin: 0;
                padding: 0;
                line-height: 1.6;
            }
            h2 {
                color: #fff;
                font-size: 18px;
                font-weight: 600;
                margin-top: 0px;
                margin-bottom: 10px;
            }
            h3 {
                color: #aaa;
                font-size: 13px;
                font-weight: 600;
                margin-top: 18px;
                margin-bottom: 8px;
            }
            p {
                color: #aaa;
                margin-top: 0;
                margin-bottom: 14px;
            }
            ul {
                margin-top: 6px;
                margin-bottom: 12px;
                padding-left: 20px;
            }
            li {
                color: #bbb;
                margin-bottom: 5px;
            }
        </style>

        <h2>Getting Started</h2>

        <p>This app analyzes your cycling position from video and provides bike fit recommendations. Using pose detection, it measures your joint angles throughout the pedal stroke and compares them to targets for your cycling discipline.</p>

        <h3>Using the App</h3>
        <p>
            The interface flows from left to right. Start by dragging your video into the drop zone on the left panel, then click "Process Video". Results will appear in the center and right panels. The analysis should take about 30 seconds depending on video length.
        </p>

        <h3>Recording Your Video</h3>
        <p>
            Record 10-15 seconds of yourself cycling from the side. Position your camera 2-3 meters away at roughly saddle height, making sure both wheels are fully visible in the frame. The camera must be stationary. Good lighting and a clean background will improve detection accuracy.
        </p>

        <p style="margin-top: 24px; color: #666; font-size: 12px;">
            Click "Get Started" to begin.
        </p>
        """
