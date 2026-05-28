"""
Simple dialog for viewing an annotated frame at a larger size.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QResizeEvent
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout


class ImagePreviewDialog(QDialog):
    """Lightweight image viewer for enlarged frame previews."""

    def __init__(self, pixmap: QPixmap, title: str = "Frame Preview", parent=None):
        super().__init__(parent)
        self._pixmap = pixmap
        self.setWindowTitle(title)
        self.setMinimumSize(1000, 700)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("""
            QLabel {
                background: #111;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self._image_label, stretch=1)
        self._update_pixmap()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap.isNull():
            self._image_label.clear()
            return

        scaled = self._pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
