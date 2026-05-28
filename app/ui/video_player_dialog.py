"""
Simple dialog for viewing an exported annotated video.
"""
from pathlib import Path
import shutil

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QFileDialog
)


class ClickableVideoWidgetMixin:
    """Mixin that toggles playback when the video surface is clicked."""

    def set_click_handler(self, fn):
        self._on_click = fn

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and getattr(self, "_on_click", None):
            self._on_click()
            event.accept()
            return
        super().mousePressEvent(event)


class VideoPlayerDialog(QDialog):
    """Lightweight video player dialog backed by Qt Multimedia."""

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self._video_path = Path(video_path)
        if not self._video_path.exists():
            raise FileNotFoundError(f"Video not found: {self._video_path}")

        try:
            from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
            from PyQt6.QtMultimediaWidgets import QVideoWidget
        except ImportError as exc:
            raise RuntimeError("Qt Multimedia playback is unavailable") from exc

        self._QMediaPlayer = QMediaPlayer
        self._QAudioOutput = QAudioOutput
        self._QVideoWidget = QVideoWidget

        self.setWindowTitle("Annotated Video")
        self.setMinimumSize(900, 620)

        self._build_ui()
        self._load_video()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        self._title_label = QLabel(self._video_path.name)
        self._title_label.setStyleSheet("color: #fff; font-size: 13px; font-weight: bold;")
        self._title_label.setWordWrap(True)
        layout.addWidget(self._title_label)

        clickable_video_widget_cls = type(
            "ClickableVideoWidget",
            (ClickableVideoWidgetMixin, self._QVideoWidget),
            {}
        )
        self._video_widget = clickable_video_widget_cls()
        self._video_widget.set_click_handler(self._toggle_playback)
        self._video_widget.setStyleSheet("background: #000; border-radius: 8px;")
        self._video_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        self._video_widget.setToolTip("Click to play or pause")
        layout.addWidget(self._video_widget, stretch=1)

        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        self._position_slider.setRange(0, 0)
        self._position_slider.sliderMoved.connect(self._set_position)
        self._position_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #1a1a1a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0066cc;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        layout.addWidget(self._position_slider)

        controls = QHBoxLayout()
        controls.setSpacing(10)

        self._play_btn = QPushButton("Pause")
        self._play_btn.clicked.connect(self._toggle_playback)
        self._play_btn.setStyleSheet("""
            QPushButton {
                background: #0066cc;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 14px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #0052a3;
            }
        """)
        controls.addWidget(self._play_btn)

        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setStyleSheet("color: #aaa; font-size: 11px;")
        controls.addWidget(self._time_label)

        controls.addStretch()

        save_btn = QPushButton("Save Copy As...")
        save_btn.clicked.connect(self._save_copy_as)
        save_btn.setStyleSheet("""
            QPushButton {
                background: #3b3b3b;
                color: white;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 8px 14px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #474747;
                border-color: #666;
            }
        """)
        controls.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background: #3b3b3b;
                color: white;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 8px 14px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #474747;
                border-color: #666;
            }
        """)
        controls.addWidget(close_btn)

        layout.addLayout(controls)

        self._audio_output = self._QAudioOutput(self)
        self._audio_output.setVolume(0.0)

        self._player = self._QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)
        self._player.setVideoOutput(self._video_widget)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

    def _load_video(self):
        self._player.setSource(QUrl.fromLocalFile(str(self._video_path.resolve())))
        self._player.play()

    def _toggle_playback(self):
        if self._player.playbackState() == self._QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _set_position(self, position: int):
        self._player.setPosition(position)

    def _on_position_changed(self, position: int):
        self._position_slider.blockSignals(True)
        self._position_slider.setValue(position)
        self._position_slider.blockSignals(False)
        self._time_label.setText(
            f"{self._format_ms(position)} / {self._format_ms(self._player.duration())}"
        )

    def _on_duration_changed(self, duration: int):
        self._position_slider.setRange(0, duration)
        self._time_label.setText(f"0:00 / {self._format_ms(duration)}")

    def _on_playback_state_changed(self, state):
        if state == self._QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("Pause")
        else:
            self._play_btn.setText("Play")

    def _save_copy_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotated Video",
            self._video_path.name,
            "Videos (*.mp4)"
        )
        if path:
            shutil.copy2(self._video_path, path)

    def closeEvent(self, event):
        """Ensure media player releases the video file before closing."""
        self._player.stop()
        self._player.setSource(QUrl())
        event.accept()

    @staticmethod
    def _format_ms(ms: int) -> str:
        total_seconds = max(0, int(ms / 1000))
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes}:{seconds:02d}"
