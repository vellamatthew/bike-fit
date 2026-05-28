import cv2
import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QFrame, QSizePolicy, QComboBox,
    QCheckBox, QSlider, QScrollArea, QGridLayout, QToolTip
)
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap, QCursor

from workers.inference_worker import VideoWorker
from processing.critical_positions import extract_angles_at_positions, select_representative_frames
from processing.fit_assessment import (
    assess_fit,
    generate_recommendations,
    Discipline,
    DeviationSeverity,
)
from processing.angle_overlay import annotate_frame_with_angles
from ui.frame_display import FrameDisplayWidget
from ui.fit_summary import FitSummaryWidget


class DropZone(QLabel):
    """Clickable drag-and-drop zone for video files."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Drop video here\nor click to browse")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(200)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #888;
                border-radius: 8px;
                color: #888;
                font-size: 14px;
            }
            QLabel:hover {
                border-color: #aaa;
                color: #aaa;
            }
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._on_file = None

    def set_callback(self, fn):
        self._on_file = fn

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm)"
        )
        if path and self._on_file:
            self._on_file(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls and self._on_file:
            self._on_file(urls[0].toLocalFile())


class HelpIconLabel(QLabel):
    """Small hoverable help icon that shows its tooltip immediately."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._instant_tooltip = ""

    def set_instant_tooltip(self, text: str):
        self._instant_tooltip = text or ""
        super().setToolTip("")

    def enterEvent(self, event):
        if self._instant_tooltip:
            QToolTip.showText(QCursor.pos(), self._instant_tooltip, self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)

    def event(self, event):
        if event.type() == QEvent.Type.ToolTip:
            return True
        return super().event(event)


class VideoPanel(QWidget):
    """Simplified video-only panel for bike fit analysis."""

    def __init__(self, parent=None, on_bike_setup=None):
        super().__init__(parent)
        self._on_bike_setup = on_bike_setup
        self._video_path: str | None = None
        self._worker: VideoWorker | None = None
        self._angle_data: list[dict] = []
        self._start_time: float = 0.0
        self._total_frames_count: int = 0
        self._current_frame_index: int = 0
        self._tdc_frames: list[int] = []  # Top Dead Centre (max flexion)
        self._bdc_frames: list[int] = []  # Bottom Dead Centre (max extension)
        self._active_side: str = "auto"  # "auto", "left", or "right"
        self._smoothing_enabled: bool = True
        self._smoothing_window: int = 9
        self._perspective_correction_enabled: bool = False
        self._per_frame_perspective_enabled: bool = False
        self._fixed_perspective_warp_enabled: bool = False
        self._normalize_wheelbase_view_enabled: bool = False
        self._export_video_enabled: bool = True
        self._exported_video_path: str | None = None
        self._annotated_video_signature: tuple | None = None
        self._current_discipline: Discipline = Discipline.ROAD
        self._fit_assessment_result: dict | None = None
        self._representative_frames: dict | None = None
        self._homography_matrix: np.ndarray | None = None  # Stores perspective correction matrix
        self._per_frame_homographies: dict[int, np.ndarray] = {}
        self._per_frame_wheel_ellipses: dict[int, list] = {}
        self._scale_factor: float | None = None  # Physical measurement calibration (mm/px)
        self._body_measurements: dict | None = None
        self._requires_reprocess_notice: bool = False
        self._long_video_warning_shown: bool = False  # Track if warning was shown
        self._trimmed_total_frames: int | None = None  # Adjusted total when trimming
        self._export_worker: 'ExportWorker' | None = None  # Background video export worker
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ---- Left Panel: Controls ----
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)

        # Drop zone
        self._drop_zone = DropZone()
        self._drop_zone.set_callback(self._load_video)
        self._drop_zone.setMinimumHeight(150)
        left_panel.addWidget(self._drop_zone)

        # Process button
        action_stack = QFrame()
        action_stack.setStyleSheet("QFrame { background: transparent; border: none; }")
        action_stack_layout = QVBoxLayout(action_stack)
        action_stack_layout.setContentsMargins(0, 0, 0, 0)
        action_stack_layout.setSpacing(0)

        self._process_btn = QPushButton("Process Video")
        self._process_btn.setEnabled(False)
        self._process_btn.setMinimumHeight(36)
        self._process_btn.clicked.connect(self._run_inference)
        self._process_btn.setStyleSheet("""
            QPushButton {
                background: #0066cc;
                color: white;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #0052a3;
            }
            QPushButton:disabled {
                background: #444;
                color: #888;
            }
        """)
        action_stack_layout.addWidget(self._process_btn)

        # Cancel button (shown during processing)
        self._cancel_btn = QPushButton("Cancel Processing")
        self._cancel_btn.setMinimumHeight(36)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_processing)
        self._cancel_btn.setStyleSheet("""
            QPushButton {
                background: #d44;
                color: white;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #b22;
            }
        """)
        action_stack_layout.addWidget(self._cancel_btn)
        left_panel.addWidget(action_stack)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setMinimumHeight(6)
        self._progress.setMaximumHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background: #2a2a2a;
            }
            QProgressBar::chunk {
                background: #0066cc;
                border-radius: 3px;
            }
        """)
        left_panel.addWidget(self._progress)

        # Status label
        self._status = QLabel("")
        self._status.setStyleSheet("color: #aaa; font-size: 12px;")
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(20)
        left_panel.addWidget(self._status)

        # ETA label
        self._eta_label = QLabel("")
        self._eta_label.setStyleSheet("color: #888; font-size: 11px;")
        self._eta_label.setVisible(False)
        self._eta_label.setMinimumHeight(16)
        left_panel.addWidget(self._eta_label)

        # Info panel (shows results summary)
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background: #2a2a2a;
                border-radius: 8px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(12, 12, 12, 12)

        self._info_label = QLabel("Upload a video to begin")
        self._info_label.setStyleSheet("color: #888; font-size: 12px;")
        self._info_label.setWordWrap(True)
        info_layout.addWidget(self._info_label)

        left_panel.addWidget(info_frame)

        # Bike setup button
        bike_setup_row = QHBoxLayout()
        bike_setup_row.setContentsMargins(0, 0, 0, 0)
        bike_setup_row.setSpacing(0)

        self._bike_setup_btn = QPushButton("Configure Bike")
        self._bike_setup_btn.setMinimumHeight(32)
        self._bike_setup_btn.setMaximumHeight(32)
        self._bike_setup_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._bike_setup_btn.clicked.connect(self._open_bike_setup)
        self._bike_setup_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                color: #dddddd;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 400;
                text-align: left;
                padding: 0 14px;
            }
            QPushButton:hover {
                background: #333333;
                color: #ffffff;
            }
            QPushButton:pressed {
                background: #383838;
            }
        """)
        bike_setup_row.addWidget(self._bike_setup_btn)
        bike_setup_row.addSpacing(6)
        bike_setup_row.addWidget(self._create_help_icon(
            "Search for your bike and calibrate wheel scale to estimate body measurements in millimeters from the video. This works best when perspective correction is enabled."
        ))
        bike_setup_row.addSpacing(8)

        self._preview_btn = QPushButton("View Annotated Video")
        self._preview_btn.setMinimumHeight(32)
        self._preview_btn.setMaximumHeight(32)
        self._preview_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._preview_btn.clicked.connect(self._view_annotated_video)
        self._preview_btn.setEnabled(False)
        self._preview_btn.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                color: #dddddd;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 400;
                text-align: left;
                padding: 0 14px;
            }
            QPushButton:hover {
                background: #333333;
                color: #ffffff;
            }
            QPushButton:pressed {
                background: #383838;
            }
            QPushButton:disabled {
                background: #2a2a2a;
                color: #666666;
            }
        """)
        bike_setup_row.addWidget(self._preview_btn)
        bike_setup_row.addStretch()
        left_panel.addLayout(bike_setup_row)

        # Calibration status label (shown after calibration)
        self._calibration_status = QLabel("")
        self._calibration_status.setStyleSheet("color: #aaa; font-size: 11px;")
        self._calibration_status.setWordWrap(True)
        self._calibration_status.setVisible(False)
        left_panel.addWidget(self._calibration_status)

        # Settings panel
        settings_frame = QFrame()
        settings_frame.setStyleSheet("""
            QFrame {
                background: #2a2a2a;
                border-radius: 8px;
            }
        """)
        settings_frame.setMinimumHeight(225)
        settings_layout = QVBoxLayout(settings_frame)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(10)

        # Settings title
        settings_title = QLabel("Analysis Settings")
        settings_title.setStyleSheet("color: #fff; font-size: 13px; font-weight: bold;")
        settings_layout.addWidget(settings_title)

        # Discipline selection
        discipline_layout = QHBoxLayout()
        discipline_layout.setSpacing(4)
        discipline_label = QLabel("Discipline:")
        discipline_label.setStyleSheet("color: #aaa; font-size: 11px;")
        discipline_layout.addWidget(discipline_label)
        discipline_layout.addWidget(self._create_help_icon(
            "Adjusts the target fit ranges for the riding position you want to evaluate."
        ))

        self._discipline_combo = QComboBox()
        self._discipline_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._discipline_combo.addItems(["Road", "Mountain", "Time Trial", "Triathlon"])
        self._discipline_combo.setStyleSheet("""
            QComboBox {
                background: #1a1a1a;
                color: #fff;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #888;
                margin-right: 6px;
            }
        """)
        self._discipline_combo.currentTextChanged.connect(self._on_discipline_changed)
        discipline_layout.addWidget(self._discipline_combo)
        settings_layout.addLayout(discipline_layout)

        # Side selection
        side_layout = QHBoxLayout()
        side_layout.setSpacing(4)
        side_label = QLabel("Detected Side:")
        side_label.setStyleSheet("color: #aaa; font-size: 11px;")
        side_layout.addWidget(side_label)
        side_layout.addWidget(self._create_help_icon(
            "Auto picks one side for the whole clip. Choose Left or Right to force a specific side."
        ))

        self._side_combo = QComboBox()
        self._side_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._side_combo.addItems(["Auto", "Left", "Right"])
        self._side_combo.setStyleSheet("""
            QComboBox {
                background: #1a1a1a;
                color: #fff;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #888;
                margin-right: 6px;
            }
        """)
        self._side_combo.currentTextChanged.connect(self._on_side_changed)
        side_layout.addWidget(self._side_combo)
        settings_layout.addLayout(side_layout)

        # Perspective correction toggle
        perspective_row = QHBoxLayout()
        perspective_row.setSpacing(4)
        self._perspective_check = QCheckBox("Apply Perspective Correction (Experimental)")
        self._perspective_check.setChecked(False)
        self._perspective_check.setStyleSheet("""
            QCheckBox {
                color: #aaa;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #444;
                border-radius: 3px;
                background: #1a1a1a;
            }
            QCheckBox::indicator:checked {
                background: #0066cc;
                border-color: #0066cc;
            }
        """)
        self._perspective_check.stateChanged.connect(self._on_perspective_correction_changed)
        perspective_row.addWidget(self._perspective_check)
        perspective_row.addWidget(self._create_help_icon(
            "Corrects side-view camera angle before analysis. Use when the bike is filmed from an angle rather than perfectly side-on."
        ))
        perspective_row.addStretch()
        settings_layout.addLayout(perspective_row)

        per_frame_row = QHBoxLayout()
        per_frame_row.setSpacing(4)
        per_frame_row.setContentsMargins(18, 0, 0, 0)
        self._per_frame_perspective_check = QCheckBox("Update Correction Per Frame")
        self._per_frame_perspective_check.setChecked(False)
        self._per_frame_perspective_check.setEnabled(False)
        self._per_frame_perspective_check.setStyleSheet(self._perspective_check.styleSheet())
        self._per_frame_perspective_check.stateChanged.connect(self._on_per_frame_perspective_changed)
        per_frame_row.addWidget(self._per_frame_perspective_check)
        per_frame_row.addWidget(self._create_help_icon(
            "Recalculates correction throughout the video. Use if the camera moves, the bike shifts, or a single correction does not stay aligned."
        ))
        per_frame_row.addStretch()
        settings_layout.addLayout(per_frame_row)

        fixed_warp_row = QHBoxLayout()
        fixed_warp_row.setSpacing(4)
        fixed_warp_row.setContentsMargins(18, 0, 0, 0)
        self._fixed_perspective_warp_check = QCheckBox("Keep Original Frame Size")
        self._fixed_perspective_warp_check.setChecked(False)
        self._fixed_perspective_warp_check.setEnabled(False)
        self._fixed_perspective_warp_check.setVisible(False)
        self._fixed_perspective_warp_check.setStyleSheet(self._perspective_check.styleSheet())
        self._fixed_perspective_warp_check.stateChanged.connect(self._on_fixed_perspective_warp_changed)
        fixed_warp_row.addWidget(self._fixed_perspective_warp_check)
        fixed_warp_help = self._create_help_icon(
            "Keeps corrected video at the original size by cropping or zooming the warped view. Use when normal correction creates huge borders or is rejected as too large."
        )
        fixed_warp_help.setVisible(False)
        fixed_warp_row.addWidget(fixed_warp_help)
        fixed_warp_row.addStretch()
        settings_layout.addLayout(fixed_warp_row)

        demo_view_row = QHBoxLayout()
        demo_view_row.setSpacing(4)
        demo_view_row.setContentsMargins(18, 0, 0, 0)
        self._normalize_wheelbase_view_check = QCheckBox("Standardize Bike View")
        self._normalize_wheelbase_view_check.setChecked(False)
        self._normalize_wheelbase_view_check.setEnabled(False)
        self._normalize_wheelbase_view_check.setStyleSheet(self._perspective_check.styleSheet())
        self._normalize_wheelbase_view_check.stateChanged.connect(self._on_normalize_wheelbase_view_changed)
        demo_view_row.addWidget(self._normalize_wheelbase_view_check)
        demo_view_row.addWidget(self._create_help_icon(
            "Keeps the corrected video at the original frame size, scales the bike to a consistent wheelbase size, and hides pose overlays. Use when normal correction creates large borders or inconsistent bike sizing."
        ))
        demo_view_row.addStretch()
        settings_layout.addLayout(demo_view_row)

        # Smoothing toggle
        smoothing_row = QHBoxLayout()
        smoothing_row.setSpacing(4)
        self._smoothing_check = QCheckBox("Apply Smoothing")
        self._smoothing_check.setChecked(True)
        self._smoothing_check.setStyleSheet("""
            QCheckBox {
                color: #aaa;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #444;
                border-radius: 3px;
                background: #1a1a1a;
            }
            QCheckBox::indicator:checked {
                background: #0066cc;
                border-color: #0066cc;
            }
        """)
        self._smoothing_check.stateChanged.connect(self._on_smoothing_changed)
        smoothing_row.addWidget(self._smoothing_check)
        smoothing_row.addWidget(self._create_help_icon(
            "Reduces frame-to-frame jitter in the detected angles before the app finds pedal positions."
        ))
        smoothing_row.addStretch()
        settings_layout.addLayout(smoothing_row)

        # Window size slider
        window_layout = QVBoxLayout()
        window_header = QHBoxLayout()
        window_header.setSpacing(4)
        window_label = QLabel(f"Smoothing Window: {self._smoothing_window}")
        window_label.setStyleSheet("color: #aaa; font-size: 11px;")
        window_header.addWidget(window_label)
        window_header.addWidget(self._create_help_icon(
            "Higher values smooth more aggressively. Lower values stay more responsive to quick angle changes."
        ))
        window_header.addStretch()
        window_layout.addLayout(window_header)
        self._window_label = window_label

        self._window_slider = QSlider(Qt.Orientation.Horizontal)
        self._window_slider.setMinimum(5)
        self._window_slider.setMaximum(15)
        self._window_slider.setValue(9)
        self._window_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._window_slider.setTickInterval(2)
        self._window_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #1a1a1a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0066cc;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
        """)
        self._window_slider.valueChanged.connect(self._on_window_changed)
        window_layout.addWidget(self._window_slider)
        settings_layout.addLayout(window_layout)

        left_panel.addWidget(settings_frame)

        left_panel.addStretch()

        # Create left panel widget
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(300)
        root.addWidget(left_widget)

        # ---- Right Panel: Fit Summary + BDC Frame ----
        right_panel = QHBoxLayout()
        right_panel.setSpacing(12)

        # BDC frame display
        self._bdc_display = FrameDisplayWidget("Visual Evidence: BDC (Bottom of Pedal Stroke)")
        self._bdc_display.setMinimumWidth(440)
        self._bdc_display.set_frame_click_handler(self._view_frame_preview)
        right_panel.addWidget(self._bdc_display, stretch=3)

        # Fit summary widget (verdict + recommendations)
        self._fit_summary = FitSummaryWidget()
        self._fit_summary.setMinimumWidth(200)
        right_panel.addWidget(self._fit_summary, stretch=2)

        root.addLayout(right_panel, stretch=1)

    def _create_help_icon(self, tooltip: str) -> QLabel:
        """Create a small help chip that shows a tooltip on hover."""
        label = HelpIconLabel("?")
        label.setFixedSize(12, 12)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.set_instant_tooltip(f'<div style="max-width: 220px; white-space: normal;">{tooltip}</div>')
        label.setCursor(Qt.CursorShape.WhatsThisCursor)
        label.setStyleSheet("""
            QLabel {
                color: #cfcfcf;
                background: #343434;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                font-size: 9px;
                font-weight: 600;
            }
            QLabel:hover {
                background: #3d3d3d;
                border-color: #5b5b5b;
                color: #ffffff;
            }
            QToolTip {
                background: rgba(56, 56, 56, 242);
                color: #f0f0f0;
                border: 1px solid rgba(255, 255, 255, 18);
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 11px;
            }
        """)
        return label

    def _open_bike_setup(self):
        """Open bike setup flow when the callback is available."""
        if self._on_bike_setup:
            scale_factor = self._on_bike_setup(
                self._video_path,
                self._get_calibration_homography() if self._perspective_correction_enabled else None
            )
            if scale_factor is not None:
                self._scale_factor = scale_factor
                self._calibration_status.setVisible(True)
                self._calibration_status.setText(f"Calibrated: {scale_factor:.4f} mm/px")

                if self._angle_data and self._tdc_frames and self._bdc_frames:
                    self._status.setText("Calibration complete. Updating measurements...")
                    self._perform_fit_assessment()
                else:
                    self._status.setText(f"Calibration complete. Scale: {scale_factor:.4f} mm/px")

    def _load_video(self, path: str):
        """Load video and extract basic info."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._status.setText("Error: Could not open video file.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self._video_path = path
        self._requires_reprocess_notice = False
        self._process_btn.setText("Process Video")
        self._process_btn.setEnabled(True)

        # Reset display, homography, and calibration
        self._bdc_display.clear()
        self._fit_summary.clear()
        self._homography_matrix = None  # Clear any previous perspective correction
        self._per_frame_homographies.clear()
        self._per_frame_wheel_ellipses.clear()
        self._scale_factor = None  # Clear previous calibration
        self._body_measurements = None
        self._calibration_status.setVisible(False)  # Hide calibration status
        self._calibration_status.setText("")
        self._clear_measurements_summary()
        self._exported_video_path = None
        self._annotated_video_signature = None
        self._bdc_display.set_frame_click_enabled(False)
        self._preview_btn.setEnabled(False)

        import os
        filename = os.path.basename(path)
        self._status.setText(f"Loaded: {filename}")
        self._info_label.setText(
            f"Video: {filename}\n"
            f"Frames: {total_frames} | FPS: {fps:.1f} | Resolution: {width}x{height}"
        )

    def _run_inference(self):
        """Start video processing worker."""
        if self._video_path is None:
            return

        status_text = "Reprocessing video with updated settings..." if self._requires_reprocess_notice else "Processing video frames..."
        self._requires_reprocess_notice = False
        self._process_btn.setText("Process Video")
        self._process_btn.setEnabled(False)
        self._process_btn.setVisible(False)
        self._cancel_btn.setVisible(True)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText(status_text)
        self._eta_label.setVisible(True)
        self._eta_label.setText("Calculating ETA...")
        self._angle_data.clear()
        self._exported_video_path = None
        self._annotated_video_signature = None
        self._bdc_display.set_frame_click_enabled(False)
        self._preview_btn.setEnabled(False)
        self._start_time = time.time()
        self._current_frame_index = 0
        self._long_video_warning_shown = False  # Reset for new processing session
        self._trimmed_total_frames = None  # Reset trim state
        self._homography_matrix = None
        self._per_frame_homographies.clear()
        self._per_frame_wheel_ellipses.clear()

        # Create and start worker
        perspective_mode = "per_frame" if self._per_frame_perspective_enabled else "single"
        self._worker = VideoWorker(
            self._video_path,
            self._perspective_correction_enabled,
            self._export_video_enabled,
            perspective_mode,
            self._fixed_perspective_warp_enabled,
            self._normalize_wheelbase_view_enabled
        )
        self._worker.total_frames.connect(self._on_total_frames)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.angle_plot_ready.connect(self._on_angle_plot_ready)
        self._worker.perspective_status.connect(self._on_perspective_status)
        self._worker.wheel_confirmation_request.connect(self._on_wheel_confirmation_request)
        self._worker.homography_computed.connect(self._on_homography_computed)
        self._worker.video_exported.connect(self._on_video_exported)
        self._worker.start()

    def _show_long_video_warning(self, eta_seconds: float):
        """Show warning dialog when video processing will take > 2 minutes."""
        from ui.long_video_warning_dialog import LongVideoWarningDialog

        self._long_video_warning_shown = True  # Mark as shown

        dialog = LongVideoWarningDialog(eta_seconds, self._total_frames_count, self)
        result = dialog.exec()

        if result == dialog.DialogCode.Accepted and dialog.should_trim():
            # User wants to trim - tell worker to stop after first 10 seconds
            print("[LongVideo] User requested video trimming - processing first 10 seconds only")

            # Calculate frame range for first 10 seconds
            if self._worker:
                # Get FPS from worker's video capture
                import cv2
                cap = cv2.VideoCapture(self._video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

                frames_for_10s = int(fps * 10)
                end_frame = min(self._total_frames_count, frames_for_10s)

                print(f"[LongVideo] Total frames: {self._total_frames_count}, FPS: {fps:.1f}")
                print(f"[LongVideo] Will process frames 0 to {end_frame} ({end_frame} frames)")

                # Tell worker to limit frame processing
                self._worker.set_frame_limit(0, end_frame)

                # Update trimmed total for UI display
                self._trimmed_total_frames = end_frame

                self._status.setText(f"Processing first 10 seconds (0-{end_frame} frames)...")
        elif result == dialog.DialogCode.Accepted:
            # User chose to continue anyway
            print(f"[LongVideo] User chose to continue with full video (ETA: {eta_seconds:.0f}s)")
        else:
            # Dialog was closed/rejected - continue processing
            print("[LongVideo] Warning dialog closed - continuing processing")

    def _cancel_processing(self):
        """Cancel video processing."""
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._status.setText("Cancelling processing...")
            self._cancel_btn.setEnabled(False)

    def _on_total_frames(self, total: int):
        """Handle total frames signal."""
        self._total_frames_count = total
        self._status.setText(f"Processing {total} frames...")

    def _on_progress(self, percent: int):
        """Update progress bar and ETA."""
        self._progress.setValue(percent)

        # Calculate ETA (use trimmed count if video was trimmed)
        total_for_calc = self._trimmed_total_frames if self._trimmed_total_frames else self._total_frames_count

        if percent > 0 and total_for_calc > 0:
            elapsed = time.time() - self._start_time
            frames_processed = (percent / 100) * total_for_calc

            if frames_processed > 0:
                frames_per_second = frames_processed / elapsed
                frames_remaining = total_for_calc - frames_processed
                eta_seconds = frames_remaining / frames_per_second

                # Check for long video after ETA stabilizes
                # Wait for: sufficient frames (>30) AND elapsed time (>3s) AND long ETA (>2min)
                if (not self._long_video_warning_shown and
                    frames_processed > 30 and
                    elapsed > 3.0 and
                    eta_seconds > 120):  # 2 minutes
                    self._show_long_video_warning(eta_seconds)

                # Format ETA
                if eta_seconds < 60:
                    eta_str = f"{int(eta_seconds)}s"
                elif eta_seconds < 3600:
                    minutes = int(eta_seconds / 60)
                    seconds = int(eta_seconds % 60)
                    eta_str = f"{minutes}m {seconds}s"
                else:
                    hours = int(eta_seconds / 3600)
                    minutes = int((eta_seconds % 3600) / 60)
                    eta_str = f"{hours}h {minutes}m"

                self._eta_label.setText(f"ETA: {eta_str} ({frames_per_second:.1f} fps)")

    def _on_frame_ready(self, annotated: np.ndarray, angles: dict):
        """Update status during processing."""
        self._current_frame_index += 1
        # Update status with current frame info (use trimmed total if available)
        total_to_display = self._trimmed_total_frames if self._trimmed_total_frames else self._total_frames_count
        self._status.setText(f"Processing frame {self._current_frame_index} of {total_to_display}...")

    def _on_finished(self, angle_records: list[dict]):
        """Handle completion of video processing."""
        print(f"[Processing] Completed with {len(angle_records)} angle records")
        self._angle_data = angle_records
        self._per_frame_homographies = {
            int(rec["frame"]): np.array(rec["homography"], dtype=np.float64)
            for rec in angle_records
            if rec.get("homography") is not None
        }
        self._per_frame_wheel_ellipses = {
            int(rec["frame"]): rec["wheel_ellipses"]
            for rec in angle_records
            if rec.get("wheel_ellipses") is not None
        }
        if self._per_frame_homographies and self._per_frame_perspective_enabled:
            print(f"[Perspective] Stored {len(self._per_frame_homographies)} per-frame homographies")
        self._progress.setVisible(False)
        self._eta_label.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setEnabled(True)
        self._process_btn.setVisible(True)
        self._process_btn.setEnabled(True)

        # Calculate statistics
        frames_processed = len(angle_records)
        frames_with_data = sum(1 for rec in angle_records if rec.get("knee_flexion") is not None)
        total_time = time.time() - self._start_time
        detection_rate = (100 * frames_with_data / frames_processed) if frames_processed > 0 else 0.0
        average_fps = (frames_processed / total_time) if total_time > 0 else 0.0
        completion_status = f"Processing complete! ({total_time:.1f}s)"

        self._info_label.setText(
            f"Processed {frames_processed} frames\n"
            f"Detected person in {frames_with_data} frames ({detection_rate:.1f}%)"
        )

        print(f"\n=== Processing Complete ===")
        print(f"Total frames: {frames_processed}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average FPS: {average_fps:.1f}")
        print(f"Frames with angles: {frames_with_data}")

        # Complete the angle-based analysis if pedal positions were found.
        if self._tdc_frames and self._bdc_frames:
            print("[Analysis] Running fit assessment")
            self._status.setText("Finalizing bike fit analysis...")

            # Perform fit assessment (with or without calibration)
            self._perform_fit_assessment()

            if self._status.text() in ("Finalizing bike fit analysis...", "Rendering annotated video..."):
                self._status.setText(completion_status)
        else:
            self._status.setText(completion_status)

    def _on_error(self, msg: str):
        """Handle worker error."""
        self._progress.setVisible(False)
        self._eta_label.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setEnabled(True)
        self._process_btn.setVisible(True)
        self._process_btn.setEnabled(True)
        self._status.setText(f"Error: {msg}")

    def _on_video_exported(self, path: str):
        """Store the exported video path and expose viewer actions."""
        self._exported_video_path = path
        self._annotated_video_signature = None
        self._preview_btn.setEnabled(True)
        self._status.setText("Finalizing bike fit analysis...")

    def _on_perspective_status(self, msg: str):
        """Handle perspective correction status updates."""
        print(f"[Perspective] {msg}")
        if msg.startswith("Video exported to:"):
            return
        self._status.setText(msg)

    def _on_wheel_confirmation_request(self, frame: np.ndarray, ellipses: list, img_shape: tuple):
        """Show wheel confirmation dialog to user."""
        from ui.wheel_confirmation_dialog import WheelConfirmationDialog

        print(f"[Perspective] Showing wheel confirmation dialog...")

        # Show dialog
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
            cancel_label="Cancel",
            confirm_label="Looks Good",
            show_measurement_overlay=False,
        )
        result = dialog.exec()

        # Send response back to worker
        if result == dialog.DialogCode.Accepted:
            print(f"[Perspective] User confirmed wheels")
            self._worker.confirm_wheels(True, ellipses, img_shape)
        else:
            print(f"[Perspective] User rejected wheels")
            self._worker.confirm_wheels(False)

    def _on_homography_computed(self, homography: np.ndarray | None):
        """Store the computed homography matrix for later use when extracting frames."""
        self._homography_matrix = homography
        if not self._per_frame_perspective_enabled:
            self._per_frame_homographies.clear()
            self._per_frame_wheel_ellipses.clear()
        if homography is not None:
            print(f"[Perspective] Homography matrix received and stored for frame extraction")
        else:
            print(f"[Perspective] No homography matrix (correction disabled or failed)")

    def _get_calibration_homography(self) -> np.ndarray | None:
        """Return a representative homography for calibration dialogs."""
        if self._per_frame_perspective_enabled and self._per_frame_homographies:
            first_frame = min(self._per_frame_homographies)
            print(f"[Calibration] Using per-frame homography from frame {first_frame}")
            return self._per_frame_homographies[first_frame]
        return self._homography_matrix

    def get_angle_data(self) -> list[dict]:
        """Return collected angle data for external use."""
        return self._angle_data

    def get_tdc_frames(self) -> list[int]:
        """Return frame indices where pedal is at Top Dead Centre (max knee flexion)."""
        return self._tdc_frames

    def get_bdc_frames(self) -> list[int]:
        """Return frame indices where pedal is at Bottom Dead Centre (max knee extension)."""
        return self._bdc_frames

    def _on_angle_plot_ready(self, raw_angles: list, smoothed_angles: list):
        """Update pedal-stroke analysis from smoothed angle data."""
        self._plot_angles(raw_angles, smoothed_angles, self._active_side)

    def _on_discipline_changed(self, text: str):
        """Handle discipline selection change."""
        discipline_map = {
            "Road": Discipline.ROAD,
            "Mountain": Discipline.MOUNTAIN,
            "Time Trial": Discipline.TIME_TRIAL,
            "Triathlon": Discipline.TRIATHLON
        }
        self._current_discipline = discipline_map.get(text, Discipline.ROAD)
        print(f"[Settings] Discipline changed to: {self._current_discipline.value}")
        if self._angle_data and self._tdc_frames and self._bdc_frames:
            self._update_analysis()

    def _on_side_changed(self, text: str):
        """Handle side selection change."""
        self._active_side = text.lower()
        print(f"[Settings] Side changed to: {self._active_side}")
        if self._angle_data:
            self._update_analysis()

    def _on_smoothing_changed(self, state: int):
        """Handle smoothing toggle change."""
        self._smoothing_enabled = bool(state)
        print(f"[Settings] Smoothing: {'enabled' if self._smoothing_enabled else 'disabled'}")
        if self._angle_data:
            self._update_analysis()

    def _on_perspective_correction_changed(self, state: int):
        """Handle perspective correction toggle change."""
        self._perspective_correction_enabled = bool(state)
        self._per_frame_perspective_check.setEnabled(self._perspective_correction_enabled)
        self._fixed_perspective_warp_check.setEnabled(self._perspective_correction_enabled)
        self._normalize_wheelbase_view_check.setEnabled(self._perspective_correction_enabled)
        if not self._perspective_correction_enabled:
            self._per_frame_perspective_check.setChecked(False)
            self._fixed_perspective_warp_check.setChecked(False)
            self._normalize_wheelbase_view_check.setChecked(False)
            self._per_frame_perspective_enabled = False
            self._fixed_perspective_warp_enabled = False
            self._normalize_wheelbase_view_enabled = False
        print(f"[Settings] Perspective correction: {'enabled' if self._perspective_correction_enabled else 'disabled'}")

        # Perspective correction requires reprocessing the video at the frame level
        # Clear existing results and prompt user to reprocess if video is loaded
        if self._video_path:
            self._angle_data.clear()
            self._tdc_frames.clear()
            self._bdc_frames.clear()
            self._bdc_display.clear()
            self._fit_summary.clear()
            self._homography_matrix = None  # Clear stored homography matrix
            self._per_frame_homographies.clear()
            self._per_frame_wheel_ellipses.clear()
            self._scale_factor = None  # Clear calibration
            self._body_measurements = None
            self._requires_reprocess_notice = True
            self._calibration_status.setVisible(False)
            self._calibration_status.setText("")
            self._clear_measurements_summary()
            self._process_btn.setText("Reprocess Video")
            self._process_btn.setEnabled(True)

            if self._perspective_correction_enabled:
                self._status.setText("Results cleared. Click 'Reprocess Video' to apply perspective correction.")
            else:
                self._status.setText("Results cleared. Click 'Reprocess Video' to continue without perspective correction.")

    def _on_per_frame_perspective_changed(self, state: int):
        """Handle per-frame perspective correction toggle change."""
        self._per_frame_perspective_enabled = bool(state) and self._perspective_correction_enabled
        print(
            "[Settings] Per-frame perspective correction: "
            f"{'enabled' if self._per_frame_perspective_enabled else 'disabled'}"
        )

        if self._video_path:
            self._angle_data.clear()
            self._tdc_frames.clear()
            self._bdc_frames.clear()
            self._bdc_display.clear()
            self._fit_summary.clear()
            self._homography_matrix = None
            self._per_frame_homographies.clear()
            self._per_frame_wheel_ellipses.clear()
            self._scale_factor = None
            self._body_measurements = None
            self._requires_reprocess_notice = True
            self._calibration_status.setVisible(False)
            self._calibration_status.setText("")
            self._clear_measurements_summary()
            self._process_btn.setText("Reprocess Video")
            self._process_btn.setEnabled(True)
            if self._per_frame_perspective_enabled:
                self._status.setText("Results cleared. Click 'Reprocess Video' to estimate perspective per frame.")
            else:
                self._status.setText("Results cleared. Click 'Reprocess Video' to use single-frame perspective correction.")

    def _clear_results_for_perspective_option_change(self, status_text: str):
        """Clear processed data when a perspective sub-option changes."""
        if not self._video_path:
            return

        self._angle_data.clear()
        self._tdc_frames.clear()
        self._bdc_frames.clear()
        self._bdc_display.clear()
        self._fit_summary.clear()
        self._homography_matrix = None
        self._per_frame_homographies.clear()
        self._per_frame_wheel_ellipses.clear()
        self._scale_factor = None
        self._body_measurements = None
        self._requires_reprocess_notice = True
        self._calibration_status.setVisible(False)
        self._calibration_status.setText("")
        self._clear_measurements_summary()
        self._process_btn.setText("Reprocess Video")
        self._process_btn.setEnabled(True)
        self._status.setText(status_text)

    def _on_fixed_perspective_warp_changed(self, state: int):
        """Handle fixed perspective warp toggle change."""
        self._fixed_perspective_warp_enabled = bool(state) and self._perspective_correction_enabled
        print(
            "[Settings] Fixed perspective warp: "
            f"{'enabled' if self._fixed_perspective_warp_enabled else 'disabled'}"
        )
        self._clear_results_for_perspective_option_change(
            "Results cleared. Click 'Reprocess Video' to update fixed perspective warp."
        )

    def _on_normalize_wheelbase_view_changed(self, state: int):
        """Handle demo wheelbase-normalized view toggle change."""
        self._normalize_wheelbase_view_enabled = bool(state) and self._perspective_correction_enabled
        print(
            "[Settings] Normalize bike scale: "
            f"{'enabled' if self._normalize_wheelbase_view_enabled else 'disabled'}"
        )
        self._clear_results_for_perspective_option_change(
            "Results cleared. Click 'Reprocess Video' to update normalized bike scale view."
        )

    def _on_window_changed(self, value: int):
        """Handle smoothing window size change."""
        # Ensure odd number for Savitzky-Golay
        if value % 2 == 0:
            value += 1
        self._smoothing_window = value
        self._window_label.setText(f"Smoothing Window: {value}")
        print(f"[Settings] Smoothing window changed to: {value}")
        if self._angle_data:
            self._update_analysis()

    def _on_export_video_changed(self, state: int):
        """Handle export video toggle change."""
        self._export_video_enabled = True
        print("[Settings] Export video: always enabled")

    def _view_annotated_video(self):
        """Open the exported annotated video in-app when possible."""
        if not self._exported_video_path:
            return

        try:
            from ui.video_player_dialog import VideoPlayerDialog
            dialog = VideoPlayerDialog(self._exported_video_path, self)
            dialog.exec()
        except Exception as exc:
            # Fallback to the system-default player.
            try:
                os.startfile(os.path.abspath(self._exported_video_path))
            except Exception as open_exc:
                self._status.setText(f"Could not open exported video: {open_exc}")

    def _view_frame_preview(self):
        """Open the current annotated frame in a larger preview dialog."""
        pixmap = self._bdc_display.get_current_pixmap()
        if pixmap is None or pixmap.isNull():
            return

        try:
            from ui.image_preview_dialog import ImagePreviewDialog
            dialog = ImagePreviewDialog(pixmap, "Annotated Frame Preview", self)
            dialog.exec()
        except Exception as exc:
            self._status.setText(f"Could not open frame preview: {exc}")

    def _update_analysis(self):
        """Re-analyze data with current settings (side, smoothing, window)."""
        if not self._angle_data:
            return

        effective_side = self._get_effective_analysis_side()

        # Get the appropriate angle key based on the resolved analysis side
        if effective_side == "left":
            angle_key = "left_knee_flexion"
        elif effective_side == "right":
            angle_key = "right_knee_flexion"
        else:
            angle_key = "knee_flexion"

        # Extract raw angles for selected side
        raw_angles = [rec.get(angle_key) for rec in self._angle_data]

        # Apply smoothing if enabled
        if self._smoothing_enabled:
            smoothed_angles = self._apply_smoothing(raw_angles, self._smoothing_window)
        else:
            smoothed_angles = raw_angles

        # Update plot with new data
        plot_side = self._active_side if self._active_side != "auto" else effective_side
        self._plot_angles(raw_angles, smoothed_angles, plot_side)

        # Perform fit assessment if we have TDC/BDC frames
        if self._tdc_frames and self._bdc_frames:
            self._perform_fit_assessment()

    def _apply_smoothing(self, angles: list, window: int) -> list:
        """Apply Savitzky-Golay smoothing to angle data."""
        # Convert None to NaN
        values_array = np.array([v if v is not None else np.nan for v in angles])

        # Only smooth if we have enough valid data points
        valid_mask = ~np.isnan(values_array)
        if np.sum(valid_mask) < window:
            return angles

        # Interpolate NaN values for smoothing
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return angles

        interpolated = values_array.copy()
        for i in range(len(values_array)):
            if np.isnan(values_array[i]) and len(valid_indices) > 0:
                # Use nearest valid value
                nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - i))]
                interpolated[i] = values_array[nearest_idx]

        # Apply filter
        try:
            polyorder = 3
            smoothed = savgol_filter(interpolated, window, polyorder)
            # Restore NaN where original was None
            smoothed[~valid_mask] = np.nan
            return smoothed.tolist()
        except:
            return angles

    def _plot_angles(self, raw_angles: list, smoothed_angles: list, side: str):
        """Plot angle data and detect TDC/BDC frames."""
        print("[Analysis] Detecting TDC/BDC frames")
        try:
            # Close any existing plot
            plt.close('all')

            # Keep the latest analysis plots available for inspection.
            analysis_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "analysis_plots")
            os.makedirs(analysis_dir, exist_ok=True)

            plt.figure(figsize=(14, 7))
            frames = np.array(range(len(raw_angles)))

            # Convert None to NaN for plotting
            raw = np.array([v if v is not None else np.nan for v in raw_angles])
            smoothed = np.array([v if v is not None else np.nan for v in smoothed_angles])

            # Find peaks (maxima - top of the wave, maximum extension)
            # Remove NaN values for peak detection
            valid_mask = ~np.isnan(smoothed)
            valid_frames = frames[valid_mask]
            valid_smoothed = smoothed[valid_mask]

            # Reset TDC/BDC frames
            self._tdc_frames = []
            self._bdc_frames = []

            if len(valid_smoothed) > 10:  # Need enough data points
                # Find BDC (Bottom Dead Centre) - maximum extension (maxima in angle)
                bdc_peaks, _ = find_peaks(valid_smoothed, distance=10, prominence=5)

                # Find TDC (Top Dead Centre) - maximum flexion (minima in angle)
                tdc_peaks, _ = find_peaks(-valid_smoothed, distance=10, prominence=5)

                # Plot main data
                plt.plot(frames, raw, 'o-', alpha=0.3, label='Raw knee_flexion', markersize=2, color='lightblue')
                plt.plot(frames, smoothed, '-', linewidth=2, label='Smoothed knee_flexion', color='blue')

                # Highlight BDC (Bottom Dead Centre - maximum extension)
                if len(bdc_peaks) > 0:
                    bdc_frames = valid_frames[bdc_peaks]
                    bdc_values = valid_smoothed[bdc_peaks]
                    plt.plot(bdc_frames, bdc_values, 'ro', markersize=10,
                            label=f'BDC - Max Extension ({len(bdc_peaks)})', zorder=5)

                    # Annotate BDC frames
                    for pf, pv in zip(bdc_frames, bdc_values):
                        plt.annotate(f'{int(pf)}', xy=(pf, pv), xytext=(0, 10),
                                   textcoords='offset points', ha='center', fontsize=8, color='red')

                    # Save BDC frames
                    self._bdc_frames = bdc_frames.tolist()

                # Highlight TDC (Top Dead Centre - maximum flexion)
                if len(tdc_peaks) > 0:
                    tdc_frames = valid_frames[tdc_peaks]
                    tdc_values = valid_smoothed[tdc_peaks]
                    plt.plot(tdc_frames, tdc_values, 'go', markersize=10,
                            label=f'TDC - Max Flexion ({len(tdc_peaks)})', zorder=5)

                    # Annotate TDC frames
                    for vf, vv in zip(tdc_frames, tdc_values):
                        plt.annotate(f'{int(vf)}', xy=(vf, vv), xytext=(0, -15),
                                   textcoords='offset points', ha='center', fontsize=8, color='green')

                    # Save TDC frames
                    self._tdc_frames = tdc_frames.tolist()

                print(f"\n[Analysis] Pedal stroke detection (Side: {side}):")
                print(f"  TDC frames (Top Dead Centre - max flexion): {self._tdc_frames if len(tdc_peaks) > 0 else 'None'}")
                print(f"  BDC frames (Bottom Dead Centre - max extension): {self._bdc_frames if len(bdc_peaks) > 0 else 'None'}")
                print(f"  Total pedal strokes detected: {max(len(self._tdc_frames), len(self._bdc_frames))}")

            else:
                # Not enough data for peak detection
                plt.plot(frames, raw, 'o-', alpha=0.5, label='Raw knee_flexion', markersize=3)
                plt.plot(frames, smoothed, '-', linewidth=2, label='Smoothed knee_flexion')

            plt.xlabel('Frame')
            plt.ylabel('Angle (degrees)')
            side_text = f" ({side.upper()})" if side != "auto" else ""
            plt.title(f'Knee Flexion: Wave Analysis with Peak Detection{side_text}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot for inspection instead of showing it.
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_filename = f"angle_analysis_{side}_{timestamp}.png"
            plot_path = os.path.join(analysis_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close('all')

            print(f"[Analysis] Plot saved to: {plot_path}")
            print(f"[Analysis] Plot updated with settings: side={side}, smoothing={self._smoothing_enabled}, window={self._smoothing_window}")
        except Exception as e:
            print(f"\n[Analysis] Could not save plot: {e}")

    def _prompt_smart_calibration(self):
        """
        Prompt user with smart calibration dialog after video processing.
        This automatically guides them through bike setup and calibration in one flow.
        """
        from PyQt6.QtWidgets import QMessageBox
        from ui.smart_calibration_dialog import SmartCalibrationDialog

        print("[Calibration] Prompting user with smart calibration dialog")

        # Check if perspective correction was enabled during processing
        needs_reprocessing = not self._perspective_correction_enabled

        # Show info prompt first
        msg = QMessageBox(self)
        msg.setWindowTitle("Calibrate Body Measurements?")
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("<b>Estimate body measurements in millimeters?</b>")

        info_text = (
            "We can calibrate using your bike's wheelbase to estimate:\n\n"
            "• Body measurements in millimeters\n"
            "• Video scale based on your bike geometry\n"
            "\n"
        )

        if needs_reprocessing:
            info_text += (
                "⚠️ For best results, we recommend enabling perspective correction.\n"
                "This will reprocess the video (~30 seconds more).\n\n"
                "Continue with calibration?"
            )
        else:
            info_text += "This takes about 30 seconds."

        msg.setInformativeText(info_text)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)

        if msg.exec() != QMessageBox.StandardButton.Yes:
            print("[Calibration] User declined calibration")
            return

        # If perspective correction wasn't enabled, offer to enable it and reprocess
        if needs_reprocessing:
            reprocess_msg = QMessageBox(self)
            reprocess_msg.setWindowTitle("Enable Perspective Correction?")
            reprocess_msg.setIcon(QMessageBox.Icon.Question)
            reprocess_msg.setText("<b>Enable perspective correction for better calibration?</b>")
            reprocess_msg.setInformativeText(
                "Perspective correction improves calibration accuracy by correcting\n"
                "camera angle distortion before measurements.\n\n"
                "This requires reprocessing the video.\n\n"
                "Recommended: Yes (Reprocess with correction)\n"
                "Alternative: No (Calibrate without correction - less accurate)"
            )
            reprocess_msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            reprocess_msg.setDefaultButton(QMessageBox.StandardButton.Yes)

            result = reprocess_msg.exec()

            if result == QMessageBox.StandardButton.Cancel:
                print("[Calibration] User cancelled calibration")
                return
            elif result == QMessageBox.StandardButton.Yes:
                # Enable perspective correction and reprocess
                print("[Calibration] Enabling perspective correction and reprocessing...")
                self._perspective_check.setChecked(True)
                self._perspective_correction_enabled = True

                # Auto-prompt flow is disabled; no retry flag is needed here.

                # Update status
                self._status.setText("Reprocessing with perspective correction for calibration...")

                # Reprocess the video
                self._run_inference()

                # Calibration will be prompted again after reprocessing finishes
                return

        # Show smart calibration dialog
        calibration_homography = self._get_calibration_homography() if self._perspective_correction_enabled else None
        if self._per_frame_perspective_enabled and calibration_homography is not None:
            self._status.setText("Calibration will use a representative per-frame perspective correction.")

        dialog = SmartCalibrationDialog(
            self._video_path,
            calibration_homography,
            self
        )

        if dialog.exec() == dialog.DialogCode.Accepted:
            scale_factor = dialog.get_scale_factor()
            if scale_factor is not None:
                self._scale_factor = scale_factor
                print(f"[Calibration] ✓ Smart calibration successful! Scale: {scale_factor:.4f} mm/px")

                # Update status
                self._calibration_status.setVisible(True)
                self._calibration_status.setText(f"Calibrated: {scale_factor:.4f} mm/px")
                self._status.setText(f"Calibration complete! Scale: {scale_factor:.4f} mm/px")

                # Re-run fit assessment with body measurements
                print("[Calibration] Re-running fit assessment with body measurements...")
                self._perform_fit_assessment()
            else:
                print("[Calibration] ✗ Smart calibration failed")
                self._status.setText("Calibration completed without scale factor")
        else:
            print("[Calibration] User cancelled smart calibration")
            self._status.setText("Calibration skipped - continuing with angle-only analysis")


    def _perform_fit_assessment(self):
        """Perform fit assessment, extract representative frames, and update UI."""
        print(f"[Analysis] Fit assessment input: {len(self._tdc_frames)} TDC frames, {len(self._bdc_frames)} BDC frames")
        if not self._angle_data or not self._tdc_frames or not self._bdc_frames:
            print("[Analysis] Skipping fit assessment - missing data")
            return

        try:
            effective_side = self._get_effective_analysis_side()

            # Select representative frames
            rep_frames = select_representative_frames(
                self._angle_data,
                self._tdc_frames,
                self._bdc_frames,
                effective_side
            )
            self._representative_frames = rep_frames

            # Extract and annotate representative frames
            consistency_metrics = rep_frames['consistency_metrics']

            # Assess fit
            assessment = assess_fit(consistency_metrics, self._current_discipline)
            self._fit_assessment_result = assessment

            # Compute body measurements if calibrated
            if self._scale_factor is not None:
                print(f"\n[Measurements] Computing body measurements with scale {self._scale_factor:.4f} mm/px")
                from processing.body_measurements import compute_average_measurements

                # Compute average measurements across BDC frames
                body_measurements = compute_average_measurements(
                    self._angle_data,
                    self._scale_factor,
                    self._bdc_frames,
                    side=effective_side if effective_side in ("left", "right") else None
                )

                # Store measurements in assessment for later use
                assessment['body_measurements'] = body_measurements
                self._body_measurements = body_measurements
                self._update_measurements_summary(body_measurements)

                # Print measurements
                print(f"\n=== Body Measurements ===")
                for key, value in body_measurements.items():
                    if value is not None and '_std' not in key and '_min' not in key and '_max' not in key:
                        std = body_measurements.get(f'{key}_std')
                        if std is not None:
                            print(f"  {key}: {value:.1f} mm (±{std:.1f})")
                        else:
                            print(f"  {key}: {value:.1f} mm")
            else:
                self._body_measurements = None
                self._clear_measurements_summary()

            # Extract and display BDC frame
            if rep_frames['bdc_representative']:
                bdc_frame = self._extract_frame_at_index(rep_frames['bdc_representative']['frame_idx'])
                if bdc_frame is not None:
                    bdc_kpts = np.array(rep_frames['bdc_representative']['keypoints'])
                    bdc_angles = rep_frames['bdc_representative']['angles']

                    # Determine actual side to use for visualization
                    viz_side = self._get_visualization_side(rep_frames['bdc_representative']['frame_idx'])

                    bdc_annotated = annotate_frame_with_angles(
                        bdc_frame, bdc_kpts, bdc_angles, assessment['assessments'], viz_side
                    )
                    self._bdc_display.set_frame(
                        bdc_annotated,
                        rep_frames['bdc_representative']['frame_idx'],
                        bdc_angles,
                        assessment,
                        viz_side
                    )

            # Update fit summary widget with assessment and recommendations
            recommendations = generate_recommendations(assessment)
            self._fit_summary.set_assessment(assessment, recommendations)
            self._refresh_exported_video_overlays(assessment, effective_side)

            # Print to console
            print(f"\n=== Fit Assessment ({self._current_discipline.value}) ===")
            for angle_name, result in assessment['assessments'].items():
                measured = result.get('measured_mean')
                if measured is not None:
                    status_icon = "✓" if result['status'] == 'in_range' else "✗"
                    print(f"{status_icon} {angle_name}: {measured:.1f}° "
                          f"(target: {result['target_min']}-{result['target_max']}°)")
                else:
                    print(f"✗ {angle_name}: Not detected")

            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. [{rec['category']}] {rec['action']}")

        except Exception as e:
            print(f"\n[ERROR] Fit assessment failed: {e}")
            import traceback
            traceback.print_exc()

    def _clear_measurements_summary(self):
        """Hide the measurements summary until calibrated data is available."""
        self._bdc_display.clear_measurements()

    def _update_measurements_summary(self, body_measurements: dict):
        """Show a compact summary of the calibrated body measurements."""
        self._bdc_display.set_measurements(body_measurements)

    def _refresh_exported_video_overlays(self, assessment: dict, side: str):
        """Re-render the exported video so it matches the assessed overlay style (background worker)."""
        if not self._exported_video_path or not self._video_path or not self._angle_data:
            return

        signature = (
            side,
            self._current_discipline.value,
            tuple(
                round(assessment["assessments"].get(key, {}).get("measured_mean") or -1, 2)
                for key in ("knee_extension_bdc", "hip_angle_bdc", "elbow_flexion", "back_angle")
            ),
            round(self._scale_factor or -1, 4),
        )
        if self._annotated_video_signature == signature:
            return

        print("[Export] Refreshing annotated video with assessed angle labels (background)...")
        self._status.setText("Rendering annotated video in background...")

        # Stop any existing export worker
        if self._export_worker and self._export_worker.isRunning():
            self._export_worker.stop()
            self._export_worker.wait()

        # Create and start background export worker
        from workers.export_worker import ExportWorker

        self._export_worker = ExportWorker(
            self._video_path,
            self._exported_video_path,
            self._angle_data,
            assessment,
            side,
            self._homography_matrix,
            self._per_frame_homographies,
            self._per_frame_wheel_ellipses,
            self._fixed_perspective_warp_enabled,
            self._normalize_wheelbase_view_enabled
        )

        # Connect signals
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished.connect(lambda path: self._on_export_finished(path, signature))
        self._export_worker.error.connect(self._on_export_error)

        # Start worker
        self._export_worker.start()

    def _on_export_progress(self, percent: int):
        """Update status during background export."""
        self._status.setText(f"Rendering annotated video... {percent}%")

    def _on_export_finished(self, path: str, signature: tuple):
        """Handle export completion."""
        self._annotated_video_signature = signature
        print("[Export] Annotated video updated with angle overlays")
        if "Rendering annotated video" in self._status.text():
            self._status.setText("Fit analysis complete.")

    def _on_export_error(self, error_msg: str):
        """Handle export error."""
        print(f"[Export] Error: {error_msg}")
        if "Rendering annotated video" in self._status.text():
            self._status.setText(f"Export error: {error_msg}")

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

    def _get_effective_analysis_side(self) -> str:
        """Resolve the side to analyze for the current dataset."""
        if self._active_side in ("left", "right"):
            return self._active_side

        detected_left = sum(1 for rec in self._angle_data if rec.get("detected_side") == "left")
        detected_right = sum(1 for rec in self._angle_data if rec.get("detected_side") == "right")

        if detected_right > detected_left:
            return "right"
        if detected_left > detected_right:
            return "left"

        left_valid = sum(1 for rec in self._angle_data if rec.get("left_knee_flexion") is not None)
        right_valid = sum(1 for rec in self._angle_data if rec.get("right_knee_flexion") is not None)

        if right_valid > left_valid:
            return "right"
        if left_valid > right_valid:
            return "left"

        # Fall back to the legacy mixed stream only if neither side has usable data.
        return "auto"

    def _get_visualization_side(self, frame_idx: int) -> str:
        """
        Determine which side to visualize based on user selection and frame data.

        Args:
            frame_idx: Index of the frame

        Returns:
            "left", "right", or None for both sides
        """
        if self._active_side == "left":
            return "left"
        elif self._active_side == "right":
            return "right"
        elif self._active_side == "auto":
            resolved_side = self._get_effective_analysis_side()
            if resolved_side in ["left", "right"]:
                return resolved_side
            return None
        else:
            return None

    def _extract_frame_at_index(self, frame_idx: int) -> np.ndarray | None:
        """
        Extract a single frame from the video file.

        If perspective correction was applied during processing, this method
        will apply the same transformation to maintain consistency between
        the frame and the keypoints that were computed on corrected frames.
        """
        if not self._video_path:
            return None

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Apply the same perspective correction used during processing.
        homography = self._per_frame_homographies.get(frame_idx, self._homography_matrix)
        if homography is not None:
            print(f"[Perspective] Applying perspective correction to extracted frame {frame_idx}")
            from processing.perspective_correction import apply_perspective_correction, deserialize_ellipses
            frame = apply_perspective_correction(
                frame,
                homography,
                fixed_output=self._fixed_perspective_warp_enabled,
                normalize_wheelbase=self._normalize_wheelbase_view_enabled,
                wheel_ellipses=deserialize_ellipses(
                    self._per_frame_wheel_ellipses.get(frame_idx)
                )
            )

        return frame

    def _update_assessment_cards(self, assessment: dict):
        """Create or update assessment cards in a 2-column grid."""
        # Clear existing cards
        for card in self._assessment_cards:
            card.deleteLater()
        self._assessment_cards.clear()

        # Create new cards for each assessment
        assessments = assessment['assessments']
        angle_order = ['knee_extension_bdc', 'knee_flexion_bdc', 'hip_angle_bdc', 'elbow_flexion', 'back_angle']

        row = 0
        col = 0
        for angle_name in angle_order:
            if angle_name in assessments:
                card = AssessmentCard(angle_name)
                card.set_assessment(assessments[angle_name])
                self._cards_layout.addWidget(card, row, col)
                self._assessment_cards.append(card)

                # Move to next position (2 columns)
                col += 1
                if col >= 2:
                    col = 0
                    row += 1
