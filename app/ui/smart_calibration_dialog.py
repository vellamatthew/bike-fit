"""
Unified calibration dialog for bike setup and scale calibration.

This dialog lets the user:
1. Search for bike geometry
2. Save the selected bike
3. Calibrate video scale for estimated body measurements
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QStackedWidget, QProgressBar, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from bike_geometry import search_bike, get_bike_geometry, BikeGeometryStorage
from processing.calibration import WheelCalibration


class BikeSearchWorker(QThread):
    """Background thread for bike search."""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self):
        try:
            results = search_bike(self.query)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class BikeGeometryWorker(QThread):
    """Background thread for fetching bike geometry."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def run(self):
        try:
            geometry = get_bike_geometry(self.url)
            self.finished.emit(geometry)
        except Exception as e:
            self.error.emit(str(e))


class CalibrationWorker(QThread):
    """Background thread for running calibration."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(float)  # scale_factor
    error = pyqtSignal(str)

    def __init__(self, video_path: str, wheelbase_mm: float, homography=None):
        super().__init__()
        self.video_path = video_path
        self.wheelbase_mm = wheelbase_mm
        self.homography = homography

    def run(self):
        try:
            if self.homography is not None:
                self.progress.emit("Analyzing perspective-corrected frames for wheel detection...")
            else:
                self.progress.emit("Analyzing video frames for wheel detection...")

            # Use WheelCalibration class
            calibrator = WheelCalibration(self.video_path, self.wheelbase_mm, homography=self.homography)
            scale_factor = calibrator.calibrate(max_frames_to_try=100, verbose=True)

            if scale_factor is None:
                self.error.emit("Could not detect bike wheels for calibration")
            else:
                self.finished.emit(scale_factor)
        except Exception as e:
            self.error.emit(str(e))


class SmartCalibrationDialog(QDialog):
    """
    Unified dialog for bike setup and calibration.

    Flow:
    1. Search for bike (or skip if already set)
    2. Select size
    3. Run calibration
    4. Show results
    """

    def __init__(self, video_path: str | None, homography=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Setup")
        self.setModal(True)
        self.setMinimumSize(800, 620)

        self._video_path = video_path
        self._bike_only_mode = video_path is None
        self._homography = homography
        self._bike_geometry = None
        self._selected_size = None
        self._scale_factor = None
        self._search_results = []

        # Check if bike is already configured
        storage = BikeGeometryStorage()
        self._has_existing_bike = storage.has_data()

        if self._has_existing_bike and not self._bike_only_mode:
            self._bike_geometry = storage.get_bike_data()
            self._selected_size = storage.get_selected_size()

        self._build_ui()

        # Always start on the bike-selection page so the user can keep or change the bike.
        self._show_search_page()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Stacked widget for different pages
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, stretch=1)

        # Page 0: Bike search
        self._search_page = self._create_search_page()
        self._stack.addWidget(self._search_page)

        # Page 1: Calibration in progress
        self._calibration_page = self._create_calibration_page()
        self._stack.addWidget(self._calibration_page)

        # Bottom buttons
        self._button_layout = QHBoxLayout()
        self._button_layout.setSpacing(12)

        self._back_btn = QPushButton("Back")
        self._back_btn.setMinimumHeight(40)
        self._back_btn.setVisible(False)
        self._back_btn.setStyleSheet("""
            QPushButton {
                background: #444;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0 24px;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #555;
            }
        """)
        self._back_btn.clicked.connect(self._on_back)
        self._button_layout.addWidget(self._back_btn)

        self._button_layout.addStretch()

        self._skip_btn = QPushButton("Skip")
        self._skip_btn.setMinimumHeight(40)
        self._skip_btn.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                color: white;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 0 24px;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #4a4a4a;
            }
        """)
        self._skip_btn.clicked.connect(self.reject)
        self._button_layout.addWidget(self._skip_btn)

        self._next_btn = QPushButton("Search")
        self._next_btn.setMinimumHeight(40)
        self._next_btn.setStyleSheet("""
            QPushButton {
                background: #0066cc;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0 24px;
                font-size: 13px;
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
        self._next_btn.clicked.connect(self._on_next)
        self._button_layout.addWidget(self._next_btn)

        layout.addLayout(self._button_layout)

    def _create_search_page(self):
        """Page for searching and selecting bike."""
        page = QFrame()
        layout = QVBoxLayout(page)
        layout.setSpacing(16)

        info_label = QLabel(
            "Uses your bike's wheelbase to estimate body measurements in millimeters from the video. "
            "It usually works best when perspective correction is enabled first."
        )
        info_label.setStyleSheet("color: #cfcfcf; font-size: 12px; line-height: 1.4;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Search input
        search_label = QLabel("Search for your bike:")
        search_label.setStyleSheet("color: #fff; font-size: 13px; font-weight: bold;")
        layout.addWidget(search_label)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("e.g., 'Trek Marlin 7' or 'Specialized Allez'")
        self._search_input.setMinimumHeight(40)
        self._search_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #444;
                border-radius: 6px;
                background: #2a2a2a;
                color: #fff;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #0066cc;
            }
        """)
        self._search_input.returnPressed.connect(self._on_search)
        search_row.addWidget(self._search_input)

        self._search_btn = QPushButton("Search")
        self._search_btn.setMinimumHeight(40)
        self._search_btn.setMinimumWidth(110)
        self._search_btn.clicked.connect(self._on_search)
        self._search_btn.setStyleSheet("""
            QPushButton {
                background: #0066cc;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0 18px;
                font-size: 13px;
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
        search_row.addWidget(self._search_btn)
        layout.addLayout(search_row)

        # Status
        self._search_status = QLabel("")
        self._search_status.setStyleSheet("color: #aaa; font-size: 11px;")
        self._search_status.setMinimumHeight(20)
        layout.addWidget(self._search_status)

        # Results area
        results_label = QLabel("Results:")
        results_label.setStyleSheet("color: #fff; font-size: 12px; font-weight: bold;")
        layout.addWidget(results_label)

        self._results_table = QTableWidget()
        self._results_table.setColumnCount(4)
        self._results_table.setHorizontalHeaderLabels(["Brand", "Model", "Year", "URL"])
        self._results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results_table.setAlternatingRowColors(False)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._results_table.setStyleSheet("""
            QTableWidget {
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 6px;
                gridline-color: #444;
            }
            QTableWidget::item {
                padding: 8px;
                color: #fff;
            }
            QTableWidget::item:selected {
                background: #0066cc;
            }
            QHeaderView::section {
                background: #1a1a1a;
                color: #aaa;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #444;
                font-weight: bold;
            }
        """)
        self._results_table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._results_table, stretch=1)

        # Size selection
        size_layout = QHBoxLayout()
        self._size_label = QLabel("Select size:")
        self._size_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self._size_label.setVisible(False)
        size_layout.addWidget(self._size_label)

        self._size_combo = QComboBox()
        self._size_combo.setMinimumHeight(36)
        self._size_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                color: #fff;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #888;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                color: #fff;
                selection-background-color: #0066cc;
                border: 1px solid #444;
            }
        """)
        self._size_combo.setVisible(False)
        size_layout.addWidget(self._size_combo)
        size_layout.addStretch()

        layout.addLayout(size_layout)

        return page

    def _create_calibration_page(self):
        """Page showing calibration in progress."""
        page = QFrame()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 60, 40, 60)
        layout.setSpacing(24)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Title
        title = QLabel("Calibrating...")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #fff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Status
        self._calib_status = QLabel("Analyzing video frames for wheel detection")
        self._calib_status.setStyleSheet("color: #aaa; font-size: 13px;")
        self._calib_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._calib_status.setWordWrap(True)
        layout.addWidget(self._calib_status)

        # Progress bar
        self._calib_progress = QProgressBar()
        self._calib_progress.setMinimum(0)
        self._calib_progress.setMaximum(0)  # Indeterminate
        self._calib_progress.setMinimumHeight(8)
        self._calib_progress.setTextVisible(False)
        self._calib_progress.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 4px;
                background: #2a2a2a;
            }
            QProgressBar::chunk {
                background: #0066cc;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._calib_progress)

        layout.addStretch()

        return page

    def _create_success_page(self):
        """Page showing successful calibration."""
        page = QFrame()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 60, 40, 60)
        layout.setSpacing(24)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Success icon
        icon = QLabel("✓")
        icon.setStyleSheet("font-size: 72px; color: #0f0;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        # Title
        title = QLabel("Calibration Complete!")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #0f0;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Details frame
        details_frame = QFrame()
        details_frame.setStyleSheet("""
            QFrame {
                background: #1a3a1a;
                border: 2px solid #0f0;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        details_layout = QVBoxLayout(details_frame)
        details_layout.setSpacing(12)

        self._success_bike_label = QLabel("Bike: Loading...")
        self._success_bike_label.setStyleSheet("color: #fff; font-size: 13px;")
        self._success_bike_label.setWordWrap(True)
        details_layout.addWidget(self._success_bike_label)

        self._success_scale_label = QLabel("Scale: Loading...")
        self._success_scale_label.setStyleSheet("color: #0f0; font-size: 13px; font-weight: bold;")
        details_layout.addWidget(self._success_scale_label)

        benefits_label = QLabel(
            "\n<b>Now enabled:</b><br>"
            "• Real-world measurements (mm)<br>"
            "• Body proportion analysis<br>"
            "• Precise adjustment recommendations"
        )
        benefits_label.setStyleSheet("color: #8f8; font-size: 12px;")
        benefits_label.setWordWrap(True)
        details_layout.addWidget(benefits_label)

        layout.addWidget(details_frame)
        layout.addStretch()

        return page

    def _show_search_page(self):
        """Show the search page."""
        self._stack.setCurrentIndex(0)
        if self._bike_geometry:
            bike_name = self._bike_geometry.get("bike_name", "Current bike")
            if self._selected_size:
                self._search_status.setText(f"Current bike: {bike_name} ({self._selected_size})")
            else:
                self._search_status.setText(f"Current bike: {bike_name}")
        else:
            self._search_status.setText("")
        if self._bike_only_mode:
            self._skip_btn.setText("Close")
        else:
            self._skip_btn.setText("Skip")
        self._next_btn.setText("Continue")
        self._next_btn.setEnabled(self._bike_geometry is not None)
        self._next_btn.setVisible(True)
        self._skip_btn.setVisible(True)
        self._back_btn.setVisible(False)

    def _show_calibration_page(self):
        """Show calibration progress page."""
        self._stack.setCurrentIndex(1)
        self._next_btn.setVisible(False)
        self._skip_btn.setVisible(False)
        self._back_btn.setVisible(False)

        # Start calibration
        self._run_calibration()

    def _show_success_page(self):
        """Show success page."""
        self._stack.setCurrentIndex(2)
        self._next_btn.setText("Done")
        self._next_btn.setVisible(True)
        self._skip_btn.setVisible(False)
        self._back_btn.setVisible(False)

        # Reconnect Done button to close dialog
        try:
            self._next_btn.clicked.disconnect()
        except TypeError:
            pass
        self._next_btn.clicked.connect(self.accept)

    def _on_search(self):
        """Handle search action."""
        query = self._search_input.text().strip()
        if not query:
            self._search_status.setText("Please enter a bike model")
            return

        self._search_status.setText(f"Searching for '{query}'...")
        self._next_btn.setEnabled(False)
        self._search_btn.setEnabled(False)
        self._search_input.setEnabled(False)
        self._results_table.clearContents()
        self._results_table.setRowCount(0)
        self._bike_geometry = None
        self._selected_size = None
        self._size_combo.setVisible(False)
        self._size_label.setVisible(False)

        # Start search
        self._search_worker = BikeSearchWorker(query)
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.error.connect(self._on_search_error)
        self._search_worker.start()

    def _on_search_finished(self, results: list):
        """Handle search results."""
        self._search_results = results
        self._search_btn.setEnabled(True)
        self._search_input.setEnabled(True)
        self._results_table.clearContents()
        self._results_table.setRowCount(0)

        if not results:
            self._search_status.setText("No bikes found. Try a different search term.")
            return

        self._search_status.setText(f"Found {len(results)} bike(s). Select one to continue.")
        self._results_table.setRowCount(len(results))
        for row, bike in enumerate(results):
            self._results_table.setItem(row, 0, QTableWidgetItem(bike["brand"]))
            self._results_table.setItem(row, 1, QTableWidgetItem(bike["model"]))
            self._results_table.setItem(row, 2, QTableWidgetItem(bike["year"]))
            self._results_table.setItem(row, 3, QTableWidgetItem(bike["url"]))

    def _on_search_error(self, error: str):
        """Handle search error."""
        self._search_status.setText(f"Search failed: {error}")
        self._search_btn.setEnabled(True)
        self._search_input.setEnabled(True)

    def _on_selection_changed(self):
        """Handle bike selection from the results table."""
        selected_rows = self._results_table.selectedItems()
        if not selected_rows:
            return

        row = self._results_table.currentRow()
        if not (0 <= row < len(self._search_results)):
            return

        result = self._search_results[row]
        self._search_status.setText(f"Loading geometry for {result['brand']} {result['model']}...")

        # Fetch full geometry
        self._geometry_worker = BikeGeometryWorker(result['url'])
        self._geometry_worker.finished.connect(self._on_geometry_loaded)
        self._geometry_worker.error.connect(self._on_geometry_error)
        self._geometry_worker.start()

    def _on_geometry_loaded(self, geometry: dict):
        """Handle geometry loaded."""
        self._bike_geometry = geometry
        bike_name = geometry.get('bike_name', 'Unknown')
        sizes = geometry.get('sizes', [])

        self._search_status.setText(f"✓ Selected: {bike_name}")

        # Show size selector if multiple sizes
        if sizes:
            self._size_combo.clear()
            self._size_combo.addItems(sizes)
            self._size_combo.setVisible(True)
            self._size_label.setVisible(True)
            self._selected_size = sizes[0]
        else:
            self._size_combo.setVisible(False)
            self._size_label.setVisible(False)
            self._selected_size = None

        # Update button to proceed based on current mode
        self._next_btn.setText("Save Bike" if self._bike_only_mode else "Continue")
        self._next_btn.setEnabled(True)

    def _on_geometry_error(self, error: str):
        """Handle geometry fetch error."""
        self._search_status.setText(f"Failed to load geometry: {error}")

    def _on_proceed_to_calibration(self):
        """User clicked 'Calibrate' after selecting bike."""
        if not self._bike_geometry:
            self._search_status.setText("Please select a bike first")
            return

        # Get selected size
        if self._size_combo.isVisible():
            self._selected_size = self._size_combo.currentText()

        # Save bike geometry
        storage = BikeGeometryStorage()
        storage.set_bike_data(self._bike_geometry, self._selected_size)

        if self._bike_only_mode:
            self.accept()
            return

        # Move to calibration page
        self._show_calibration_page()

    def _run_calibration(self):
        """Run the calibration process."""
        # Get wheelbase from bike geometry
        wheelbase_str = None
        if self._bike_geometry:
            measurements = self._bike_geometry.get('measurements', {})
            wheelbase_values = measurements.get('Wheelbase', {})

            if self._selected_size and self._selected_size in wheelbase_values:
                wheelbase_str = wheelbase_values[self._selected_size]
            elif wheelbase_values:
                # Use first available size
                wheelbase_str = next(iter(wheelbase_values.values()))

        if not wheelbase_str:
            self._calib_status.setText("Error: No wheelbase measurement found")
            self._on_calibration_error("No wheelbase measurement available")
            return

        # Parse wheelbase to mm
        try:
            wheelbase_mm = float(wheelbase_str.replace('mm', '').strip())
        except (ValueError, AttributeError):
            try:
                wheelbase_mm = float(wheelbase_str.strip())
            except:
                self._on_calibration_error(f"Invalid wheelbase format: {wheelbase_str}")
                return

        # Start calibration worker
        self._calib_worker = CalibrationWorker(
            self._video_path,
            wheelbase_mm,
            self._homography
        )
        self._calib_worker.progress.connect(self._on_calibration_progress)
        self._calib_worker.finished.connect(self._on_calibration_finished)
        self._calib_worker.error.connect(self._on_calibration_error)
        self._calib_worker.start()

    def _on_calibration_progress(self, message: str):
        """Update calibration progress."""
        self._calib_status.setText(message)

    def _on_calibration_finished(self, scale_factor: float):
        """Handle successful calibration."""
        self._scale_factor = scale_factor

        # Let the user confirm wheel detection before keeping calibration.
        if self._show_wheel_visualization():
            self.accept()
        else:
            self._scale_factor = None
            self.reject()

    def _show_wheel_visualization(self) -> bool:
        """Let the user confirm wheel detection before keeping calibration."""
        try:
            from processing.calibration import detect_wheels_for_calibration

            result = detect_wheels_for_calibration(
                self._video_path,
                homography=self._homography,
                max_frames_to_try=100,
                verbose=False
            )

            if result is not None:
                frame_idx, frame, ellipses = result

                wheelbase_values = self._bike_geometry.get("measurements", {}).get("Wheelbase", {}) if self._bike_geometry else {}
                wheelbase_label = ""
                if self._selected_size and self._selected_size in wheelbase_values:
                    wheelbase_label = f"Wheelbase: {wheelbase_values[self._selected_size]}"
                elif wheelbase_values:
                    wheelbase_label = f"Wheelbase: {next(iter(wheelbase_values.values()))}"

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
                        "The yellow ellipses should be centered on the two bike wheels.\n\n"
                        f"Scale factor: {self._scale_factor:.4f} mm/px"
                    ),
                    cancel_label="Cancel Calibration",
                    confirm_label="Looks Good",
                    baseline_label=wheelbase_label,
                )

                return dialog.exec() == dialog.DialogCode.Accepted
        except Exception as e:
            print(f"[Calibration] Could not show wheel visualization: {e}")

        return False

    def _on_calibration_error(self, error: str):
        """Handle calibration error."""
        self._calib_status.setText(f"Calibration failed: {error}")

        # Show retry option
        self._next_btn.setText("Retry")
        self._next_btn.setVisible(True)
        try:
            self._next_btn.clicked.disconnect()
        except TypeError:
            pass
        self._next_btn.clicked.connect(self._show_calibration_page)

        self._skip_btn.setText("Skip Calibration")
        self._skip_btn.setVisible(True)

    def _on_next(self):
        """Handle next button for the current page."""
        current_page = self._stack.currentIndex()

        if current_page == 0:  # Search page
            if self._bike_geometry:
                self._on_proceed_to_calibration()
            else:
                self._on_search()

    def _on_back(self):
        """Handle back button."""
        current_page = self._stack.currentIndex()
        if current_page > 0:
            self._stack.setCurrentIndex(current_page - 1)

    def get_scale_factor(self) -> float | None:
        """Return the computed scale factor."""
        return self._scale_factor
