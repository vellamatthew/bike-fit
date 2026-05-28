from PyQt6.QtWidgets import (
    QMainWindow, QStatusBar, QLabel
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QCloseEvent

from ui.video_panel import VideoPanel
from bike_geometry import BikeGeometryStorage


class ModelLoader(QThread):
    """Preloads YOLO model on a background thread at startup."""
    done  = pyqtSignal()
    error = pyqtSignal(str)

    def run(self):
        try:
            from inference.yolo_pose import get_model
            get_model()
            self.done.emit()
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bike Fit Analyser")
        self.setMinimumSize(1040, 760)
        self._build_ui()
        self._preload_model()

    def _build_ui(self):
        self._video_panel = VideoPanel(on_bike_setup=self._open_bike_setup_dialog)
        self.setCentralWidget(self._video_panel)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._model_status = QLabel("Loading model…")
        self._status_bar.addPermanentWidget(self._model_status)

        # Bike geometry status
        self._bike_status = QLabel("No bike selected")
        self._bike_status.setStyleSheet("color: #888; margin-right: 12px;")
        self._status_bar.addPermanentWidget(self._bike_status)
        self._update_bike_status()

    def _preload_model(self):
        self._loader = ModelLoader()
        self._loader.done.connect(self._on_model_ready)
        self._loader.error.connect(self._on_model_error)
        self._loader.start()

    def _on_model_ready(self):
        self._model_status.setText("Model ready ✓")

    def _on_model_error(self, msg: str):
        self._model_status.setText(f"Model error: {msg}")

    def _open_bike_setup_dialog(self, video_path: str | None = None, homography=None):
        """Open consolidated bike setup dialog."""
        from ui.smart_calibration_dialog import SmartCalibrationDialog

        dialog = SmartCalibrationDialog(video_path, homography, self)
        if dialog.exec():
            self._update_bike_status()
            return dialog.get_scale_factor()
        return None

    def _search_new_bike(self, parent_dialog):
        """Open bike search dialog."""
        from ui.smart_calibration_dialog import SmartCalibrationDialog

        dialog = SmartCalibrationDialog(None, None, self)
        if dialog.exec():
            # Bike was selected - update status and close parent
            self._update_bike_status()
            parent_dialog.accept()

    def _clear_bike_from_setup(self, parent_dialog):
        """Clear bike data from setup dialog."""
        from PyQt6.QtWidgets import QMessageBox

        storage = BikeGeometryStorage()

        reply = QMessageBox.question(
            self,
            "Clear Bike Data",
            "Are you sure you want to clear the stored bike geometry?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            storage.clear()
            self._update_bike_status()
            parent_dialog.accept()

            QMessageBox.information(
                self,
                "Data Cleared",
                "Bike geometry data has been cleared."
            )

    def _update_bike_status(self):
        """Update bike status label in status bar."""
        storage = BikeGeometryStorage()
        if storage.has_data():
            bike_data = storage.get_bike_data()
            bike_name = bike_data.get('bike_name', 'Unknown')
            selected_size = storage.get_selected_size()

            # Shorten bike name if too long
            max_len = 40
            if len(bike_name) > max_len:
                bike_name = bike_name[:max_len-3] + "..."

            if selected_size:
                self._bike_status.setText(f"Bike: {bike_name} ({selected_size})")
            else:
                self._bike_status.setText(f"Bike: {bike_name}")
            self._bike_status.setStyleSheet("color: #00aa00; margin-right: 12px;")
        else:
            self._bike_status.setText("No bike selected")
            self._bike_status.setStyleSheet("color: #888; margin-right: 12px;")

    def closeEvent(self, event: QCloseEvent):
        # Stop any running workers cleanly
        if hasattr(self._video_panel, "_worker") and self._video_panel._worker:
            self._video_panel._worker.stop()
            # Wait briefly for the worker to exit cooperatively.
            if not self._video_panel._worker.wait(3000):  # 3 second timeout
                print("[WARNING] Worker thread is still shutting down.")
        event.accept()
