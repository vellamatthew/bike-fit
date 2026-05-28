"""Dialog for searching and selecting bike geometry."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QMessageBox,
    QProgressDialog, QHeaderView, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from bike_geometry import search_bike, get_bike_geometry, BikeGeometryStorage


class SearchWorker(QThread):
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


class GeometryWorker(QThread):
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


class BikeGeometryDialog(QDialog):
    """Dialog for searching and selecting bike models."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bike Geometry Search")
        self.setMinimumSize(800, 600)
        self._search_results: list[dict] = []
        self._selected_bike_url: str | None = None
        self._bike_geometry: dict | None = None
        self._search_worker: SearchWorker | None = None
        self._geometry_worker: GeometryWorker | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("Search Bike Geometry Database")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Description
        desc = QLabel(
            "Search for your bike model to retrieve geometry measurements from geometrygeeks.bike.\n"
            "Enter the brand and model (e.g., 'ritchey ascent' or 'trek marlin')."
        )
        desc.setStyleSheet("color: #888; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Search section
        search_layout = QHBoxLayout()
        search_layout.setSpacing(8)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Enter bike brand and model (e.g., 'Trek Marlin 7')")
        self._search_input.setMinimumHeight(36)
        self._search_input.returnPressed.connect(self._on_search)
        self._search_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
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
        search_layout.addWidget(self._search_input)

        self._search_btn = QPushButton("Search")
        self._search_btn.setMinimumHeight(36)
        self._search_btn.setMinimumWidth(100)
        self._search_btn.clicked.connect(self._on_search)
        self._search_btn.setStyleSheet("""
            QPushButton {
                background: #0066cc;
                color: white;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
                padding: 0 16px;
            }
            QPushButton:hover {
                background: #0052a3;
            }
            QPushButton:disabled {
                background: #444;
                color: #888;
            }
        """)
        search_layout.addWidget(self._search_btn)

        layout.addLayout(search_layout)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self._status_label.setMinimumHeight(20)
        layout.addWidget(self._status_label)

        # Results table
        results_label = QLabel("Search Results:")
        results_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #fff;")
        layout.addWidget(results_label)

        self._results_table = QTableWidget()
        self._results_table.setColumnCount(4)
        self._results_table.setHorizontalHeaderLabels(["Brand", "Model", "Year", "URL"])
        self._results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
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
        layout.addWidget(self._results_table)

        # Size selection section (hidden initially)
        self._size_section = QHBoxLayout()
        self._size_section.setSpacing(8)

        size_label = QLabel("Select Size:")
        size_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self._size_section.addWidget(size_label)

        self._size_combo = QComboBox()
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
        self._size_section.addWidget(self._size_combo)
        self._size_section.addStretch()

        # Create widget to hold size section
        self._size_widget = QLabel("")  # Placeholder
        self._size_layout_container = QHBoxLayout()
        layout.addLayout(self._size_section)

        # Hide size section initially
        self._size_combo.setVisible(False)
        size_label.setVisible(False)

        # Store reference to size label for later
        self._size_label = size_label

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._select_btn = QPushButton("Select Bike")
        self._select_btn.setEnabled(False)
        self._select_btn.setMinimumHeight(36)
        self._select_btn.setMinimumWidth(120)
        self._select_btn.clicked.connect(self._on_select)
        self._select_btn.setStyleSheet("""
            QPushButton {
                background: #00aa00;
                color: white;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
                padding: 0 20px;
            }
            QPushButton:hover {
                background: #008800;
            }
            QPushButton:disabled {
                background: #444;
                color: #888;
            }
        """)
        button_layout.addWidget(self._select_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(36)
        cancel_btn.setMinimumWidth(100)
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #444;
                color: white;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
                padding: 0 20px;
            }
            QPushButton:hover {
                background: #555;
            }
        """)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def _on_search(self):
        """Handle search button click."""
        query = self._search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Empty Query", "Please enter a bike brand and model to search.")
            return

        # Disable UI during search
        self._search_btn.setEnabled(False)
        self._search_input.setEnabled(False)
        self._status_label.setText(f"Searching for '{query}'...")
        self._results_table.setRowCount(0)
        self._select_btn.setEnabled(False)

        # Start search worker
        self._search_worker = SearchWorker(query)
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.error.connect(self._on_search_error)
        self._search_worker.start()

    def _on_search_finished(self, results: list[dict]):
        """Handle search results."""
        self._search_results = results
        self._search_btn.setEnabled(True)
        self._search_input.setEnabled(True)

        if not results:
            self._status_label.setText("No bikes found. Try a different search query.")
            return

        self._status_label.setText(f"Found {len(results)} bike(s)")

        # Populate table
        self._results_table.setRowCount(len(results))
        for row, bike in enumerate(results):
            self._results_table.setItem(row, 0, QTableWidgetItem(bike['brand']))
            self._results_table.setItem(row, 1, QTableWidgetItem(bike['model']))
            self._results_table.setItem(row, 2, QTableWidgetItem(bike['year']))
            self._results_table.setItem(row, 3, QTableWidgetItem(bike['url']))

    def _on_search_error(self, error_msg: str):
        """Handle search error."""
        self._search_btn.setEnabled(True)
        self._search_input.setEnabled(True)
        self._status_label.setText(f"Search failed: {error_msg}")
        QMessageBox.critical(self, "Search Error", f"Failed to search bikes:\n{error_msg}")

    def _on_selection_changed(self):
        """Handle table selection change."""
        selected_rows = self._results_table.selectedItems()
        if selected_rows:
            row = self._results_table.currentRow()
            if 0 <= row < len(self._search_results):
                self._selected_bike_url = self._search_results[row]['url']
                self._fetch_geometry(self._selected_bike_url)

    def _fetch_geometry(self, url: str):
        """Fetch full geometry for selected bike."""
        self._status_label.setText("Fetching bike geometry...")
        self._select_btn.setEnabled(False)
        self._size_combo.setVisible(False)
        self._size_label.setVisible(False)

        # Start geometry worker
        self._geometry_worker = GeometryWorker(url)
        self._geometry_worker.finished.connect(self._on_geometry_finished)
        self._geometry_worker.error.connect(self._on_geometry_error)
        self._geometry_worker.start()

    def _on_geometry_finished(self, geometry: dict):
        """Handle geometry fetch completion."""
        self._bike_geometry = geometry
        bike_name = geometry.get('bike_name', 'Unknown')
        sizes = geometry.get('sizes', [])

        self._status_label.setText(f"Loaded geometry for {bike_name}")

        # Populate size dropdown
        if sizes:
            self._size_combo.clear()
            self._size_combo.addItems(sizes)
            self._size_combo.setVisible(True)
            self._size_label.setVisible(True)
        else:
            self._size_combo.setVisible(False)
            self._size_label.setVisible(False)

        self._select_btn.setEnabled(True)

    def _on_geometry_error(self, error_msg: str):
        """Handle geometry fetch error."""
        self._status_label.setText(f"Failed to load geometry: {error_msg}")
        QMessageBox.critical(self, "Geometry Error", f"Failed to fetch bike geometry:\n{error_msg}")

    def _on_select(self):
        """Handle select button click."""
        if not self._bike_geometry:
            return

        # Get selected size
        selected_size = None
        if self._size_combo.isVisible() and self._size_combo.count() > 0:
            selected_size = self._size_combo.currentText()

        # Store in singleton
        storage = BikeGeometryStorage()
        storage.set_bike_data(self._bike_geometry, selected_size)

        # Show confirmation with measurements
        bike_name = self._bike_geometry.get('bike_name', 'Unknown')
        size_text = f" (Size: {selected_size})" if selected_size else ""
        measurements = self._bike_geometry.get('measurements', {})

        # Build measurements text
        measurements_text = "\nMeasurements:\n"
        if selected_size:
            # Show measurements for selected size only
            for param_name, size_values in measurements.items():
                value = size_values.get(selected_size, 'N/A')
                measurements_text += f"  {param_name}: {value}\n"
        else:
            # Show all measurements for all sizes
            for param_name, size_values in measurements.items():
                measurements_text += f"  {param_name}: {size_values}\n"

        QMessageBox.information(
            self,
            "Bike Selected",
            f"Successfully stored geometry for:\n{bike_name}{size_text}\n"
            f"{measurements_text}"
        )

        self.accept()

    def get_selected_geometry(self) -> dict | None:
        """Return the selected bike geometry (for external access)."""
        return self._bike_geometry

    def get_selected_size(self) -> str | None:
        """Return the selected size."""
        if self._size_combo.isVisible() and self._size_combo.count() > 0:
            return self._size_combo.currentText()
        return None
