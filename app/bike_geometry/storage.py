"""In-memory storage for bike geometry data."""


class BikeGeometryStorage:
    """Singleton storage for selected bike geometry data."""

    _instance = None
    _bike_data: dict | None = None
    _selected_size: str | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_bike_data(self, bike_data: dict, selected_size: str | None = None):
        """
        Store bike geometry data.

        Args:
            bike_data: Dictionary from get_bike_geometry()
            selected_size: Optional selected size (e.g., "17.5 in")
        """
        self._bike_data = bike_data
        self._selected_size = selected_size

    def get_bike_data(self) -> dict | None:
        """Get stored bike geometry data."""
        return self._bike_data

    def get_selected_size(self) -> str | None:
        """Get selected bike size."""
        return self._selected_size

    def has_data(self) -> bool:
        """Check if bike data is stored."""
        return self._bike_data is not None

    def clear(self):
        """Clear stored bike data."""
        self._bike_data = None
        self._selected_size = None

    def get_measurement(self, measurement_name: str) -> dict | None:
        """
        Get measurement values for all sizes.

        Args:
            measurement_name: Name of measurement (e.g., "Reach", "Stack")

        Returns:
            Dictionary mapping size to value, or None if not found
        """
        if not self._bike_data:
            return None

        return self._bike_data.get('measurements', {}).get(measurement_name)

    def get_measurement_for_size(self, measurement_name: str, size: str | None = None) -> str | None:
        """
        Get measurement value for a specific size.

        Args:
            measurement_name: Name of measurement
            size: Size to query (defaults to selected size)

        Returns:
            Measurement value as string, or None if not found
        """
        size_to_use = size or self._selected_size
        if not size_to_use:
            return None

        measurement_dict = self.get_measurement(measurement_name)
        if not measurement_dict:
            return None

        return measurement_dict.get(size_to_use)
