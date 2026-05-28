"""
Collapsible card widget showing angle assessment with color-coded status.
"""
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class AssessmentCard(QFrame):
    """Card displaying angle assessment with color-coded border."""

    def __init__(self, angle_name: str, parent=None):
        super().__init__(parent)
        self._angle_name = angle_name
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet("""
            QFrame {
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 2px;
                padding: 6px 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Content label
        self._content = QLabel("")
        self._content.setStyleSheet("color: #ddd; font-size: 10px; line-height: 1.3;")
        self._content.setWordWrap(True)
        layout.addWidget(self._content)

    def set_assessment(self, assessment: dict):
        """
        Display assessment information.

        Args:
            assessment: Dict from fit_assessment.assess_angle()
        """
        lines = []

        # Title
        display_name = self._format_angle_name(self._angle_name)
        lines.append(f"<b>{display_name}</b>")

        # Main assessment
        measured = assessment.get('measured_mean')
        if measured is not None:
            target_min = assessment['target_min']
            target_max = assessment['target_max']
            status = assessment['status']

            # Status icon and value
            if status == 'in_range':
                icon = "✓"
                color = "#0a0"
            elif assessment.get('severity'):
                severity_str = assessment['severity'].value if hasattr(assessment['severity'], 'value') else str(assessment['severity'])
                if severity_str == 'marginal':
                    icon = "⚠"
                    color = "#fa0"
                else:
                    icon = "✗"
                    color = "#f00"
            else:
                icon = "?"
                color = "#888"

            lines.append(f"<span style='color: {color};'>{icon} {measured:.1f}°</span> (target: {target_min}-{target_max}°)")

            # Range info
            measured_range = assessment.get('measured_range')
            count = assessment.get('count', 0)
            if measured_range and measured_range[0] is not None and count > 1:
                range_min, range_max = measured_range
                lines.append(f"Range: {range_min:.0f}°-{range_max:.0f}° ({count} strokes)")

            # Consistency warning
            consistency_warning = assessment.get('consistency_warning')
            if consistency_warning:
                lines.append(f"<span style='color: #fa0;'>⚠ {consistency_warning}</span>")

        else:
            lines.append("<span style='color: #f00;'>✗ Not detected</span>")

        # Update content
        self._content.setText("<br>".join(lines))

        # Update border color based on severity
        self._update_border_color(assessment)

    def _update_border_color(self, assessment: dict):
        """Update frame border color based on assessment status."""
        status = assessment.get('status')

        if status == 'in_range':
            border_color = "#0a0"  # Green
        elif status == 'not_detected':
            border_color = "#f00"  # Red
        else:
            severity = assessment.get('severity')
            if severity:
                severity_str = severity.value if hasattr(severity, 'value') else str(severity)
                if severity_str == 'marginal':
                    border_color = "#fa0"  # Orange
                else:
                    border_color = "#f00"  # Red
            else:
                border_color = "#444"  # Gray

        self.setStyleSheet(f"""
            QFrame {{
                background: #2a2a2a;
                border: 1px solid {border_color};
                border-radius: 2px;
                padding: 6px 8px;
            }}
        """)

    @staticmethod
    def _format_angle_name(name: str) -> str:
        """Format angle name for display."""
        name_map = {
            'knee_extension_bdc': 'KNEE EXTENSION @ BDC',
            'knee_flexion_bdc': 'KNEE FLEXION @ BDC',
            'hip_angle_bdc': 'HIP ANGLE (minimum)',
            'elbow_flexion': 'ELBOW ANGLE (static)',
            'back_angle': 'BACK ANGLE (static)'
        }
        return name_map.get(name, name.upper().replace('_', ' '))
