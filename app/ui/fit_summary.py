"""
User-friendly fit summary widget that provides clear, actionable feedback.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class FitSummaryWidget(QWidget):
    """Main fit summary with overall verdict and prioritized recommendations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recommendations = []
        self._build_ui()
        self.clear()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Overall verdict section
        self._verdict_frame = QFrame()
        self._verdict_frame.setStyleSheet("""
            QFrame {
                background: #242424;
                border: none;
                border-radius: 10px;
            }
        """)
        verdict_layout = QVBoxLayout(self._verdict_frame)
        verdict_layout.setContentsMargins(24, 24, 24, 24)
        verdict_layout.setSpacing(8)

        self._verdict_icon = QLabel("")
        self._verdict_icon.setStyleSheet("font-size: 56px;")
        self._verdict_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verdict_icon.setVisible(False)
        verdict_layout.addWidget(self._verdict_icon)

        self._verdict_title = QLabel("Upload a video to begin")
        self._verdict_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #fff;")
        self._verdict_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verdict_title.setWordWrap(True)
        verdict_layout.addWidget(self._verdict_title)

        self._verdict_subtitle = QLabel("")
        self._verdict_subtitle.setStyleSheet("font-size: 13px; color: #aaa;")
        self._verdict_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verdict_subtitle.setWordWrap(True)
        self._verdict_subtitle.setVisible(False)
        verdict_layout.addWidget(self._verdict_subtitle)

        layout.addWidget(self._verdict_frame)

        # Recommendations section
        self._rec_title = QLabel("What to adjust:")
        self._rec_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #f0f0f0; margin-top: 4px;")
        self._rec_title.setVisible(False)
        layout.addWidget(self._rec_title)

        # Scrollable recommendations container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

        rec_container = QWidget()
        self._recommendations_layout = QVBoxLayout(rec_container)
        self._recommendations_layout.setSpacing(10)
        self._recommendations_layout.setContentsMargins(0, 0, 0, 0)

        scroll.setWidget(rec_container)
        layout.addWidget(scroll, stretch=1)

    def set_assessment(self, assessment: dict, recommendations: list):
        """
        Update the summary with assessment results.

        Args:
            assessment: Full assessment dict from fit_assessment.assess_fit()
            recommendations: List of recommendations from generate_recommendations()
        """
        summary = assessment['summary']
        total = summary['total_angles']
        good = summary['in_range']
        marginal = summary['marginal']
        poor = summary['poor']
        issues = marginal + poor

        # Determine overall verdict
        if issues == 0:
            # Perfect fit
            self._verdict_icon.setText("✓")
            self._verdict_icon.setVisible(True)
            self._verdict_icon.setStyleSheet("font-size: 48px; color: #7bd88f; background: transparent; border: none;")
            self._verdict_title.setText("EXCELLENT FIT!")
            self._verdict_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #7bd88f; background: transparent; border: none;")
            self._verdict_subtitle.setText(f"All {total} measurements are within target ranges")
            self._verdict_subtitle.setVisible(True)
            self._verdict_subtitle.setStyleSheet("font-size: 12px; color: #a9cdb1; background: transparent; border: none;")
            self._verdict_frame.setStyleSheet("""
                QFrame {
                    background: #243127;
                    border: 1px solid #324438;
                    border-radius: 12px;
                }
            """)
        elif poor == 0 and issues <= 2:
            # Minor issues only
            self._verdict_icon.setText("⚠")
            self._verdict_icon.setVisible(True)
            self._verdict_icon.setStyleSheet("font-size: 48px; color: #e2b45e; background: transparent; border: none;")
            self._verdict_title.setText("GOOD FIT")
            self._verdict_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #e2b45e; background: transparent; border: none;")

            if issues == 1:
                self._verdict_subtitle.setText(f"{good} out of {total} measurements good • 1 minor adjustment")
            else:
                self._verdict_subtitle.setText(f"{good} out of {total} measurements good • {issues} minor adjustments")

            self._verdict_subtitle.setVisible(True)
            self._verdict_subtitle.setStyleSheet("font-size: 12px; color: #c9b28a; background: transparent; border: none;")
            self._verdict_frame.setStyleSheet("""
                QFrame {
                    background: #312b22;
                    border: 1px solid #4a4030;
                    border-radius: 12px;
                }
            """)
        else:
            # At least one significant issue
            self._verdict_icon.setText("✗")
            self._verdict_icon.setVisible(True)
            self._verdict_icon.setStyleSheet("font-size: 48px; color: #df7a7a; background: transparent; border: none;")
            self._verdict_title.setText("NEEDS ADJUSTMENT")
            self._verdict_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #df7a7a; background: transparent; border: none;")

            summary_parts = []
            if poor == 1:
                summary_parts.append("1 significant adjustment")
            elif poor > 1:
                summary_parts.append(f"{poor} significant adjustments")

            if marginal == 1:
                summary_parts.append("1 minor adjustment")
            elif marginal > 1:
                summary_parts.append(f"{marginal} minor adjustments")

            if summary_parts:
                self._verdict_subtitle.setText(f"{good} out of {total} measurements good • " + " • ".join(summary_parts))
            else:
                self._verdict_subtitle.setText(f"{issues} measurements out of range")
            self._verdict_subtitle.setVisible(True)
            self._verdict_subtitle.setStyleSheet("font-size: 12px; color: #c89c9c; background: transparent; border: none;")
            self._verdict_frame.setStyleSheet("""
                QFrame {
                    background: #312424;
                    border: 1px solid #4a3434;
                    border-radius: 12px;
                }
            """)

        # Clear existing recommendations
        while self._recommendations_layout.count():
            item = self._recommendations_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add recommendations
        if recommendations and recommendations[0]['category'] != 'none':
            self._rec_title.setVisible(True)

            # Show top 3 recommendations
            for i, rec in enumerate(recommendations[:3], 1):
                rec_widget = RecommendationCard(i, rec, assessment['assessments'])
                self._recommendations_layout.addWidget(rec_widget)

            # Add spacer at bottom
            self._recommendations_layout.addStretch()
        else:
            # No adjustments needed
            self._rec_title.setVisible(False)
            no_adjust = QLabel("No adjustments needed!\n\nYour bike fit is spot-on. Keep riding!")
            no_adjust.setStyleSheet("""
                font-size: 14px;
                color: #7bd88f;
                padding: 16px;
                background: #243127;
                border-radius: 6px;
                border: 1px solid #324438;
            """)
            no_adjust.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._recommendations_layout.addWidget(no_adjust)

    def clear(self):
        """Reset the widget to initial state."""
        self._verdict_icon.setText("")
        self._verdict_icon.setVisible(False)
        self._verdict_title.setText("Fit Summary")
        self._verdict_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #fff;")
        self._verdict_subtitle.setText("Overall fit feedback and recommendations will appear here after processing.")
        self._verdict_subtitle.setStyleSheet("font-size: 13px; color: #888;")
        self._verdict_subtitle.setVisible(True)
        self._verdict_frame.setStyleSheet("""
            QFrame {
                background: #242424;
                border: none;
                border-radius: 10px;
            }
        """)

        # Clear recommendations
        while self._recommendations_layout.count():
            item = self._recommendations_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._rec_title.setVisible(False)


class RecommendationCard(QFrame):
    """Single actionable recommendation card with expandable technical details."""

    def __init__(self, priority: int, recommendation: dict, assessments: dict, parent=None):
        super().__init__(parent)
        self._priority = priority
        self._recommendation = recommendation
        self._assessments = assessments
        self._expanded = False
        self._build_ui()

    def _build_ui(self):
        rec = self._recommendation
        severity = rec['severity']
        self.setObjectName("recommendationCard")

        # Color based on severity
        if hasattr(severity, 'value'):
            severity_str = severity.value
        else:
            severity_str = str(severity)

        if severity_str == 'marginal':
            self._accent_color = "#d7a23a"
            self._bg_color = "#2f2a22"
            self._priority_icon = ""
        else:
            self._accent_color = "#d46a6a"
            self._bg_color = "#302424"
            self._priority_icon = ""

        self.setStyleSheet(f"""
            QFrame#recommendationCard {{
                background: {self._bg_color};
                border: 1px solid #333;
                border-radius: 10px;
            }}
        """)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(18, 14, 18, 14)
        self._main_layout.setSpacing(8)

        # Priority and action
        action_layout = QHBoxLayout()
        action_layout.setSpacing(10)

        priority_label = QLabel(f"{self._priority_icon} #{self._priority}")
        priority_label.setStyleSheet(f"""
            font-size: 13px;
            font-weight: 600;
            color: {self._accent_color};
        """)
        action_layout.addWidget(priority_label)

        # Get smart action text
        action_main = self._get_smart_action_text()

        action_text = QLabel(action_main)
        action_text.setStyleSheet("font-size: 14px; font-weight: 600; color: #f0f0f0;")
        action_text.setWordWrap(True)
        action_layout.addWidget(action_text, stretch=1)

        self._main_layout.addLayout(action_layout)

        # Reason
        reason_label = QLabel(f"Why: {rec['reason']}")
        reason_label.setStyleSheet("font-size: 12px; color: #b7b7b7;")
        reason_label.setWordWrap(True)
        self._main_layout.addWidget(reason_label)

        # Expandable details button
        self._details_btn = QPushButton("Show details ▼")
        self._details_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {self._accent_color};
                border: none;
                text-align: left;
                padding: 2px 0;
                font-size: 11px;
            }}
            QPushButton:hover {{
                color: #fff;
                text-decoration: underline;
            }}
        """)
        self._details_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._details_btn.clicked.connect(self._toggle_details)
        self._main_layout.addWidget(self._details_btn)

        # Technical details (hidden by default)
        self._details_frame = QFrame()
        self._details_frame.setObjectName("recommendationDetails")
        self._details_frame.setStyleSheet("""
            QFrame#recommendationDetails {
                background: #262626;
                border: 1px solid #303030;
                border-radius: 8px;
            }
        """)
        self._details_frame.setVisible(False)

        details_layout = QVBoxLayout(self._details_frame)
        details_layout.setContentsMargins(10, 10, 10, 10)
        details_layout.setSpacing(6)

        details_title = QLabel("Technical Details:")
        details_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #888;")
        details_layout.addWidget(details_title)

        # Extract technical info from the recommendation
        details_text = self._get_technical_details()

        details_label = QLabel(details_text)
        details_label.setStyleSheet("font-size: 11px; color: #a5a5a5; line-height: 1.45; background: transparent; border: none;")
        details_label.setWordWrap(True)
        details_layout.addWidget(details_label)

        self._main_layout.addWidget(self._details_frame)

    def _get_smart_action_text(self) -> str:
        """Generate user-friendly action text with mm estimates where applicable."""
        action = self._recommendation['action']
        category = self._recommendation['category']

        # Extract the base action (before parentheses)
        if '(' in action:
            base_action = action.split('(')[0].strip()
        else:
            base_action = action

        # Convert to smart recommendations with generic directives
        if 'saddle' in base_action.lower():
            if 'raise' in base_action.lower():
                return "Raise Your Saddle"
            elif 'lower' in base_action.lower():
                return "Lower Your Saddle"

        if 'handlebar' in base_action.lower() or 'stem' in base_action.lower():
            if 'raise' in base_action.lower() or 'shorten' in base_action.lower():
                return "Raise Handlebars or Shorten Stem"
            elif 'lower' in base_action.lower() or 'lengthen' in base_action.lower():
                return "Lower Handlebars or Lengthen Stem"

        # Fallback to title case.
        return base_action.title()

    def _get_technical_details(self) -> str:
        """Extract technical details from assessment data."""
        # Parse the action string to find which angle this is about
        action = self._recommendation['action'].lower()

        # Map action text to assessment keys
        if 'knee extension' in action or 'knee too bent' in self._recommendation['reason'].lower():
            key = 'knee_extension_bdc'
        elif 'knee flexion' in action:
            key = 'knee_flexion_bdc'
        elif 'hip' in action:
            key = 'hip_angle_bdc'
        elif 'elbow' in action:
            key = 'elbow_flexion'
        elif 'back' in action:
            key = 'back_angle'
        else:
            return "No technical details available"

        if key not in self._assessments:
            return "No technical details available"

        assessment = self._assessments[key]
        measured = assessment.get('measured_mean')
        target_min = assessment.get('target_min')
        target_max = assessment.get('target_max')
        deviation = assessment.get('deviation')
        count = assessment.get('count', 0)
        std = assessment.get('std')

        details = []
        if measured is not None:
            details.append(f"• Measured: {measured:.1f}°")
            details.append(f"• Target: {target_min}°–{target_max}°")

            if deviation is not None:
                if deviation > 0:
                    details.append(f"• Deviation: +{deviation:.1f}° (too high)")
                elif deviation < 0:
                    details.append(f"• Deviation: {deviation:.1f}° (too low)")
                else:
                    details.append(f"• Deviation: In range ✓")

            if count > 1 and std is not None:
                if std < 3:
                    consistency = "Excellent"
                elif std < 5:
                    consistency = "Good"
                else:
                    consistency = "Variable"
                details.append(f"• Consistency: {consistency} ({count} samples, σ={std:.1f}°)")
            elif count == 1:
                details.append(f"• Confidence: Low (only 1 sample)")

        return "\n".join(details)

    def _toggle_details(self):
        """Toggle visibility of technical details."""
        self._expanded = not self._expanded
        self._details_frame.setVisible(self._expanded)

        if self._expanded:
            self._details_btn.setText("Hide details ▲")
        else:
            self._details_btn.setText("Show details ▼")
