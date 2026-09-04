from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from src.ui.theme import COLORS


class ResultPanel(QFrame):
    TEXT = {
        "NORMAL": ("✓ NORMAL", "INSPECTION OK"),
        "DEFECT": ("⚠ DEFECT", "SURFACE ANOMALY DETECTED"),
        "RECHECK": ("! RECHECK", "INSPECTION REQUIRED"),
    }

    def __init__(self):
        super().__init__(); self.setObjectName("resultPanel")
        self.primary = QLabel(); self.primary.setAlignment(Qt.AlignCenter); self.primary.setObjectName("resultPrimary")
        self.secondary = QLabel(); self.secondary.setAlignment(Qt.AlignCenter); self.secondary.setObjectName("resultSecondary")
        box = QVBoxLayout(self); box.addStretch(); box.addWidget(self.primary); box.addWidget(self.secondary); box.addStretch()

    def set_result(self, state: str, visible: bool = True):
        state = state if state in self.TEXT else "RECHECK"
        primary, secondary = self.TEXT[state]; color = COLORS["normal" if state == "NORMAL" else "defect" if state == "DEFECT" else "warning"]
        self.primary.setText(primary if visible else "INSPECTION IN PROGRESS")
        self.secondary.setText(secondary if visible else "ACTIVE VISION SEQUENCE")
        self.setStyleSheet(f"#resultPanel {{ border:2px solid {color}; background:#101820; }} #resultPrimary {{ color:{color}; }}")
