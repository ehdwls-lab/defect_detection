from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from src.ui.image_utils import ndarray_to_qimage


class ImagePanel(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.title = QLabel(title.upper())
        self.title.setObjectName("panelTitle")
        self.image = QLabel("NO DATA")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(260, 180)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self); layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(self.title); layout.addWidget(self.image, 1)
        self.setObjectName("imagePanel")

    def set_array(self, array):
        self._pixmap = QPixmap.fromImage(ndarray_to_qimage(array)) if array is not None else None
        self._rescale()

    def _rescale(self):
        if self._pixmap is None:
            self.image.setText("NO DATA"); self.image.setPixmap(QPixmap()); return
        self.image.setText("")
        self.image.setPixmap(self._pixmap.scaled(
            self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event); self._rescale()
