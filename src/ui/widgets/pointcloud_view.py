"""Optional PyVista point-cloud view with a safe depth fallback."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.ui.widgets.image_panel import ImagePanel


class PointCloudView(QWidget):
    def __init__(self):
        super().__init__(); self.title = QLabel("3D SURFACE / INSPECTION PLANE"); self.title.setObjectName("panelTitle")
        self.info = QLabel("3D DATA NOT AVAILABLE"); self.info.setObjectName("technicalLabel")
        self.fallback = ImagePanel("")
        box = QVBoxLayout(self); box.addWidget(self.title); box.addWidget(self.info); box.addWidget(self.fallback, 1)
        self.interactor = None
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            try:
                from pyvistaqt import QtInteractor
                self.interactor = QtInteractor(self)
                box.replaceWidget(self.fallback, self.interactor); self.fallback.hide()
            except ImportError:
                pass

    def set_data(self, ply: Path | None, depth_image, info: str):
        self.info.setText(info)
        if self.interactor is not None and ply is not None and ply.is_file():
            try:
                import pyvista as pv
                self.interactor.clear(); cloud = pv.read(str(ply))
                self.interactor.add_mesh(cloud, color="#9eabb3", point_size=2,
                                         render_points_as_spheres=True)
                self.interactor.set_background("#090d12"); self.interactor.camera_position = "iso"
                self.interactor.reset_camera(); return
            except Exception:
                pass
        self.fallback.show(); self.fallback.set_array(depth_image)
        if ply is None and depth_image is None: self.info.setText("3D DATA NOT AVAILABLE")
