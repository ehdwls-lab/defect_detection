"""Industrial HMI for read-only production inspection replay."""
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import QElapsedTimer, QTimer, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QProgressBar, QPushButton, QTabWidget, QVBoxLayout, QWidget)
from src.ui.image_utils import (anomaly_localization_overlay, depth_preview, load_bgr,
                                roi_contour_overlay)
from src.ui.inspection_presenter import display_judgement, load_inspection_view
from src.ui.run_replay import REPLAY_STAGES, ReplayCursor
from src.ui.theme import COLORS, STYLESHEET
from src.ui.widgets.image_panel import ImagePanel
from src.ui.widgets.pointcloud_view import PointCloudView
from src.ui.widgets.result_panel import ResultPanel


class Metric(QFrame):
    def __init__(self, title):
        super().__init__(); self.setObjectName("metric"); self.setMaximumHeight(72)
        name = QLabel(title); name.setObjectName("panelTitle")
        self.value = QLabel("--"); self.value.setObjectName("metricValue")
        box = QVBoxLayout(self); box.setContentsMargins(10, 5, 10, 5); box.addWidget(name); box.addWidget(self.value)


class IndustrialDashboard(QMainWindow):
    def __init__(self, run_dir: str | Path, *, watch=False):
        super().__init__(); self.run_dir = Path(run_dir); self.watch = watch
        self.setWindowTitle("Active Vision Surface Inspection"); self.resize(1920, 1080)
        self.setMinimumSize(1600, 900); self.setStyleSheet(STYLESHEET)
        root = QWidget(); self.setCentralWidget(root); outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 8, 12, 8); outer.setSpacing(7)
        self._header(outer); self._stages(outer); self._metrics(outer)
        self.tabs = QTabWidget(); outer.addWidget(self.tabs, 1)
        self._main_tab(); self._technical_tab(); self._system_bar(outer)
        self.replay = ReplayCursor(); self.clock = QElapsedTimer(); self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self._replay_tick); self.elapsed_before_play = 0.0
        self.reveal = 6; self.reload()
        if watch:
            self.controls.hide(); self.live_timer = QTimer(self); self.live_timer.timeout.connect(self.reload)
            self.live_timer.start(1000); self._set_stage(6)
        else: self._restart()

    def _header(self, outer):
        frame = QFrame(); frame.setObjectName("header"); row = QHBoxLayout(frame)
        brand = QLabel("ACTIVE VISION  SURFACE INSPECTION"); brand.setObjectName("brand")
        self.header_info = QLabel(); self.header_info.setObjectName("runLabel")
        row.addWidget(brand); row.addStretch(); row.addWidget(self.header_info); outer.addWidget(frame)

    def _stages(self, outer):
        row = QHBoxLayout(); self.stage_labels = []
        for name in REPLAY_STAGES:
            label = QLabel(name); label.setAlignment(Qt.AlignCenter); label.setObjectName("stagePending")
            row.addWidget(label); self.stage_labels.append(label)
        outer.addLayout(row)

    def _metrics(self, outer):
        row = QHBoxLayout(); self.status = Metric("INSPECTION"); self.score = Metric("ANOMALY")
        self.pose = Metric("PLATFORM"); self.transport = Metric("TRANSPORT")
        for widget in (self.status, self.score, self.pose, self.transport): row.addWidget(widget)
        outer.addLayout(row)

    def _main_tab(self):
        page = QWidget(); layout = QHBoxLayout(page); layout.setContentsMargins(6, 6, 6, 6)
        left = QFrame(); left.setObjectName("imagePanel"); left_box = QVBoxLayout(left)
        title = QHBoxLayout(); name = QLabel("LIVE INSPECTION"); name.setObjectName("panelTitle")
        self.roi_badge = QLabel("ROI ACTIVE"); self.roi_badge.setStyleSheet(f"color:{COLORS['normal']};font-weight:700")
        self.plane_badge = QLabel("PLANE -- / --")
        title.addWidget(name); title.addWidget(self.roi_badge); title.addStretch(); title.addWidget(self.plane_badge)
        self.live_view = ImagePanel(""); left_box.addLayout(title); left_box.addWidget(self.live_view, 1)
        right = QWidget(); right_box = QVBoxLayout(right); right_box.setContentsMargins(0, 0, 0, 0)
        self.pointcloud = PointCloudView(); self.localization = ImagePanel("ANOMALY LOCALIZATION")
        self.anomaly_level = QLabel("ANOMALY LEVEL   -- x TH"); self.anomaly_level.setObjectName("metricValue")
        self.anomaly_values = QLabel("MAX SCORE --     THRESHOLD --"); self.anomaly_values.setObjectName("technicalLabel")
        anomaly = QFrame(); anomaly.setObjectName("imagePanel"); anomaly_box = QVBoxLayout(anomaly)
        anomaly_box.addWidget(self.localization, 1); anomaly_box.addWidget(self.anomaly_level); anomaly_box.addWidget(self.anomaly_values)
        self.result_panel = ResultPanel()
        right_box.addWidget(self.pointcloud, 35); right_box.addWidget(anomaly, 40); right_box.addWidget(self.result_panel, 25)
        layout.addWidget(left, 59); layout.addWidget(right, 41); self.tabs.addTab(page, "INSPECTION")

    def _technical_tab(self):
        page = QWidget(); layout = QVBoxLayout(page); toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("POSE")); self.selector = QComboBox(); self.selector.currentIndexChanged.connect(self._show_pose)
        self.debug = QCheckBox("SHOW FULL PATCH GRID"); self.debug.toggled.connect(self._show_pose)
        toolbar.addWidget(self.selector); toolbar.addWidget(self.debug); toolbar.addStretch(); layout.addLayout(toolbar)
        grid = QGridLayout(); names = ("RGB RAW", "DEPTH PROJECTION", "INSPECTION MASK", "SURFACE PATCH OVERLAY",
            "ANOMALY RAW HEATMAP", "ANOMALY OVERLAY", "BOARD PLANE OVERLAY", "STRUCTURED LIGHT / PLY")
        self.detail_panels = [ImagePanel(name) for name in names]
        for i, panel in enumerate(self.detail_panels): grid.addWidget(panel, i // 4, i % 4)
        layout.addLayout(grid, 1); self.technical_text = QLabel(); self.technical_text.setObjectName("technicalLabel")
        self.technical_text.setTextInteractionFlags(Qt.TextSelectableByMouse); layout.addWidget(self.technical_text)
        self.controls = QFrame(); controls = QHBoxLayout(self.controls); controls.addWidget(QLabel("REPLAY CONTROLS"))
        self.play_button = QPushButton("PLAY"); self.pause_button = QPushButton("PAUSE"); self.restart_button = QPushButton("RESTART")
        self.speed = QComboBox(); self.speed.addItems(("1x", "2x")); self.play_button.clicked.connect(self._play)
        self.pause_button.clicked.connect(self._pause); self.restart_button.clicked.connect(self._restart)
        for widget in (self.play_button, self.pause_button, self.restart_button, self.speed): controls.addWidget(widget)
        controls.addStretch(); layout.addWidget(self.controls); self.tabs.addTab(page, "TECHNICAL DETAIL")

    def _system_bar(self, outer):
        frame = QFrame(); frame.setObjectName("header"); row = QHBoxLayout(frame)
        for name in ("CAMERA", "DEPTH", "STRUCTURED LIGHT", "LIGHTING", "PLATFORM", "CONVEYOR"):
            label = QLabel(f"{name}  ●"); label.setStyleSheet(f"color:{COLORS['normal']}"); row.addWidget(label)
        row.addStretch(); self.progress = QProgressBar(); self.progress.setFixedWidth(220); row.addWidget(self.progress)
        outer.addWidget(frame)

    def _speed(self): return 2.0 if self.speed.currentText() == "2x" else 1.0
    def _play(self): self.clock.restart(); self.replay_timer.start(50)
    def _pause(self):
        if self.replay_timer.isActive(): self.elapsed_before_play += self.clock.elapsed() / 1000 * self._speed()
        self.replay_timer.stop()
    def _restart(self):
        self.replay.restart(); self.reveal = -1; self.elapsed_before_play = 0.0
        self.clock.start(); self.replay_timer.start(50); self._set_stage(0); self._show_pose(self.selector.currentIndex())
    def _replay_tick(self):
        for _ in self.replay.advance(self.elapsed_before_play + self.clock.elapsed() / 1000 * self._speed()):
            self.reveal = self.replay.index; self._set_stage(self.reveal); self._show_pose(self.selector.currentIndex())
        if self.replay.index == len(self.replay.events) - 1: self.replay_timer.stop()
    def _set_stage(self, active):
        for i, label in enumerate(self.stage_labels):
            label.setObjectName("stageDone" if i < active else "stageActive" if i == active else "stagePending")
            label.setText(("✓ " if i < active else "") + REPLAY_STAGES[i]); label.style().unpolish(label); label.style().polish(label)
        self.progress.setMaximum(6); self.progress.setValue(max(0, active))

    def show_final_state(self):
        """Render the completed replay frame for deterministic screenshots."""
        self.replay_timer.stop(); self.reveal = 6; self._set_stage(6)
        self._show_pose(self.selector.currentIndex())

    def reload(self):
        try: self.view = load_inspection_view(self.run_dir)
        except (FileNotFoundError, ValueError) as exc:
            self.status.value.setText("N/A"); self.technical_text.setText(str(exc)); return
        digits = "".join(c for c in self.view.run_name if c.isdigit()); cycle = f"#{int(digits[-3:]):03d}" if digits else "#001"
        self.header_info.setText(f"SYSTEM READY  ●     PART: {self.view.product}     CYCLE: {cycle}     MODE: AUTO")
        self.status.value.setText(self.view.status); self.transport.value.setText("OUT COMPLETE" if self.view.transport_complete else "--")
        old = self.selector.currentIndex(); labels = [f"PLANE {p.index + 1} / {len(self.view.poses)}  ·  {p.name}" for p in self.view.poses]
        self.selector.blockSignals(True); self.selector.clear(); self.selector.addItems(labels)
        self.selector.setCurrentIndex(max(0, min(old, len(labels) - 1))); self.selector.blockSignals(False); self._show_pose(self.selector.currentIndex())

    def _show_pose(self, index=0):
        if not hasattr(self, "view") or not (0 <= index < len(self.view.poses)): return
        pose = self.view.poses[index]; total = len(self.view.poses); state = display_judgement(self.view.judgement)
        relative = None if pose.score is None or not pose.threshold else pose.score / pose.threshold
        self.score.value.setText("--" if relative is None else f"{relative:.2f}x TH")
        self.pose.value.setText(f"R {pose.roll or 0:+.1f}°   P {pose.pitch or 0:+.1f}°   Z {pose.z or 0:.1f}cm")
        self.plane_badge.setText(f"PLANE {index + 1} / {total}")
        rgb = load_bgr(pose.rgb); live = roi_contour_overlay(rgb, pose.mask) if self.reveal >= 3 else rgb if self.reveal >= 2 else None
        if self.debug.isChecked() and pose.patch_overlay: live = load_bgr(pose.patch_overlay)
        if self.reveal >= 4 and pose.overlay is not None:
            live = roi_contour_overlay(load_bgr(pose.overlay), pose.mask)
        self.live_view.set_array(live); raw_heat = load_bgr(pose.heatmap)
        localization = anomaly_localization_overlay(rgb, raw_heat, pose.score, pose.threshold)
        self.localization.set_array(localization if self.reveal >= 4 else None)
        self.anomaly_level.setText("ANOMALY LEVEL   --" if relative is None else f"ANOMALY LEVEL   {relative:.2f}x TH")
        self.anomaly_values.setText(f"MAX SCORE  {pose.score:.6f}     THRESHOLD  {pose.threshold:.6f}" if pose.score is not None and pose.threshold is not None else "MAX SCORE --     THRESHOLD --")
        self.result_panel.set_result(state, visible=self.reveal >= 5); depth = depth_preview(pose.depth)
        info = f"PLANE {index + 1} / {total}    ROLL {pose.roll or 0:+.1f}°    PITCH {pose.pitch or 0:+.1f}°    Z {pose.z or 0:.1f} cm"
        self.pointcloud.set_data(self.view.ply if self.reveal >= 1 else None, depth if self.reveal >= 1 else None, info)
        details = (rgb, depth, load_bgr(pose.mask), load_bgr(pose.patch_overlay), raw_heat,
                   load_bgr(pose.overlay), load_bgr(pose.board_overlay), depth)
        for panel, image in zip(self.detail_panels, details): panel.set_array(image)
        self.technical_text.setText(f"RUN: {self.view.run_dir}\nCYCLE RESULT: {self.view.run_dir / 'cycle_result.json'}\nPLY: {self.view.ply or 'N/A'}\nSCORE: {pose.score if pose.score is not None else '--'}   THRESHOLD: {pose.threshold if pose.threshold is not None else '--'}   STATUS: {pose.status}")
