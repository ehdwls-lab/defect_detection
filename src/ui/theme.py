"""Central industrial HMI palette and stylesheet."""

COLORS = {
    "background": "#090d12", "panel": "#101820", "panel_alt": "#131f29",
    "border": "#2a3a46", "text": "#d8e2e8", "muted": "#788b98",
    "normal": "#30d39b", "warning": "#f0ad36", "defect": "#ff4f5e",
    "info": "#27b6e6", "pending": "#34434d",
}

STYLESHEET = f"""
QMainWindow, QWidget {{ background:{COLORS['background']}; color:{COLORS['text']}; font-family:sans-serif; }}
#header {{ background:{COLORS['panel']}; border-bottom:2px solid {COLORS['info']}; }}
#brand {{ font-size:22px; font-weight:700; color:#f4f8fb; }}
#runLabel, #panelTitle {{ color:{COLORS['muted']}; font-size:11px; font-weight:700; }}
#resultBadge {{ font-size:38px; font-weight:800; padding:10px 24px; border:2px solid {COLORS['border']}; }}
#resultPrimary {{ font-size:42px; font-weight:900; }} #resultSecondary {{ font-size:16px; font-weight:700; }}
#technicalLabel {{ color:{COLORS['muted']}; font-family:monospace; font-size:12px; }}
#metric, #imagePanel {{ background:{COLORS['panel']}; border:1px solid {COLORS['border']}; }}
#metric {{ padding:8px; }} #metricValue {{ font-size:20px; font-weight:700; color:#f0f5f8; }}
#stageDone {{ color:{COLORS['normal']}; border-bottom:2px solid {COLORS['normal']}; padding:8px; }}
#stageActive {{ color:#ffffff; background:{COLORS['info']}; padding:11px; font-size:14px; font-weight:700; }}
#stagePending {{ color:{COLORS['muted']}; padding:11px; font-size:14px; }}
QComboBox, QPushButton {{ background:{COLORS['panel_alt']}; padding:7px; border:1px solid {COLORS['border']}; }}
QPushButton:hover {{ border-color:{COLORS['info']}; }}
QTabWidget::pane {{ border:1px solid {COLORS['border']}; }}
QTabBar::tab {{ background:{COLORS['panel']}; padding:9px 18px; }}
QTabBar::tab:selected {{ color:{COLORS['info']}; border-bottom:2px solid {COLORS['info']}; }}
QProgressBar {{ background:{COLORS['panel_alt']}; border:0; height:7px; }}
QProgressBar::chunk {{ background:{COLORS['info']}; }}
"""
