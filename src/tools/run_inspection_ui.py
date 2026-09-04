"""Launch the read-only production inspection dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--screenshot-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from src.ui.industrial_dashboard import IndustrialDashboard
    except ImportError as exc:
        raise SystemExit(
            "PySide6 is required for the dashboard. Install the UI dependencies first."
        ) from exc
    app = QApplication.instance() or QApplication([])
    window = IndustrialDashboard(args.run, watch=args.mode == "live")
    window.show()
    if args.screenshot:
        if args.screenshot_only:
            window.show_final_state()
        def save_screenshot() -> None:
            args.screenshot.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(args.screenshot.expanduser().resolve())):
                raise RuntimeError(f"failed to save screenshot: {args.screenshot}")
            if args.screenshot_only:
                app.quit()
        QTimer.singleShot(1200, save_screenshot)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
