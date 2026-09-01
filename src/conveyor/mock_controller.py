from __future__ import annotations

import logging


class MockConveyorController:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.connected = False
        self.moving = False

    def _event(self, name: str) -> None:
        self.events.append(name)
        logging.getLogger(__name__).info("[CONVEYOR] %s", name)

    def connect(self) -> None:
        self.connected = True
        self._event("mock connected")

    def close(self) -> None:
        self.connected = False
        self._event("mock closed")

    def wait_for_object(self) -> None:
        """Deprecated future sensor hook retained for compatibility."""
        self._event("mock object detected")

    def move_to_inspection(self) -> None:
        self.moving = True
        self._event("mock moved to inspection")

    def stop(self) -> None:
        self.moving = False
        self._event("mock stopped")

    def move_out(self) -> None:
        self.moving = True
        self._event("mock moved out")

    def wait_until_stopped(self, timeout: float | None = None) -> None:
        self.moving = False
        self._event("mock target reached")
