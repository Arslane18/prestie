# /// script
# requires-python = ">=3.12"
# dependencies = ["pywebview[qt]>=6.2"]
# ///
"""Prestie companion window, native on Windows (pywebview + Qt WebEngine).

Runs on the Windows side with Windows' own uv, while the backend keeps running
in WSL (`prestie serve`); WSL forwards localhost, so the window simply loads
http://localhost:<port>. `prestie serve --open` starts it for you, or run:

    uv run \\\\wsl.localhost\\Ubuntu\\...\\prestie\\desktop\\prestie_window.py

The window is frameless, transparent and stays on top of the game (WoW in
"Windowed (Fullscreen)" mode): the page draws its own title bar and calls back
into Python through `window.pywebview.api` for pin / minimize / close.

It uses pywebview's Qt backend, not the default WebView2 one: WebView2 has no
per-pixel transparency (measured in prototypes/transparency-probe), and the
page needs it to fade its background out while the text stays fully opaque,
like WoW's chat frame. Qt is downloaded once by uv (~200 MB).
"""

import argparse
from collections.abc import Sequence
from typing import Any

DEFAULT_URL = "http://localhost:8000"
WINDOW_TITLE = "Prestie"
WINDOW_WIDTH = 420
WINDOW_HEIGHT = 760
MIN_SIZE = (340, 480)
SCREEN_MARGIN = 24
GUI = "qt"
RESIZE_EDGES = frozenset({"left", "right", "top", "bottom"})
OPPOSITE_EDGES = (frozenset({"left", "right"}), frozenset({"top", "bottom"}))


def parse_edges(raw: str) -> frozenset[str]:
    """Edges to resize from, sent by the page as "left,bottom"; validated here."""
    edges = frozenset(part.strip() for part in str(raw).split(",") if part.strip())
    if not edges or not edges <= RESIZE_EDGES or any(pair <= edges for pair in OPPOSITE_EDGES):
        raise ValueError(f"invalid resize edges: {raw!r}")
    return edges


class QtSystemGestures:
    """Moves and resizes the window the way Windows does for a title bar or a
    border (snapping included), through Qt's startSystemMove/startSystemResize.

    A frameless Qt window has no native borders to resize from, and pywebview's
    JS drag regions do not move it reliably. pywebview calls the JS API from
    worker threads, so each call is queued to Qt's GUI thread; it still lands
    while the mouse button is held, which is all the system needs.
    """

    def __init__(self, window: Any) -> None:
        self._window = window
        self._caller: Any = None
        self._qt_edges: dict[str, Any] = {}

    def _run_on_gui_thread(self, action: Any) -> None:
        if self._caller is None:
            self._caller = self._make_caller()
        self._caller.call.emit(action)

    def _make_caller(self) -> Any:
        from qtpy.QtCore import QObject, Qt, Signal, Slot
        from qtpy.QtWidgets import QApplication

        class GuiThreadCaller(QObject):
            call = Signal(object)

            @Slot(object)
            def run(self, action: Any) -> None:
                action()

        self._qt_edges = {
            "left": Qt.Edge.LeftEdge,
            "right": Qt.Edge.RightEdge,
            "top": Qt.Edge.TopEdge,
            "bottom": Qt.Edge.BottomEdge,
        }
        caller = GuiThreadCaller()
        caller.moveToThread(QApplication.instance().thread())
        caller.call.connect(caller.run)
        return caller

    def _handle(self) -> Any:
        return self._window.native.windowHandle()

    def start_move(self) -> None:
        self._run_on_gui_thread(lambda: self._handle().startSystemMove())

    def start_resize(self, edges: frozenset[str]) -> None:
        def resize() -> None:
            flags = [self._qt_edges[edge] for edge in sorted(edges)]
            combined = flags[0]
            for flag in flags[1:]:
                combined |= flag
            self._handle().startSystemResize(combined)

        self._run_on_gui_thread(resize)


class WindowControls:
    """Exposed to the page as `window.pywebview.api`.

    pywebview exposes every public attribute to JavaScript, so the window
    reference stays private.
    """

    def __init__(self) -> None:
        self._window: Any = None
        self._gestures: Any = None

    def _attach(self, window: Any, gestures: Any) -> None:
        self._window = window
        self._gestures = gestures

    def start_move(self) -> None:
        self._gestures.start_move()

    def start_resize(self, edges: str) -> None:
        self._gestures.start_resize(parse_edges(edges))

    def minimize(self) -> None:
        self._window.minimize()

    def close(self) -> None:
        self._window.destroy()

    def set_on_top(self, on_top: bool) -> bool:
        self._window.on_top = bool(on_top)
        return self._window.on_top


def right_edge_position(
    screen_width: int, screen_height: int, width: int, height: int, *, margin: int
) -> tuple[int, int]:
    """Top-left corner that puts the window against the right edge, centered."""
    return max(0, screen_width - width - margin), max(0, (screen_height - height) // 2)


def window_settings(url: str, *, x: int, y: int) -> dict[str, Any]:
    """Keyword arguments for `webview.create_window`."""
    return {
        "title": WINDOW_TITLE,
        "url": url,
        "width": WINDOW_WIDTH,
        "height": WINDOW_HEIGHT,
        "x": x,
        "y": y,
        "min_size": MIN_SIZE,
        "frameless": True,
        "easy_drag": False,  # the page starts moves itself (start_move)
        "on_top": True,
        "transparent": True,  # the page paints its own translucent background
        "text_select": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prestie companion window")
    parser.add_argument("--url", default=DEFAULT_URL, help="Prestie API address")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    import webview  # imported here so the pure helpers are testable without it

    screen = webview.screens[0]
    x, y = right_edge_position(
        screen.width, screen.height, WINDOW_WIDTH, WINDOW_HEIGHT, margin=SCREEN_MARGIN
    )
    controls = WindowControls()
    window = webview.create_window(js_api=controls, **window_settings(args.url, x=x, y=y))
    controls._attach(window, QtSystemGestures(window))
    webview.start(gui=GUI)


if __name__ == "__main__":
    main()
