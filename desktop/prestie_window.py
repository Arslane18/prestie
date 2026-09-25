# /// script
# requires-python = ">=3.12"
# dependencies = ["pywebview>=5"]
# ///
"""Prestie companion window, native on Windows (pywebview + WebView2).

Runs on the Windows side with Windows' own uv, while the backend keeps running
in WSL (`prestie serve`); WSL forwards localhost, so the window simply loads
http://localhost:<port>. `prestie serve --open` starts it for you, or run:

    uv run \\\\wsl.localhost\\Ubuntu\\...\\prestie\\desktop\\prestie_window.py

The window is frameless and stays on top of the game (WoW in "Windowed
(Fullscreen)" mode): the page draws its own title bar and calls back into
Python through `window.pywebview.api` for pin / minimize / close.
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
BACKGROUND = "#0d0f14"  # same as the page, so no white flash while loading


class WindowControls:
    """Exposed to the page as `window.pywebview.api`.

    pywebview exposes every public attribute to JavaScript, so the window
    reference stays private.
    """

    def __init__(self) -> None:
        self._window: Any = None

    def _attach(self, window: Any) -> None:
        self._window = window

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
    window = webview.create_window(
        WINDOW_TITLE,
        args.url,
        js_api=controls,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        x=x,
        y=y,
        min_size=MIN_SIZE,
        frameless=True,
        easy_drag=False,  # drag only from the page's title bar
        on_top=True,
        background_color=BACKGROUND,
        text_select=True,
    )
    controls._attach(window)
    webview.start()


if __name__ == "__main__":
    main()
