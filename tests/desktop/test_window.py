"""Tests for the Windows-side launcher's pure logic (pywebview is not needed)."""

import importlib.util
from pathlib import Path

LAUNCHER = Path(__file__).parents[2] / "desktop" / "prestie_window.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("prestie_window", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_declares_its_own_dependencies():
    # `uv run prestie_window.py` on Windows installs pywebview from this block.
    header = LAUNCHER.read_text(encoding="utf-8")

    assert "# /// script" in header
    assert '"pywebview' in header


def test_launcher_uses_the_qt_backend_for_real_transparency():
    # WebView2 (pywebview's default on Windows) has no per-pixel transparency:
    # measured in prototypes/transparency-probe, only Qt fades the background
    # while keeping the text fully opaque.
    header = LAUNCHER.read_text(encoding="utf-8")
    launcher = load_launcher()

    assert '"pywebview[qt]' in header
    assert launcher.GUI == "qt"


def test_window_is_transparent_frameless_and_on_top():
    launcher = load_launcher()

    settings = launcher.window_settings("http://localhost:8000", x=10, y=20)

    assert settings["url"] == "http://localhost:8000"
    assert settings["transparent"] is True
    assert settings["frameless"] is True
    assert settings["on_top"] is True
    assert settings["easy_drag"] is False
    assert (settings["x"], settings["y"]) == (10, 20)
    assert (settings["width"], settings["height"]) == (420, 760)


def test_window_sits_against_the_right_edge_vertically_centered():
    launcher = load_launcher()

    assert launcher.right_edge_position(1920, 1080, 420, 760, margin=24) == (1476, 160)


def test_window_position_never_goes_off_screen():
    launcher = load_launcher()

    assert launcher.right_edge_position(300, 400, 420, 760, margin=24) == (0, 0)


def test_url_defaults_to_the_local_api():
    launcher = load_launcher()

    assert launcher.parse_args([]).url == "http://localhost:8000"
    assert launcher.parse_args(["--url", "http://localhost:8123"]).url.endswith("8123")


class FakeWindow:
    def __init__(self):
        self.on_top = True
        self.calls: list[str] = []

    def minimize(self):
        self.calls.append("minimize")

    def destroy(self):
        self.calls.append("destroy")


class FakeSystemGestures:
    """Stands in for Qt's startSystemMove / startSystemResize."""

    def __init__(self):
        self.calls: list[tuple] = []

    def start_move(self):
        self.calls.append(("move",))

    def start_resize(self, edges):
        self.calls.append(("resize", edges))


def attached_controls(launcher):
    controls = launcher.WindowControls()
    window, gestures = FakeWindow(), FakeSystemGestures()
    controls._attach(window, gestures)
    return controls, window, gestures


def test_title_bar_drag_hands_the_move_to_the_system():
    launcher = load_launcher()
    controls, _, gestures = attached_controls(launcher)

    controls.start_move()

    assert gestures.calls == [("move",)]


def test_edge_drag_hands_the_resize_to_the_system():
    launcher = load_launcher()
    controls, _, gestures = attached_controls(launcher)

    controls.start_resize("left,bottom")

    assert gestures.calls == [("resize", frozenset({"left", "bottom"}))]


def test_resize_rejects_unknown_or_missing_edges():
    # The edges come from the page: validate them at the boundary.
    launcher = load_launcher()
    controls, _, gestures = attached_controls(launcher)

    for bad in ("", "left,middle", "diagonal", "left,right"):
        try:
            controls.start_resize(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted")
    assert gestures.calls == []


def test_controls_drive_the_window():
    launcher = load_launcher()
    controls, window, _ = attached_controls(launcher)

    assert controls.set_on_top(False) is False
    assert window.on_top is False
    controls.minimize()
    controls.close()
    assert window.calls == ["minimize", "destroy"]


def test_controls_expose_no_public_window_reference():
    # pywebview exposes every public attribute of js_api to the page.
    launcher = load_launcher()
    public = [
        name for name in vars(launcher.WindowControls()) if not name.startswith("_")
    ]

    assert public == []
