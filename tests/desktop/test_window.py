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


def test_controls_drive_the_window():
    launcher = load_launcher()
    window = FakeWindow()
    controls = launcher.WindowControls()
    controls._attach(window)

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
