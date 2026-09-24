import os

import pytest

from prestie.character.state import CharacterStateError
from prestie.character.watcher import SavedVariablesWatcher

SAVED = """\
PrestieDB = {{\r
["schema"] = 1,\r
["snapshot"] = {{\r
["capturedAt"] = 1790278000,\r
["character"] = "Testeur",\r
["realm"] = "Hyjal",\r
["level"] = {level},\r
["class"] = {{\r
["name"] = "Chevalier de la mort",\r
["file"] = "DEATHKNIGHT",\r
}},\r
["quests"] = {{\r
}},\r
}},\r
}}\r
"""


def write(path, level, mtime):
    path.write_text(SAVED.format(level=level), encoding="utf-8", newline="")
    os.utime(path, (mtime, mtime))


@pytest.fixture
def saved_file(tmp_path):
    path = tmp_path / "Prestie.lua"
    write(path, level=80, mtime=1_000)
    return path


def test_latest_reads_the_state(saved_file):
    assert SavedVariablesWatcher(saved_file).latest().level == 80


def test_latest_rereads_only_when_the_file_changed(saved_file, monkeypatch):
    watcher = SavedVariablesWatcher(saved_file)
    watcher.latest()
    reads = []
    original = type(saved_file).read_text
    monkeypatch.setattr(
        type(saved_file),
        "read_text",
        lambda self, *a, **k: reads.append(self) or original(self, *a, **k),
    )

    watcher.latest()
    write(saved_file, level=81, mtime=2_000)
    state = watcher.latest()

    assert len(reads) == 1
    assert state.level == 81


def test_poll_returns_a_state_only_on_change(saved_file):
    watcher = SavedVariablesWatcher(saved_file)

    first = watcher.poll()
    unchanged = watcher.poll()
    write(saved_file, level=82, mtime=3_000)
    changed = watcher.poll()

    assert first.level == 80
    assert unchanged is None
    assert changed.level == 82


def test_poll_returns_none_while_the_file_does_not_exist(tmp_path):
    assert SavedVariablesWatcher(tmp_path / "Prestie.lua").poll() is None


def test_latest_explains_a_missing_file(tmp_path):
    with pytest.raises(CharacterStateError, match="/reload"):
        SavedVariablesWatcher(tmp_path / "Prestie.lua").latest()


def test_a_half_written_file_is_retried_on_the_next_read(saved_file):
    watcher = SavedVariablesWatcher(saved_file)
    saved_file.write_text('PrestieDB = {\n["schema"] = 1,\n', encoding="utf-8")
    os.utime(saved_file, (2_000, 2_000))

    with pytest.raises(CharacterStateError, match="line"):
        watcher.latest()
    write(saved_file, level=83, mtime=2_000)

    assert watcher.latest().level == 83


def test_watch_yields_each_change(saved_file):
    watcher = SavedVariablesWatcher(saved_file)
    levels = iter([81, 82])

    def sleep(_seconds):
        level = next(levels, None)
        if level is None:
            raise KeyboardInterrupt
        write(saved_file, level=level, mtime=1_000 + level)

    seen = []
    with pytest.raises(KeyboardInterrupt):
        for state in watcher.watch(interval=0.1, sleep=sleep):
            seen.append(state.level)

    assert seen == [80, 81, 82]


def test_watch_reports_a_broken_file_once_and_keeps_watching(saved_file):
    watcher = SavedVariablesWatcher(saved_file)
    errors = []
    steps = iter(["break", "wait", "fix"])

    def sleep(_seconds):
        step = next(steps, None)
        if step == "break":
            saved_file.write_text("PrestieDB = {", encoding="utf-8")
            os.utime(saved_file, (5_000, 5_000))
        elif step == "fix":
            write(saved_file, level=84, mtime=6_000)
        elif step is None:
            raise KeyboardInterrupt

    seen = []
    with pytest.raises(KeyboardInterrupt):
        for state in watcher.watch(interval=0.1, sleep=sleep, on_error=errors.append):
            seen.append(state.level)

    assert seen == [80, 84]
    assert len(errors) == 1
