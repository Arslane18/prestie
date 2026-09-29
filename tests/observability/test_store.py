import json
from datetime import UTC, datetime

from prestie.observability.store import JsonlTraceStore


class Trace:
    def __init__(self, started_at, question):
        self.started_at = started_at
        self.question = question

    def to_json(self):
        return {"started_at": self.started_at.isoformat(), "question": self.question}


def test_traces_are_appended_to_one_file_per_day(tmp_path):
    store = JsonlTraceStore(tmp_path / "traces")

    store.write(Trace(datetime(2026, 9, 29, 10, tzinfo=UTC), "a"))
    store.write(Trace(datetime(2026, 9, 29, 23, tzinfo=UTC), "b"))
    store.write(Trace(datetime(2026, 9, 30, 1, tzinfo=UTC), "c"))

    day = (tmp_path / "traces" / "2026-09-29.jsonl").read_text().splitlines()
    assert [json.loads(line)["question"] for line in day] == ["a", "b"]
    assert (tmp_path / "traces" / "2026-09-30.jsonl").exists()


def test_all_traces_are_read_back_oldest_day_first(tmp_path):
    store = JsonlTraceStore(tmp_path)
    store.write(Trace(datetime(2026, 9, 30, tzinfo=UTC), "later"))
    store.write(Trace(datetime(2026, 9, 29, tzinfo=UTC), "earlier"))

    assert [t["question"] for t in store.read_all()] == ["earlier", "later"]


def test_reading_an_empty_store_yields_nothing(tmp_path):
    assert list(JsonlTraceStore(tmp_path / "missing").read_all()) == []


def test_a_damaged_line_is_skipped_and_counted(tmp_path):
    store = JsonlTraceStore(tmp_path)
    store.write(Trace(datetime(2026, 9, 29, tzinfo=UTC), "ok"))
    with (tmp_path / "2026-09-29.jsonl").open("a") as handle:
        handle.write('{"question": "cut in the mid\n')  # crash mid-write

    loaded = store.load()

    assert [t["question"] for t in loaded.traces] == ["ok"]
    assert loaded.damaged_lines == 1
