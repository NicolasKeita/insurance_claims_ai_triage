"""Flush progress boundaries and atomically replace the latest saved state."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ProgressJournal:
    """A separate run directory preserves interrupted attempts across retries."""

    def __init__(self, output_dir, scenario_id):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
        self.directory = Path(output_dir) / "progress" / scenario_id / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / "events.jsonl").open("x", encoding="utf-8")
        self.scenario_id = scenario_id

    def record(self, event):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "scenario_id": self.scenario_id,
            **event,
        }
        self.stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        atomic_json(self.directory / "last_state.json", event)

    def checkpoint(self, observation):
        atomic_json(self.directory / "checkpoint.json", observation.model_dump(mode="json"))

    def close(self):
        self.stream.close()
