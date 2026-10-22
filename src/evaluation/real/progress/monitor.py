"""Active stage, periodic elapsed-time updates and durable interruption evidence."""

from threading import Event, Lock, Thread
from time import perf_counter

from evaluation.real.progress.gpu import gpu_text, read_gpu
from evaluation.real.progress.journal import ProgressJournal

LABELS = {
    "document": "Extract documents",
    "rag_generation": "Generate grounded answer with Ollama",
    "agent": "Review claim with the Ollama agent",
    "similar_indexing": "Index similar claims",
}


class ScenarioProgress:
    """One heartbeat thread observes the active operation; it never executes test stages."""

    def __init__(self, output_dir, scenario_id, reference=None, *, interval=5):
        self.journal = ProgressJournal(output_dir, scenario_id)
        self.reference = reference or {}
        self.interval = interval
        self.started = perf_counter()
        self.current = None
        self.lock, self.stop = Lock(), Event()
        self.thread = Thread(target=self._heartbeat, daemon=True, name="evaluation-progress")
        self.closed = False

    def __enter__(self):
        self.journal.record({"event": "scenario_started", "status": "RUNNING"})
        print(f"  Progress files: {self.journal.directory}", flush=True)
        if "total" in self.reference:
            print(f"  Previous recorded run: {self.reference['total']:.1f}s (reference only; ETA unknown).", flush=True)
        else:
            print(f"  ETA unknown; active steps and elapsed time will update every {self.interval:g}s.", flush=True)
        self.thread.start()
        return self

    def __call__(self, component, status, observation, **details):
        with self.lock:
            now = perf_counter()
            if status == "RUNNING":
                self.current = (component, now)
            event = {
                "event": "stage_started" if status == "RUNNING" else "stage_finished",
                "component": component,
                "status": status,
                "elapsed_seconds": now - self.started,
                **details,
            }
            self.journal.record(event)
            self.journal.checkpoint(observation)
            if status != "RUNNING":
                self.current = None
            label = LABELS.get(component, component.replace("_", " ").capitalize())
            duration = f" | {details['duration_seconds']:.1f}s" if "duration_seconds" in details else ""
            print(f"  [{status}] {component}: {label}{duration} | total {now - self.started:.1f}s", flush=True)
            reason = observation.trace[-1].outputs.get("error") if observation.trace else None
            if component == "agent" and status == "FAIL" and reason == details.get("reason") and reason:
                print(f"    Reason: {reason}", flush=True)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                active = self.current
            if active is None:
                continue
            sample = read_gpu()
            with self.lock:
                if self.closed or self.stop.is_set() or active != self.current:
                    continue
                now = perf_counter()
                self.journal.record(
                    {
                        "event": "heartbeat",
                        "component": active[0],
                        "status": "RUNNING",
                        "stage_elapsed_seconds": now - active[1],
                        "elapsed_seconds": now - self.started,
                        "gpu": sample,
                    }
                )
                print(
                    f"  [RUNNING] {active[0]} | step {now - active[1]:.1f}s | total {now - self.started:.1f}s | "
                    + gpu_text(sample),
                    flush=True,
                )

    def complete(self, observation):
        with self.lock:
            self.current = None
            self.journal.checkpoint(observation)
            self.journal.record(
                {
                    "event": "scenario_finished",
                    "status": "FINISHED",
                    "elapsed_seconds": perf_counter() - self.started,
                    "component_statuses": {key: item.status for key, item in observation.components.items()},
                }
            )

    def __exit__(self, error_type, error, traceback):
        self.stop.set()
        self.thread.join(timeout=2)
        with self.lock:
            self.closed = True
            try:
                if error_type is not None:
                    self.journal.record(
                        {
                            "event": "scenario_interrupted",
                            "status": "INTERRUPTED",
                            "component": self.current[0] if self.current else None,
                            "reason": error_type.__name__,
                            "elapsed_seconds": perf_counter() - self.started,
                        }
                    )
            finally:
                self.journal.close()
