"""Active stage, periodic elapsed-time updates and durable interruption evidence."""

from threading import Event, Lock, Thread
from time import perf_counter

from evaluation.real.progress.console import print_heartbeat, print_stage, print_started
from evaluation.real.progress.estimate import HistoricalEstimate
from evaluation.real.progress.gpu import read_gpu
from evaluation.real.progress.journal import ProgressJournal


class ScenarioProgress:
    """One heartbeat thread observes the active operation; it never executes test stages."""

    def __init__(self, output_dir, scenario_id, reference=None, *, interval=5, gpu_reader=None):
        self.journal = ProgressJournal(output_dir, scenario_id)
        self.reference = reference or {}
        self.estimate = HistoricalEstimate(self.reference)
        self.interval = interval
        self.gpu_reader = gpu_reader or read_gpu
        self.started = perf_counter()
        self.current = None
        self.lock, self.stop = Lock(), Event()
        self.thread = Thread(target=self._heartbeat, daemon=True, name="evaluation-progress")
        self.closed = False

    def __enter__(self):
        estimate = self._estimate(perf_counter())
        self.journal.record({"event": "scenario_started", "status": "RUNNING", **estimate})
        print_started(self.journal.directory, self.reference, estimate, self.interval)
        self.thread.start()
        return self

    def __call__(self, component, status, observation, **details):
        with self.lock:
            now = perf_counter()
            if status == "RUNNING":
                self.current = (component, now)
            else:
                self.current = None
                self.estimate.completed.add(component)
            estimate = self._estimate(now)
            event = {
                "event": "stage_started" if status == "RUNNING" else "stage_finished",
                "component": component,
                "status": status,
                "elapsed_seconds": now - self.started,
                **details,
                **estimate,
            }
            self.journal.record(event)
            self.journal.checkpoint(observation)
            print_stage(event, observation)

    def _estimate(self, now):
        component, started = self.current if self.current is not None else (None, now)
        return self.estimate.snapshot(now - self.started, component, now - started)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                active = self.current
            if active is None:
                continue
            sample = self.gpu_reader()
            with self.lock:
                if self.closed or self.stop.is_set() or active != self.current:
                    continue
                now = perf_counter()
                estimate = self._estimate(now)
                self.journal.record(
                    {
                        "event": "heartbeat",
                        "component": active[0],
                        "status": "RUNNING",
                        "stage_elapsed_seconds": now - active[1],
                        "elapsed_seconds": now - self.started,
                        "gpu": sample,
                        **estimate,
                    }
                )
                print_heartbeat(active[0], now - active[1], now - self.started, sample, estimate)

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
                    "estimated_remaining_seconds": 0,
                    "eta_reason": None,
                    "reference_total_seconds": self.reference.get("total"),
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
