"""Append-only, synchronized safety events and disk failure handling."""

import json
import os
from datetime import datetime, timezone


class VisionSafetyStop(RuntimeError):
    """Stop before the next operation when a safety guard is no longer available."""


class SafetyJournal:
    """Journal operations share the monitor lock with its sampling worker."""

    @staticmethod
    def _timestamp():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _flush(stream):
        stream.flush()
        os.fsync(stream.fileno())

    def _write_event(self, event_name, details):
        event = {**details, "timestamp": self._timestamp(), "run_id": self.run_id, "event": event_name}
        self._events.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self._flush(self._events)

    def event(self, event_name, **details):
        with self._lock:
            if not self._started or self._closed:
                raise VisionSafetyStop("Safety monitor is not running")
            try:
                self._write_event(event_name, details)
            except (OSError, ValueError, TypeError) as error:
                self._failure = self._failure or f"Safety event could not be saved: {error}"
                self._stop.set()
                raise VisionSafetyStop(self._failure) from error

    def _latch(self, reason):
        with self._lock:
            if self._failure is not None:
                return
            self._failure = reason
            self._stop.set()
            if self._events is not None and not self._closed:
                try:
                    self._write_event("safety_stop", {"reason": reason})
                except (OSError, ValueError, TypeError):
                    pass

    def _close_streams(self):
        failure = None
        for stream in (self._gpu, self._events):
            if stream is not None:
                try:
                    stream.close()
                except OSError as error:
                    failure = error
        if failure is not None and self._failure is None:
            self._failure = f"Safety log could not be closed: {failure}"
            raise VisionSafetyStop(self._failure) from failure
