"""Supervise only launcher-owned process groups, including startup failures."""

import os
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .configuration import configuration


def wait_ready(url, processes, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A server exited during startup. Check its terminal output above.")
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except HTTPError as error:
            raise RuntimeError(
                f"Server returned HTTP {error.code}: {url}. Check the configured demo snapshots."
            ) from error
        except (URLError, OSError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Server not ready after {timeout}s: {url}. Check the terminal output above.")


def stop(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        _force_stop(process)


def _force_stop(process):
    """Windows venv launchers may own a real Python child; stop the owned tree."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True, timeout=10
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Could not stop child {process.pid}: {error}", file=sys.stderr)


def run(options):
    env, jobs = configuration(options)
    processes = []
    flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    try:
        for command, cwd, url in jobs:
            print(f"Starting {url}", flush=True)
            processes.append(subprocess.Popen(command, cwd=cwd, env=env, **flags))
            wait_ready(url, processes)
        print(f"\nClaim Review: {jobs[-1][2]}\nKeep this terminal open. Ctrl+C stops both servers.", flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(0.3)
        raise RuntimeError("A server stopped. Both launcher-owned servers are being shut down.")
    except KeyboardInterrupt:
        print("\nStopping Claim Review...", flush=True)
        return 0
    finally:
        for process in reversed(processes):
            stop(process)
