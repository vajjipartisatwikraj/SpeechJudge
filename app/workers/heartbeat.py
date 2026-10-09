"""Worker heartbeats so a model-less API gateway can report readiness (Requirement 2).

Each worker periodically writes its per-service readiness to a short-TTL Redis key.
Keys expire on their own, so a dead worker stops counting as ready.
"""
from __future__ import annotations

import json
import os
import socket
import threading
from typing import Callable

PREFIX = "speechjudge:worker:"
INTERVAL_S = 10
TTL_S = 30


def start_heartbeat(client, readiness: Callable[[], dict[str, bool]]) -> threading.Thread:
    key = f"{PREFIX}{socket.gethostname()}:{os.getpid()}"
    stop = threading.Event()

    def _beat() -> None:
        while not stop.is_set():
            try:
                client.set(key, json.dumps(readiness()), ex=TTL_S)
            except Exception:  # noqa: BLE001 - never let a Redis blip kill the worker
                pass
            stop.wait(INTERVAL_S)

    t = threading.Thread(target=_beat, name="sj-heartbeat", daemon=True)
    t.start()
    return t


def aggregate_worker_readiness(client) -> dict[str, bool]:
    """Readiness of the best live worker: a single worker with every service loaded if one
    exists, otherwise all-False (so a partially loaded fleet is never reported ready)."""
    names = ("whisper", "gopt", "qwen", "acoustic")
    for key in client.scan_iter(match=f"{PREFIX}*"):
        raw = client.get(key)
        if raw is None:
            continue
        try:
            report = json.loads(raw)
        except ValueError:
            continue
        if all(bool(report.get(n)) for n in names):
            return {n: True for n in names}
    return {n: False for n in names}
