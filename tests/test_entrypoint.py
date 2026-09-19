"""Smoke test: the service boots via ``python -m provenance`` and creates
its schema in the configured database location on first start."""

from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
import time

import httpx


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_module_entrypoint_serves_requests_and_creates_schema(tmp_path):
    db_path = tmp_path / "service.db"
    port = _free_port()
    env = {
        **os.environ,
        "PROVENANCE_DATABASE_URL": f"sqlite:///{db_path}",
        "PROVENANCE_HOST": "127.0.0.1",
        "PROVENANCE_PORT": str(port),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "provenance"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 15
        while True:
            try:
                response = httpx.get(f"{base}/v1/contents", timeout=1.0)
                break
            except httpx.TransportError:
                if time.monotonic() > deadline:
                    raise AssertionError("service did not start in time")
                time.sleep(0.1)
        assert response.status_code == 200
        assert response.json() == []

        response = httpx.post(
            f"{base}/v1/actors",
            json={"actor_id": "actor-1", "name": "Alice", "actor_type": "person"},
        )
        assert response.status_code == 201
    finally:
        process.terminate()
        process.wait(timeout=10)

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"actors", "contents", "audit_events"} <= tables
