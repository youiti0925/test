"""Tests for the double-click launcher's pure helpers.

We don't actually start Flask here — that's covered by tests/test_web.py.
This file just makes sure the dep-check and port-finder behave as
expected so a broken launch.py is caught before anyone double-clicks it.
"""
from __future__ import annotations

import socket

import pytest

import launch


def test_required_modules_constant_is_non_empty():
    assert "flask" in launch.REQUIRED_MODULES
    assert "anthropic" in launch.REQUIRED_MODULES


def test_missing_deps_returns_list():
    # In the test env we install everything, so the result should be empty.
    # We only assert the return type so this test is robust to different envs.
    result = launch.missing_deps()
    assert isinstance(result, list)
    for name in result:
        assert isinstance(name, str)


def test_find_free_port_returns_bindable_port():
    port = launch.find_free_port(range(15000, 15010))
    # Re-bind to confirm the port is actually free at the moment of return.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


def test_find_free_port_skips_busy_ports():
    # Hold one port open, ensure the finder picks a different one.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        held_port = held.getsockname()[1]
        # Build a candidate range that includes the held port and one free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as scout:
            scout.bind(("127.0.0.1", 0))
            free_port = scout.getsockname()[1]
        chosen = launch.find_free_port([held_port, free_port])
        assert chosen != held_port


def test_find_free_port_raises_when_all_busy():
    sockets = []
    try:
        ports = []
        for _ in range(3):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            sockets.append(s)
            ports.append(s.getsockname()[1])
        with pytest.raises(RuntimeError):
            launch.find_free_port(ports)
    finally:
        for s in sockets:
            s.close()
