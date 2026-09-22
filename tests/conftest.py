"""All tests are offline: providers and socket connections need explicit mocks."""
import os
import socket
import threading

# Prevent LiteLLM's import-time refresh of its public price map.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["DO_NOT_TRACK"] = "1"

import litellm
import pytest


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    attempted = []
    pair_context = threading.local()
    original_pair = socket.socketpair
    original_connect = socket.socket.connect

    def offline_socketpair(*args, **kwargs):
        # Windows implements asyncio's self-pipe with a local socket pair.
        # Only that construction may connect; ordinary localhost HTTP is blocked.
        pair_context.active = True
        try:
            return original_pair(*args, **kwargs)
        finally:
            pair_context.active = False

    def guarded_connect(sock, address):
        if getattr(pair_context, "active", False) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        return blocked_network()

    def blocked_network(*args, **kwargs):
        attempted.append("network")
        raise AssertionError("Network calls are forbidden in offline tests")

    def blocked_provider(*args, **kwargs):
        attempted.append("provider")
        raise AssertionError("Mock the model boundary explicitly")

    monkeypatch.setattr(socket, "socketpair", offline_socketpair)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked_network)
    monkeypatch.setattr(socket, "create_connection", blocked_network)
    monkeypatch.setattr(litellm, "completion", blocked_provider)
    monkeypatch.setenv("SEA_MEMORY_EMBEDDER", "hashing")
    yield
    # Model code can catch exceptions: fail even if the call was swallowed.
    assert not attempted, f"Unexpected external calls: {attempted}"
