"""Tests for offline demo data generation."""

from self_evolving.dashboard.demo_data import generate_demo_data
from self_evolving.persistence.sqlite_store import SQLiteStore


def test_generate_demo_data(tmp_path):
    db_path = str(tmp_path / "sea.db")
    benchmark_dir = str(tmp_path / "benchmarks")

    result = generate_demo_data(db_path=db_path, benchmark_dir=benchmark_dir)

    assert len(result["generated_runs"]) == 4

    store = SQLiteStore(db_path)
    runs = store.list_runs(limit=10)
    assert len(runs) == 4

    session_dirs = list((tmp_path / "benchmarks").iterdir())
    assert len(session_dirs) == 1
    assert (session_dirs[0] / "summary.json").exists()
    assert result["data_source"] == "synthetic_deterministic_fixture"
    assert result["benchmark_summary"]["data_source"] == "synthetic_deterministic_fixture"
    assert result["benchmark_summary"]["evaluation_protocol"]["prompt_optimization"] == "heldout"
    memories = store.list_memory("demo-agent")
    assert len(memories) == 4
    assert all("Synthetic lesson" in item["content"] for item in memories)


def test_demo_cannot_mutate_live_provider_methods(tmp_path, monkeypatch):
    from self_evolving.core.agent import BaseAgent
    from self_evolving.evolution.prompt.opro import OPROOptimizer
    original_call = BaseAgent._call_llm
    original_optimize = OPROOptimizer.optimize
    # A user's choice of external embedder must not make the offline demo download a model.
    monkeypatch.setenv("SEA_MEMORY_EMBEDDER", "sentence_transformers")
    generate_demo_data(str(tmp_path / "demo.db"), str(tmp_path / "runs"))
    assert BaseAgent._call_llm is original_call
    assert OPROOptimizer.optimize is original_optimize


def test_demo_cli_is_offline_even_during_package_import(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    probe = '''
import contextlib, io, runpy, sys
def guard(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("Offline CLI attempted network access")
sys.addaudithook(guard)
sys.argv = ["examples/07_generate_demo_data.py", "--db-path", sys.argv[1],
            "--benchmark-dir", sys.argv[2]]
output = io.StringIO()
with contextlib.redirect_stdout(output):
    runpy.run_path("examples/07_generate_demo_data.py", run_name="__main__")
assert "synthetic_deterministic_fixture" in output.getvalue()
print("offline CLI completed with network audit guard")
'''
    result = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path / "cli.db"), str(tmp_path / "benchmarks")],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "offline CLI completed" in result.stdout
    assert (tmp_path / "cli.db").exists()
