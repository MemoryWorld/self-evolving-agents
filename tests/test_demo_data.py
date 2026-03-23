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
