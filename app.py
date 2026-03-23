"""Streamlit dashboard for self-evolving agents."""

from __future__ import annotations

import os

import streamlit as st

from self_evolving.dashboard.data import (
    load_agent_memory,
    load_benchmark_sessions,
    load_benchmark_variant,
    load_recent_runs,
    load_run_detail,
)


DB_PATH = os.getenv("SEA_DB_PATH", ".data/sea.db")
BENCHMARK_DIR = os.getenv("SEA_BENCHMARK_DIR", "runs/benchmarks")


def render_overview() -> None:
    st.header("Overview")
    runs = load_recent_runs(DB_PATH, limit=100)
    sessions = load_benchmark_sessions(BENCHMARK_DIR)

    col1, col2, col3 = st.columns(3)
    col1.metric("Recent runs", len(runs))
    col2.metric("Benchmark sessions", len(sessions))
    col3.metric("Successful runs", sum(1 for run in runs if run["success"]))

    st.subheader("Recent Runs")
    if not runs:
        st.info("No persisted runs yet.")
    else:
        st.dataframe(runs, use_container_width=True)


def render_runs() -> None:
    st.header("Runs")
    runs = load_recent_runs(DB_PATH, limit=100)
    if not runs:
        st.info("No persisted runs yet.")
        return

    run_options = {f"{run['run_id']} | {run['goal']}": run["run_id"] for run in runs}
    selected_label = st.selectbox("Select a run", list(run_options.keys()))
    run_id = run_options[selected_label]

    detail = load_run_detail(DB_PATH, run_id)
    if detail is None:
        st.warning("Run not found.")
        return

    st.subheader("Run Metadata")
    st.json(
        {
            key: value
            for key, value in detail.items()
            if key != "steps"
        }
    )

    st.subheader("Steps")
    st.dataframe(detail["steps"], use_container_width=True)


def render_memory() -> None:
    st.header("Agent Memory")
    runs = load_recent_runs(DB_PATH, limit=100)
    agent_ids = sorted({run["agent_id"] for run in runs})
    if not agent_ids:
        st.info("No agent memory available yet.")
        return

    agent_id = st.selectbox("Select an agent", agent_ids)
    memories = load_agent_memory(DB_PATH, agent_id, limit=100)
    if not memories:
        st.info("No memory entries found for this agent.")
        return

    st.dataframe(memories, use_container_width=True)


def render_benchmarks() -> None:
    st.header("Benchmarks")
    sessions = load_benchmark_sessions(BENCHMARK_DIR)
    if not sessions:
        st.info("No benchmark artifacts found yet.")
        return

    labels = [
        f"{session['generated_at']} | {session['task_count']} tasks"
        for session in sessions
    ]
    selected_label = st.selectbox("Select a benchmark session", labels)
    session = sessions[labels.index(selected_label)]

    st.subheader("Summary")
    st.json(session)

    variants = list(session["variants"].keys())
    selected_variant = st.selectbox("Select a variant", variants)
    variant = load_benchmark_variant(session["session_dir"], selected_variant)
    if variant is not None:
        st.subheader(f"Variant: {selected_variant}")
        st.json(variant)


def main() -> None:
    st.set_page_config(page_title="Self-Evolving Agents Dashboard", layout="wide")
    st.title("Self-Evolving Agents Dashboard")
    st.caption(f"DB: {DB_PATH} | Benchmarks: {BENCHMARK_DIR}")

    page = st.sidebar.radio(
        "Page",
        ["Overview", "Runs", "Memory", "Benchmarks"],
    )

    if page == "Overview":
        render_overview()
    elif page == "Runs":
        render_runs()
    elif page == "Memory":
        render_memory()
    else:
        render_benchmarks()


if __name__ == "__main__":
    main()
