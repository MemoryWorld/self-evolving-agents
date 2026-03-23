"""Streamlit dashboard for self-evolving agents."""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import streamlit as st

from self_evolving.dashboard.data import (
    build_benchmark_comparison,
    load_agent_memory,
    load_benchmark_sessions,
    load_job,
    load_jobs,
    load_benchmark_variant,
    load_recent_runs,
    load_run_detail,
    summarize_memory,
    trigger_benchmark,
    trigger_run,
)


DB_PATH = os.getenv("SEA_DB_PATH", ".data/sea.db")
BENCHMARK_DIR = os.getenv("SEA_BENCHMARK_DIR", "runs/benchmarks")
API_BASE = os.getenv("SEA_API_BASE", "http://127.0.0.1:8000")


def _remember_job(job_id: str) -> None:
    active_jobs = st.session_state.setdefault("active_job_ids", [])
    if job_id not in active_jobs:
        active_jobs.insert(0, job_id)
    st.session_state["active_job_ids"] = active_jobs[:10]


def render_job_monitor() -> None:
    st.subheader("Job Monitor")
    active_job_ids = st.session_state.get("active_job_ids", [])

    try:
        recent_jobs = load_jobs(API_BASE, limit=10)
    except Exception as exc:
        st.warning(f"Could not load jobs from API: {exc}")
        return

    jobs_by_id = {job["job_id"]: job for job in recent_jobs}
    ordered_jobs = []
    for job_id in active_job_ids:
        job = jobs_by_id.get(job_id)
        if job is None:
            try:
                job = load_job(API_BASE, job_id)
            except Exception:
                continue
        ordered_jobs.append(job)

    for job in recent_jobs:
        if job["job_id"] not in {item["job_id"] for item in ordered_jobs}:
            ordered_jobs.append(job)

    if not ordered_jobs:
        st.info("No jobs have been submitted yet.")
        return

    active_count = 0
    for job in ordered_jobs[:10]:
        is_active = job["status"] in {"queued", "running"}
        if is_active:
            active_count += 1

        label = f"{job['kind']} | {job['status']} | {job['job_id'][:8]}"
        with st.expander(label, expanded=is_active):
            st.progress(int(job["progress"]))
            st.caption(f"Stage: {job['stage']}")
            st.json(
                {
                    "job_id": job["job_id"],
                    "kind": job["kind"],
                    "status": job["status"],
                    "progress": job["progress"],
                    "stage": job["stage"],
                    "metadata": job.get("metadata", {}),
                    "error": job.get("error"),
                }
            )
            if job.get("result") is not None:
                st.subheader("Result")
                st.json(job["result"])

    if active_count:
        st.caption("Active jobs detected. Refreshing every 2 seconds.")
        time.sleep(2)
        st.rerun()


def render_overview() -> None:
    st.header("Overview")
    runs = load_recent_runs(DB_PATH, limit=100)
    sessions = load_benchmark_sessions(BENCHMARK_DIR)
    success_count = sum(1 for run in runs if run["success"])
    success_rate = (success_count / len(runs)) if runs else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recent runs", len(runs))
    col2.metric("Benchmark sessions", len(sessions))
    col3.metric("Successful runs", success_count)
    col4.metric("Run success rate", f"{success_rate:.1%}")

    st.subheader("Recent Runs")
    if not runs:
        st.info("No persisted runs yet.")
    else:
        runs_df = pd.DataFrame(runs)
        st.dataframe(runs_df, use_container_width=True)

        by_agent = (
            runs_df.groupby("agent_id", as_index=False)["success"]
            .mean()
            .rename(columns={"success": "success_rate"})
        )
        st.subheader("Success Rate By Agent")
        st.bar_chart(by_agent.set_index("agent_id"))


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
    steps_df = pd.DataFrame(detail["steps"])
    st.dataframe(steps_df, use_container_width=True)

    if not steps_df.empty:
        st.subheader("Step Feedback Types")
        feedback_counts = (
            steps_df["feedback_type"]
            .fillna("none")
            .value_counts()
            .rename_axis("feedback_type")
            .to_frame("count")
        )
        st.bar_chart(feedback_counts)


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

    summary = summarize_memory(memories)
    col1, col2, col3 = st.columns(3)
    col1.metric("Entries", summary["total_entries"])
    col2.metric("Avg importance", f"{summary['avg_importance']:.2f}")
    col3.metric("Total accesses", summary["total_accesses"])

    memories_df = pd.DataFrame(memories)
    st.dataframe(memories_df, use_container_width=True)

    if not memories_df.empty:
        top_access = (
            memories_df[["content", "access_count"]]
            .sort_values("access_count", ascending=False)
            .head(10)
            .set_index("content")
        )
        st.subheader("Top Accessed Memory Entries")
        st.bar_chart(top_access)


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

    comparison_rows = build_benchmark_comparison(session)
    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        st.subheader("Variant Comparison")
        st.dataframe(comparison_df, use_container_width=True)

        chart_df = comparison_df[["variant", "success_rate", "mean_steps"]].set_index("variant")
        st.subheader("Success Rate And Mean Steps")
        st.bar_chart(chart_df)

    variants = list(session["variants"].keys())
    selected_variant = st.selectbox("Select a variant", variants)
    variant = load_benchmark_variant(session["session_dir"], selected_variant)
    if variant is not None:
        st.subheader(f"Variant: {selected_variant}")
        st.json(variant)
        episodes = variant.get("episodes", [])
        if episodes:
            st.subheader("Episode Breakdown")
            st.dataframe(pd.DataFrame(episodes), use_container_width=True)


def render_control_plane() -> None:
    st.header("Control Plane")
    st.caption(f"API: {API_BASE}")
    render_job_monitor()

    with st.expander("Create QA Run", expanded=True):
        with st.form("qa_run_form"):
            goal = st.text_input("Goal", value="What is the capital of France?")
            reference_answer = st.text_input("Reference answer", value="Paris")
            agent_id = st.text_input("Agent ID", value="dashboard-agent")
            model = st.text_input("Model override", value="")
            use_memory = st.checkbox("Use memory", value=True)
            submitted = st.form_submit_button("Run Task")

        if submitted:
            payload = {
                "goal": goal,
                "reference_answer": reference_answer,
                "agent_id": agent_id or None,
                "model": model or None,
                "use_memory": use_memory,
            }
            try:
                result = trigger_run(API_BASE, payload)
                _remember_job(result["job_id"])
                st.success(f"Run job created: {result['job_id']}")
                st.json(result)
                st.rerun()
            except Exception as exc:
                st.error(f"Run trigger failed: {exc}")

    with st.expander("Create Benchmark Session", expanded=True):
        default_tasks = [
            {"goal": "What is the capital of France?", "reference_answer": "Paris"},
            {"goal": "What is 12 * 7?", "reference_answer": "84"},
            {"goal": "Who wrote Hamlet?", "reference_answer": "Shakespeare"},
            {"goal": "What planet is known as the Red Planet?", "reference_answer": "Mars"},
        ]
        with st.form("benchmark_form"):
            tasks_json = st.text_area(
                "Tasks JSON",
                value=json.dumps(default_tasks, indent=2),
                height=220,
            )
            variant_options = ["baseline", "memory", "reflexion", "prompt_optimization"]
            variants = st.multiselect("Variants", variant_options, default=variant_options)
            benchmark_model = st.text_input("Model override for benchmark", value="")
            benchmark_submitted = st.form_submit_button("Run Benchmark")

        if benchmark_submitted:
            try:
                tasks = json.loads(tasks_json)
                payload = {
                    "tasks": tasks,
                    "variants": variants,
                    "model": benchmark_model or None,
                }
                result = trigger_benchmark(API_BASE, payload)
                _remember_job(result["job_id"])
                st.success(f"Benchmark job created: {result['job_id']}")
                st.json(result)
                st.rerun()
            except Exception as exc:
                st.error(f"Benchmark trigger failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="Self-Evolving Agents Dashboard", layout="wide")
    st.title("Self-Evolving Agents Dashboard")
    st.caption(f"DB: {DB_PATH} | Benchmarks: {BENCHMARK_DIR}")

    page = st.sidebar.radio(
        "Page",
        ["Overview", "Runs", "Memory", "Benchmarks", "Control Plane"],
    )

    if page == "Overview":
        render_overview()
    elif page == "Runs":
        render_runs()
    elif page == "Memory":
        render_memory()
    elif page == "Benchmarks":
        render_benchmarks()
    else:
        render_control_plane()


if __name__ == "__main__":
    main()
