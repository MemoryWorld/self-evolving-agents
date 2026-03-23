"""API tests."""

from fastapi.testclient import TestClient

from self_evolving.core.agent import BaseAgent
from self_evolving.service.api import create_app


def test_run_qa_and_fetch_persisted_data(tmp_path, monkeypatch):
    monkeypatch.setattr(BaseAgent, "_call_llm", lambda self, messages: "ANSWER: Paris")

    app = create_app(str(tmp_path / "sea.db"))
    client = TestClient(app)

    response = client.post(
        "/runs/qa",
        json={
            "goal": "What is the capital of France?",
            "reference_answer": "Paris",
            "agent_id": "api-agent",
            "use_memory": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"]
    assert payload["agent_id"] == "api-agent"
    assert payload["run_id"]

    runs = client.get("/runs")
    assert runs.status_code == 200
    assert len(runs.json()) == 1

    detail = client.get(f"/runs/{payload['run_id']}")
    assert detail.status_code == 200
    assert detail.json()["run_id"] == payload["run_id"]
