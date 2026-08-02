from collections.abc import Callable
from pathlib import Path
from time import sleep

from fastapi.testclient import TestClient
from llm_lab.app import create_app
from llm_lab.config import Settings
from llm_lab.runtime import GenerationResult
from llm_lab.schemas import CreateRunRequest, ModelInfo, TokenTrace


class FakeRuntime:
    def inspect(self) -> ModelInfo:
        return ModelInfo(
            path="/models/qwen",
            available=True,
            loaded=True,
            architecture="Qwen3ForCausalLM",
            layers=28,
            hidden_size=1024,
            vocabulary_size=151936,
        )

    def generate(
        self, request: CreateRunRequest, on_text: Callable[[str], None]
    ) -> GenerationResult:
        on_text("workspace")
        return GenerationResult(
            output_text="workspace",
            tokens=[TokenTrace(index=0, token_id=1, text="workspace", logprob=-0.1)],
        )


def settings(tmp_path: Path) -> Settings:
    return Settings(
        model_path=tmp_path / "model",
        data_dir=tmp_path / "runs",
        database_path=tmp_path / "runs.sqlite3",
        device="cpu",
        dtype="float32",
        cors_origins=("http://localhost:3000",),
        jlens_path=None,
        trace_token_limit=256,
    )


def test_health_reports_model_metadata(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path), FakeRuntime())) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["model"]["architecture"] == "Qwen3ForCausalLM"


def test_run_completes_and_emits_events(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path), FakeRuntime())) as client:
        created = client.post("/api/v1/runs", json={"prompt": "Inspect this"})
        run_id = created.json()["id"]
        for _ in range(50):
            record = client.get(f"/api/v1/runs/{run_id}").json()
            if record["status"] == "completed":
                break
            sleep(0.01)

        events = client.get(f"/api/v1/runs/{run_id}/events").json()

    assert created.status_code == 202
    assert record["output_text"] == "workspace"
    assert [event["type"] for event in events] == [
        "run.queued",
        "model.loading",
        "run.started",
        "token.generated",
        "run.completed",
    ]
