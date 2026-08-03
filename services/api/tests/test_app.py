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
        self,
        request: CreateRunRequest,
        on_text: Callable[[str], None],
        should_stop: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        on_text("workspace")
        return GenerationResult(
            output_text="workspace",
            tokens=[TokenTrace(index=0, token_id=1, text="workspace", logprob=-0.1)],
        )


class StoppableRuntime(FakeRuntime):
    def generate(
        self,
        request: CreateRunRequest,
        on_text: Callable[[str], None],
        should_stop: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        for _ in range(200):
            if should_stop and should_stop():
                break
            sleep(0.005)
        on_text("partial")
        return GenerationResult(
            output_text="partial",
            tokens=[TokenTrace(index=0, token_id=1, text="partial")],
        )


class NonFiniteRuntime(FakeRuntime):
    def generate(
        self,
        request: CreateRunRequest,
        on_text: Callable[[str], None],
        should_stop: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            output_text="safe",
            tokens=[TokenTrace(index=0, token_id=1, text="safe", logprob=None)],
            trace_summary={"legacy_non_finite": float("-inf")},
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

        for _ in range(50):
            events = client.get(f"/api/v1/runs/{run_id}/events").json()
            if events and events[-1]["type"] == "run.completed":
                break
            sleep(0.01)

    assert created.status_code == 202
    assert record["output_text"] == "workspace"
    assert [event["type"] for event in events] == [
        "run.queued",
        "model.loading",
        "run.started",
        "token.generated",
        "run.completed",
    ]


def test_completed_run_and_artifacts_can_be_deleted(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    with TestClient(create_app(configured, FakeRuntime())) as client:
        created = client.post("/api/v1/runs", json={"prompt": "Delete this"})
        run_id = created.json()["id"]
        for _ in range(50):
            record = client.get(f"/api/v1/runs/{run_id}").json()
            if record["status"] == "completed":
                break
            sleep(0.01)

        assert (configured.data_dir / run_id / "result.json").is_file()
        deleted = client.delete(f"/api/v1/runs/{run_id}")

        assert deleted.status_code == 200
        assert deleted.json() == {"id": run_id, "deleted": True}
        assert client.get(f"/api/v1/runs/{run_id}").status_code == 404
        assert not (configured.data_dir / run_id).exists()


def test_result_endpoint_returns_strict_json_for_diagnostics(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    with TestClient(create_app(configured, NonFiniteRuntime())) as client:
        created = client.post("/api/v1/runs", json={"prompt": "Inspect this"})
        run_id = created.json()["id"]
        for _ in range(50):
            record = client.get(f"/api/v1/runs/{run_id}").json()
            if record["status"] == "completed":
                break
            sleep(0.01)

        result = client.get(f"/api/v1/runs/{run_id}/result")
        artifact_text = (configured.data_dir / run_id / "result.json").read_text()

    assert result.status_code == 200
    assert result.json()["traces"]["legacy_non_finite"] is None
    assert "Infinity" not in artifact_text


def test_running_run_can_be_stopped_before_deletion(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path), StoppableRuntime())) as client:
        created = client.post("/api/v1/runs", json={"prompt": "Keep going"})
        run_id = created.json()["id"]
        for _ in range(50):
            record = client.get(f"/api/v1/runs/{run_id}").json()
            if record["status"] == "running":
                break
            sleep(0.01)

        assert client.delete(f"/api/v1/runs/{run_id}").status_code == 409
        stopped = client.post(f"/api/v1/runs/{run_id}/cancel")
        for _ in range(50):
            record = client.get(f"/api/v1/runs/{run_id}").json()
            if record["status"] == "cancelled":
                break
            sleep(0.01)

    assert stopped.status_code == 200
    assert record["status"] == "cancelled"
    assert record["output_text"] == "partial"
