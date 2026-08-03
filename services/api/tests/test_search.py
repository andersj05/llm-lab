from collections.abc import Callable
from pathlib import Path
from time import sleep

from fastapi.testclient import TestClient
from llm_lab.app import create_app
from llm_lab.config import Settings
from llm_lab.runtime import GenerationResult
from llm_lab.schemas import CreateRunRequest, ModelInfo, TokenTrace


class SearchRuntime:
    def inspect(self) -> ModelInfo:
        return ModelInfo(path="/models/qwen", available=True, loaded=True)

    def generate(
        self,
        request: CreateRunRequest,
        on_text: Callable[[str], None],
        should_stop: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        if "Propose 2 substantially different" in request.prompt:
            output = """[
              {
                "name": "formal",
                "niche": "formal",
                "executor_prompt": "derive it",
                "coordinates": {"breadth": 0.2, "rigor": 0.9, "novelty": 0.3}
              },
              {
                "name": "analogy",
                "niche": "analogy",
                "executor_prompt": "reframe it",
                "coordinates": {"breadth": 0.8, "rigor": 0.4, "novelty": 0.9}
              }
            ]"""
        elif "Return only one JSON object" in request.prompt:
            output = (
                '{"residuals":{"correctness":0.01,"completeness":0.02,'
                '"constraints":0.0,"evidence":0.03},"feedback":"done","solved":true}'
            )
        else:
            output = "A checked candidate solution."
        on_text(output)
        return GenerationResult(
            output_text=output,
            tokens=[TokenTrace(index=0, token_id=1, text=output)],
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
        trace_token_limit=64,
    )


def test_residual_search_runs_branches_and_stops_when_solved(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path), SearchRuntime())) as client:
        created = client.post(
            "/api/v1/searches",
            json={
                "problem": "Find a candidate",
                "goal": "Produce a verified candidate",
                "options": {
                    "iterations": 3,
                    "branches": 2,
                    "branch_tokens": 64,
                    "idea_tokens": 64,
                    "evaluator_tokens": 64,
                },
            },
        )
        search_id = created.json()["id"]
        for _ in range(200):
            record = client.get(f"/api/v1/searches/{search_id}").json()
            if record["status"] in {"completed", "failed", "cancelled"}:
                break
            sleep(0.01)

    assert created.status_code == 202
    assert record["status"] == "completed"
    assert record["iteration"] == 0
    assert len(record["candidates"]) == 2
    assert record["best_candidate_id"]
    assert len(record["nodes"]) == 5
    assert [node["kind"] for node in record["nodes"]].count("executor") == 2
    assert [node["kind"] for node in record["nodes"]].count("evaluator") == 2
    assert all(node["status"] == "completed" for node in record["nodes"])
    assert all(candidate["evaluation"]["solved"] for candidate in record["candidates"])


def test_search_validation_rejects_a_single_branch(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path), SearchRuntime())) as client:
        response = client.post(
            "/api/v1/searches",
            json={
                "problem": "x",
                "goal": "y",
                "options": {"branches": 1},
            },
        )

    assert response.status_code == 422
