from pathlib import Path

from llm_lab.repository import RunRepository
from llm_lab.schemas import CreateRunRequest, EventType, RunStatus


def test_repository_persists_run_lifecycle(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path / "runs.sqlite3")
    repository.initialize()
    request = CreateRunRequest(prompt="What is represented at this layer?")

    created = repository.create_run("run-1", request)
    event = repository.append_event("run-1", EventType.RUN_QUEUED)
    completed = repository.update_run(
        "run-1",
        RunStatus.COMPLETED,
        output_text="A latent concept.",
        tokens_generated=4,
    )

    assert created.status is RunStatus.QUEUED
    assert event.sequence == 1
    assert completed.status is RunStatus.COMPLETED
    assert completed.output_text == "A latent concept."
    assert repository.list_events("run-1") == [event]
    assert repository.list_runs() == [completed]
