from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from llm_lab import __version__
from llm_lab.artifacts import ArtifactStore
from llm_lab.config import Settings, get_settings
from llm_lab.coordinator import RunCoordinator, artifact_media_type
from llm_lab.repository import RunRepository
from llm_lab.runtime import ModelRuntime, Runtime
from llm_lab.schemas import (
    CreateRunRequest,
    DeleteRunResponse,
    HealthResponse,
    ModelInfo,
    RunEvent,
    RunRecord,
)


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    repository = RunRepository(resolved_settings.database_path)
    artifacts = ArtifactStore(resolved_settings.data_dir)
    resolved_runtime = runtime or ModelRuntime(resolved_settings)
    coordinator = RunCoordinator(repository, artifacts, resolved_runtime)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        repository.initialize()
        artifacts.initialize()
        await coordinator.start()
        yield
        await coordinator.stop()

    app = FastAPI(
        title="LLM Lab API",
        version=__version__,
        description="Local model execution and interpretability trace orchestration",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.state.settings = resolved_settings
    app.state.repository = repository
    app.state.artifacts = artifacts
    app.state.runtime = resolved_runtime
    app.state.coordinator = coordinator

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        model = await _in_thread(resolved_runtime.inspect)
        return HealthResponse(
            status="ok" if model.available else "degraded",
            version=__version__,
            model=model,
        )

    @app.get("/api/v1/models/local", response_model=ModelInfo)
    async def local_model() -> ModelInfo:
        return await _in_thread(resolved_runtime.inspect)

    @app.post("/api/v1/runs", response_model=RunRecord, status_code=202)
    async def create_run(payload: CreateRunRequest) -> RunRecord:
        return await coordinator.submit(payload)

    @app.get("/api/v1/runs", response_model=list[RunRecord])
    async def list_runs(limit: int = Query(default=50, ge=1, le=200)) -> list[RunRecord]:
        return await _in_thread(repository.list_runs, limit)

    @app.get("/api/v1/runs/{run_id}", response_model=RunRecord)
    async def get_run(run_id: str) -> RunRecord:
        record = await _in_thread(repository.get_run, run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return record

    @app.get("/api/v1/runs/{run_id}/events", response_model=list[RunEvent])
    async def list_events(run_id: str, after: int = Query(default=0, ge=0)) -> list[RunEvent]:
        if await _in_thread(repository.get_run, run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return await _in_thread(repository.list_events, run_id, after)

    @app.get("/api/v1/runs/{run_id}/result")
    async def get_run_result(run_id: str) -> dict[str, Any]:
        if await _in_thread(repository.get_run, run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        result = await _in_thread(artifacts.read_json, run_id, "result.json")
        if not isinstance(result, dict):
            raise HTTPException(status_code=404, detail="Run result is not ready")
        return result

    @app.post("/api/v1/runs/{run_id}/cancel", response_model=RunRecord)
    async def cancel_run(run_id: str) -> RunRecord:
        try:
            return await coordinator.cancel(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Run not found") from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.delete("/api/v1/runs/{run_id}", response_model=DeleteRunResponse)
    async def delete_run(run_id: str) -> DeleteRunResponse:
        try:
            await coordinator.delete(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Run not found") from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return DeleteRunResponse(id=run_id)

    @app.get("/api/v1/runs/{run_id}/artifacts/{name}")
    async def get_artifact(run_id: str, name: str) -> FileResponse:
        path = artifacts.resolve_artifact(run_id, name)
        if path is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(path, media_type=artifact_media_type(path), filename=path.name)

    @app.websocket("/api/v1/runs/{run_id}/events/ws")
    async def run_events(websocket: WebSocket, run_id: str, after: int = 0) -> None:
        record = await _in_thread(repository.get_run, run_id)
        if record is None:
            await websocket.close(code=4404, reason="Run not found")
            return
        await websocket.accept()
        try:
            async for event in coordinator.events.subscribe(run_id, after):
                await websocket.send_json(event.model_dump(mode="json"))
                if event.type.value.startswith("run.") and event.type.value in {
                    "run.completed",
                    "run.failed",
                    "run.cancelled",
                }:
                    break
        except WebSocketDisconnect:
            return

    return app


async def _in_thread(function: Any, *args: Any) -> Any:
    import asyncio

    return await asyncio.to_thread(function, *args)


app = create_app()
