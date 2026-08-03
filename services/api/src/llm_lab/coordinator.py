from __future__ import annotations

import asyncio
import threading
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from llm_lab.artifacts import ArtifactStore
from llm_lab.repository import RunRepository
from llm_lab.runtime import Runtime
from llm_lab.schemas import CreateRunRequest, EventType, RunEvent, RunRecord, RunStatus


class EventBus:
    def __init__(self, repository: RunRepository) -> None:
        self.repository = repository
        self._subscribers: dict[str, set[asyncio.Queue[RunEvent]]] = defaultdict(set)

    async def publish(
        self, run_id: str, event_type: EventType, payload: dict[str, object] | None = None
    ) -> RunEvent:
        event = await asyncio.to_thread(self.repository.append_event, run_id, event_type, payload)
        for queue in tuple(self._subscribers[run_id]):
            queue.put_nowait(event)
        return event

    async def subscribe(self, run_id: str, after_sequence: int = 0) -> AsyncIterator[RunEvent]:
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        self._subscribers[run_id].add(queue)
        last_sequence = after_sequence
        try:
            existing = await asyncio.to_thread(self.repository.list_events, run_id, after_sequence)
            for event in existing:
                last_sequence = max(last_sequence, event.sequence)
                yield event
            while True:
                event = await queue.get()
                if event.sequence <= last_sequence:
                    continue
                last_sequence = event.sequence
                yield event
        finally:
            self._subscribers[run_id].discard(queue)
            if not self._subscribers[run_id]:
                self._subscribers.pop(run_id, None)


class RunCoordinator:
    def __init__(
        self,
        repository: RunRepository,
        artifacts: ArtifactStore,
        runtime: Runtime,
    ) -> None:
        self.repository = repository
        self.artifacts = artifacts
        self.runtime = runtime
        self.events = EventBus(repository)
        self._queue: asyncio.Queue[tuple[str, CreateRunRequest]] = asyncio.Queue()
        self._cancelled: set[str] = set()
        self._stop_signals: dict[str, threading.Event] = {}
        self._active_run_id: str | None = None
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker(), name="llm-lab-model-worker")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._worker_task
        self._worker_task = None

    async def submit(self, request: CreateRunRequest) -> RunRecord:
        run_id = uuid4().hex
        record = await asyncio.to_thread(self.repository.create_run, run_id, request)
        self.artifacts.write_json(
            run_id,
            "manifest.json",
            {"run_id": run_id, "request": request.model_dump(mode="json")},
        )
        await self.events.publish(run_id, EventType.RUN_QUEUED)
        self._stop_signals[run_id] = threading.Event()
        await self._queue.put((run_id, request))
        return record

    async def cancel(self, run_id: str) -> RunRecord:
        record = await asyncio.to_thread(self.repository.get_run, run_id)
        if record is None:
            raise KeyError(run_id)
        if record.status.terminal:
            return record
        if record.status is RunStatus.QUEUED and self._active_run_id != run_id:
            self._cancelled.add(run_id)
            record = await asyncio.to_thread(
                self.repository.update_run, run_id, RunStatus.CANCELLED
            )
            await self.events.publish(run_id, EventType.RUN_CANCELLED)
            return record
        stop_signal = self._stop_signals.get(run_id)
        if stop_signal is None:
            raise RuntimeError("The run is no longer attached to the model worker")
        stop_signal.set()
        return record

    async def delete(self, run_id: str) -> None:
        record = await asyncio.to_thread(self.repository.get_run, run_id)
        if record is None:
            raise KeyError(run_id)
        if not record.status.terminal:
            raise RuntimeError("Stop the run before deleting it")
        await asyncio.to_thread(self.artifacts.delete_run, run_id)
        deleted = await asyncio.to_thread(self.repository.delete_run, run_id)
        if not deleted:
            raise KeyError(run_id)

    async def _worker(self) -> None:
        while True:
            run_id, request = await self._queue.get()
            try:
                if run_id in self._cancelled:
                    continue
                self._active_run_id = run_id
                await self._execute(run_id, request)
            finally:
                self._active_run_id = None
                self._cancelled.discard(run_id)
                self._stop_signals.pop(run_id, None)
                self._queue.task_done()

    async def _execute(self, run_id: str, request: CreateRunRequest) -> None:
        loop = asyncio.get_running_loop()
        stop_signal = self._stop_signals[run_id]
        try:
            await asyncio.to_thread(self.repository.update_run, run_id, RunStatus.LOADING)
            await self.events.publish(run_id, EventType.MODEL_LOADING)

            chunk_index = 0

            def on_text(chunk: str) -> None:
                nonlocal chunk_index
                index = chunk_index
                chunk_index += 1
                future = asyncio.run_coroutine_threadsafe(
                    self.events.publish(
                        run_id,
                        EventType.TOKEN_GENERATED,
                        {"index": index, "text": chunk},
                    ),
                    loop,
                )
                future.result(timeout=10)

            await asyncio.to_thread(self.repository.update_run, run_id, RunStatus.RUNNING)
            await self.events.publish(run_id, EventType.RUN_STARTED)
            result = await asyncio.to_thread(
                self.runtime.generate,
                request,
                on_text,
                stop_signal.is_set,
            )

            result_payload = {
                "output_text": result.output_text,
                "tokens": [token.model_dump(mode="json") for token in result.tokens],
                "generation": result.generation,
                "traces": result.trace_summary,
            }
            result_path = self.artifacts.write_json(run_id, "result.json", result_payload)
            artifact_names = [result_path.name]
            if result.raw_tensors:
                tensor_path = self.artifacts.write_tensors(run_id, result.raw_tensors)
                artifact_names.append(tensor_path.name)
            if result.trace_summary:
                await self.events.publish(
                    run_id,
                    EventType.TRACE_READY,
                    {"artifacts": artifact_names},
                )

            if stop_signal.is_set():
                record = await asyncio.to_thread(
                    self.repository.update_run,
                    run_id,
                    RunStatus.CANCELLED,
                    output_text=result.output_text,
                    tokens_generated=len(result.tokens),
                )
                await self.events.publish(
                    run_id,
                    EventType.RUN_CANCELLED,
                    {"tokens_generated": record.tokens_generated, "artifacts": artifact_names},
                )
                return

            record = await asyncio.to_thread(
                self.repository.update_run,
                run_id,
                RunStatus.COMPLETED,
                output_text=result.output_text,
                tokens_generated=len(result.tokens),
            )
            await self.events.publish(
                run_id,
                EventType.RUN_COMPLETED,
                {
                    "tokens_generated": record.tokens_generated,
                    "artifacts": artifact_names,
                },
            )
        except Exception as error:
            if stop_signal.is_set():
                await asyncio.to_thread(
                    self.repository.update_run,
                    run_id,
                    RunStatus.CANCELLED,
                    error=str(error),
                )
                await self.events.publish(
                    run_id,
                    EventType.RUN_CANCELLED,
                    {"message": "Run stopped"},
                )
                return
            await asyncio.to_thread(
                self.repository.update_run,
                run_id,
                RunStatus.FAILED,
                error=str(error),
            )
            await self.events.publish(run_id, EventType.RUN_FAILED, {"message": str(error)})


def artifact_media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".parquet": "application/vnd.apache.parquet",
        ".safetensors": "application/octet-stream",
    }.get(path.suffix, "application/octet-stream")
