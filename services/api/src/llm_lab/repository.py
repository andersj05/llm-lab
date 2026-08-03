from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from llm_lab.schemas import CreateRunRequest, EventType, RunEvent, RunRecord, RunStatus


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


class RunRepository:
    """Small SQLite repository with one connection per operation."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._write_lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    system_prompt TEXT,
                    request_json TEXT NOT NULL,
                    output_text TEXT,
                    error TEXT,
                    tokens_generated INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_created_at
                ON runs(created_at DESC)
                """
            )
            connection.execute("PRAGMA optimize")

    def create_run(self, run_id: str, request: CreateRunRequest) -> RunRecord:
        created_at = _timestamp()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, status, prompt, system_prompt, request_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    RunStatus.QUEUED,
                    request.prompt,
                    request.system_prompt,
                    request.model_dump_json(),
                    created_at,
                ),
            )
        record = self.get_run(run_id)
        if record is None:
            raise RuntimeError(f"Run {run_id} was not persisted")
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._to_run(row) if row else None

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._to_run(row) for row in rows]

    def update_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        output_text: str | None = None,
        error: str | None = None,
        tokens_generated: int | None = None,
    ) -> RunRecord:
        now = _timestamp()
        started_at = now if status in {RunStatus.LOADING, RunStatus.RUNNING} else None
        completed_at = now if status.terminal else None
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?,
                    output_text = COALESCE(?, output_text),
                    error = COALESCE(?, error),
                    tokens_generated = COALESCE(?, tokens_generated),
                    started_at = COALESCE(started_at, ?),
                    completed_at = COALESCE(completed_at, ?)
                WHERE id = ?
                """,
                (
                    status,
                    output_text,
                    error,
                    tokens_generated,
                    started_at,
                    completed_at,
                    run_id,
                ),
            )
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    def append_event(
        self, run_id: str, event_type: EventType, payload: dict[str, object] | None = None
    ) -> RunEvent:
        created_at = _timestamp()
        payload_json = json.dumps(payload or {}, separators=(",", ":"))
        with self._write_lock, self._connect() as connection:
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO events (run_id, sequence, type, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, sequence, event_type, created_at, payload_json),
            )
        return RunEvent(
            run_id=run_id,
            sequence=sequence,
            type=event_type,
            created_at=datetime.fromisoformat(created_at),
            payload=json.loads(payload_json),
        )

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[RunEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM events
                WHERE run_id = ? AND sequence > ?
                ORDER BY sequence ASC
                """,
                (run_id, after_sequence),
            ).fetchall()
        return [
            RunEvent(
                run_id=row["run_id"],
                sequence=row["sequence"],
                type=EventType(row["type"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        ]

    def delete_run(self, run_id: str) -> bool:
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cursor.rowcount > 0

    @staticmethod
    def _to_run(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=row["id"],
            status=RunStatus(row["status"]),
            prompt=row["prompt"],
            system_prompt=row["system_prompt"],
            request=CreateRunRequest.model_validate_json(row["request_json"]),
            output_text=row["output_text"],
            error=row["error"],
            tokens_generated=row["tokens_generated"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=(datetime.fromisoformat(row["started_at"]) if row["started_at"] else None),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
        )
