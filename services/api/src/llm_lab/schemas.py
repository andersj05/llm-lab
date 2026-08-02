from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunStatus(StrEnum):
    QUEUED = "queued"
    LOADING = "loading"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED}


class EventType(StrEnum):
    RUN_QUEUED = "run.queued"
    RUN_STARTED = "run.started"
    MODEL_LOADING = "model.loading"
    MODEL_READY = "model.ready"
    TOKEN_GENERATED = "token.generated"
    TRACE_READY = "trace.ready"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"


class GenerationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    do_sample: bool = True
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    enable_thinking: bool = True

    @model_validator(mode="after")
    def normalize_greedy_generation(self) -> GenerationOptions:
        if not self.do_sample:
            self.temperature = 0
        return self


class TraceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logprobs: bool = True
    hidden_states: bool = False
    attentions: bool = False
    jspace: bool = False
    persist_raw: bool = False


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=100_000)
    system_prompt: str | None = Field(default=None, max_length=20_000)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    traces: TraceOptions = Field(default_factory=TraceOptions)


class TokenTrace(BaseModel):
    index: int
    token_id: int
    text: str
    logprob: float | None = None
    top_alternatives: list[dict[str, int | float | str]] = Field(default_factory=list)


class RunEvent(BaseModel):
    run_id: str
    sequence: int
    type: EventType
    created_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    id: str
    status: RunStatus
    prompt: str
    system_prompt: str | None
    request: CreateRunRequest
    output_text: str | None = None
    error: str | None = None
    tokens_generated: int = 0
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ModelInfo(BaseModel):
    path: str
    available: bool
    loaded: bool
    architecture: str | None = None
    parameter_dtype: str | None = None
    layers: int | None = None
    hidden_size: int | None = None
    vocabulary_size: int | None = None
    weight_bytes: int | None = None
    device: str | None = None
    jspace_available: bool = False
    jspace_reason: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str = "llm-lab-api"
    version: str
    model: ModelInfo
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
