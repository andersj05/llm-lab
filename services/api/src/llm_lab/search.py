from __future__ import annotations

import asyncio
import json
import math
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from llm_lab.coordinator import RunCoordinator
from llm_lab.schemas import CreateRunRequest, GenerationOptions, RunRecord, RunStatus, TraceOptions


def _now() -> datetime:
    return datetime.now(UTC)


class SearchStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SearchNodeKind(StrEnum):
    BRAINSTORM = "brainstorm"
    EXECUTOR = "executor"
    EVALUATOR = "evaluator"


class SearchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iterations: int = Field(default=4, ge=1, le=50)
    branches: int = Field(default=4, ge=2, le=8)
    branch_tokens: int = Field(default=1024, ge=64, le=8192)
    idea_tokens: int = Field(default=768, ge=64, le=4096)
    evaluator_tokens: int = Field(default=512, ge=64, le=2048)
    solved_threshold: float = Field(default=0.08, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class CreateSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: str = Field(min_length=1, max_length=100_000)
    goal: str = Field(min_length=1, max_length=20_000)
    evaluator_rubric: str | None = Field(default=None, max_length=20_000)
    options: SearchOptions = Field(default_factory=SearchOptions)


class SearchIdea(BaseModel):
    id: str
    iteration: int
    name: str
    niche: str
    executor_prompt: str
    coordinates: dict[str, float]
    parent_candidate_id: str | None = None
    predicted_improvement: float | None = None


class SearchNode(BaseModel):
    id: str
    kind: SearchNodeKind
    iteration: int
    label: str
    status: RunStatus = RunStatus.QUEUED
    idea_id: str | None = None
    parent_node_id: str | None = None
    run_id: str | None = None
    output_excerpt: str | None = None
    error: str | None = None


class ResidualEvaluation(BaseModel):
    residuals: dict[str, float]
    feedback: str
    solved: bool = False

    @property
    def score(self) -> float:
        values = [min(1.0, max(0.0, value)) for value in self.residuals.values()]
        return math.sqrt(sum(value * value for value in values) / len(values)) if values else 1.0


class CandidateRecord(BaseModel):
    id: str
    idea: SearchIdea
    run_id: str
    evaluator_run_id: str
    output: str
    evaluation: ResidualEvaluation
    score: float
    created_at: datetime = Field(default_factory=_now)


class SearchRecord(BaseModel):
    id: str
    status: SearchStatus
    request: CreateSearchRequest
    iteration: int = 0
    trust_radius: float = 0.35
    nodes: list[SearchNode] = Field(default_factory=list)
    candidates: list[CandidateRecord] = Field(default_factory=list)
    best_candidate_id: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None


class SearchStore:
    """Small JSON store for restart-safe experiment inspection."""

    def __init__(self, root: Path) -> None:
        self.root = root / "searches"
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, record: SearchRecord) -> None:
        with self._lock:
            directory = self.root / record.id
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "state.json.tmp"
            temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(directory / "state.json")

    def get(self, search_id: str) -> SearchRecord | None:
        path = self.root / search_id / "state.json"
        if not path.is_file():
            return None
        return SearchRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[SearchRecord]:
        records = [
            record
            for path in self.root.glob("*/state.json")
            if (record := self.get(path.parent.name)) is not None
        ]
        return sorted(records, key=lambda record: record.created_at, reverse=True)


class SearchPolicy(Protocol):
    trust_radius: float

    def propose(self, record: SearchRecord, count: int) -> list[SearchIdea]: ...

    def observe(self, candidates: list[CandidateRecord]) -> None: ...


class ResidualTrustRegionPolicy:
    """Derivative-free local linear search over explicit semantic controls."""

    axes = ("breadth", "rigor", "novelty")

    def __init__(self, radius: float = 0.35) -> None:
        self.trust_radius = radius

    def observe(self, candidates: list[CandidateRecord]) -> None:
        ratios: list[float] = []
        by_id = {candidate.id: candidate for candidate in candidates}
        for candidate in candidates:
            parent_id = candidate.idea.parent_candidate_id
            predicted = candidate.idea.predicted_improvement
            if not parent_id or not predicted or parent_id not in by_id:
                continue
            actual = by_id[parent_id].score - candidate.score
            ratios.append(actual / predicted)
        if not ratios:
            return
        ratio = max(ratios)
        if ratio < 0.1:
            self.trust_radius = max(0.05, self.trust_radius * 0.5)
        elif ratio > 0.75:
            self.trust_radius = min(0.8, self.trust_radius * 1.5)

    def propose(self, record: SearchRecord, count: int) -> list[SearchIdea]:
        archive = _diverse_archive(record.candidates)
        if not archive:
            return _fallback_seed_ideas(record.iteration, count)

        best = min(archive, key=lambda candidate: candidate.score)
        gradient = self._fit_score_gradient(archive)
        norm = math.sqrt(sum(value * value for value in gradient.values()))
        if norm < 1e-8:
            gradient = {"breadth": -1.0, "rigor": 1.0, "novelty": 0.5}
            norm = math.sqrt(sum(value * value for value in gradient.values()))

        ideas: list[SearchIdea] = []
        for index in range(count):
            scale = 1.0 if index == 0 else 0.55
            step = {axis: -scale * self.trust_radius * gradient[axis] / norm for axis in self.axes}
            if index:
                exploratory_axis = self.axes[(index - 1) % len(self.axes)]
                step[exploratory_axis] += self.trust_radius * (0.6 if index % 2 else -0.6)
            coordinates = {
                axis: _clip(best.idea.coordinates.get(axis, 0.5) + step[axis]) for axis in self.axes
            }
            predicted = max(1e-4, -sum(gradient[axis] * step[axis] for axis in self.axes))
            ideas.append(
                SearchIdea(
                    id=uuid4().hex,
                    iteration=record.iteration,
                    name=f"Trust-region variation {index + 1} of {best.idea.name}",
                    niche=best.idea.niche if index == 0 else f"{best.idea.niche}-variation-{index}",
                    executor_prompt=_mutation_prompt(best, coordinates),
                    coordinates=coordinates,
                    parent_candidate_id=best.id,
                    predicted_improvement=predicted,
                )
            )
        return ideas

    def _fit_score_gradient(self, candidates: list[CandidateRecord]) -> dict[str, float]:
        gradient: dict[str, float] = {}
        mean_score = sum(candidate.score for candidate in candidates) / len(candidates)
        for axis in self.axes:
            values = [candidate.idea.coordinates.get(axis, 0.5) for candidate in candidates]
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) + 1e-3
            covariance = sum(
                (value - mean_value) * (candidate.score - mean_score)
                for value, candidate in zip(values, candidates, strict=True)
            )
            gradient[axis] = covariance / variance
        return gradient


class SearchCoordinator:
    def __init__(self, runs: RunCoordinator, store: SearchStore) -> None:
        self.runs = runs
        self.store = store
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancelled: set[str] = set()

    async def start(self) -> None:
        self.store.initialize()
        for record in self.store.list():
            if record.status in {SearchStatus.QUEUED, SearchStatus.RUNNING}:
                record.status = SearchStatus.FAILED
                record.error = "API restarted while this search was active"
                record.completed_at = _now()
                self.store.save(record)

    async def stop(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def submit(self, request: CreateSearchRequest) -> SearchRecord:
        record = SearchRecord(id=uuid4().hex, status=SearchStatus.QUEUED, request=request)
        self.store.save(record)
        self._tasks[record.id] = asyncio.create_task(self._execute(record.id))
        return record

    async def cancel(self, search_id: str) -> SearchRecord:
        record = self.store.get(search_id)
        if record is None:
            raise KeyError(search_id)
        self._cancelled.add(search_id)
        task = self._tasks.get(search_id)
        if task:
            task.cancel()
        for node in record.nodes:
            if node.run_id and not node.status.terminal:
                try:
                    await self.runs.cancel(node.run_id)
                except (KeyError, RuntimeError):
                    continue
                node.status = RunStatus.CANCELLED
        record.status = SearchStatus.CANCELLED
        record.completed_at = _now()
        self.store.save(record)
        return record

    async def _execute(self, search_id: str) -> None:
        record = self.store.get(search_id)
        if record is None:
            return
        policy = ResidualTrustRegionPolicy(record.trust_radius)
        try:
            record.status = SearchStatus.RUNNING
            self.store.save(record)
            ideas = await self._brainstorm(record)
            for iteration in range(record.request.options.iterations):
                if search_id in self._cancelled:
                    raise asyncio.CancelledError
                record.iteration = iteration
                if iteration:
                    policy.observe(record.candidates)
                    record.trust_radius = policy.trust_radius
                    ideas = policy.propose(record, record.request.options.branches)
                executor_nodes: list[SearchNode] = []
                for idea in ideas:
                    parent_node_id = next(
                        (
                            node.id
                            for node in reversed(record.nodes)
                            if node.kind is SearchNodeKind.BRAINSTORM
                        ),
                        None,
                    )
                    if idea.parent_candidate_id:
                        parent = next(
                            candidate
                            for candidate in record.candidates
                            if candidate.id == idea.parent_candidate_id
                        )
                        parent_node_id = next(
                            node.id
                            for node in record.nodes
                            if node.run_id == parent.evaluator_run_id
                        )
                    executor_nodes.append(
                        SearchNode(
                            id=uuid4().hex,
                            kind=SearchNodeKind.EXECUTOR,
                            iteration=iteration,
                            label=idea.name,
                            idea_id=idea.id,
                            parent_node_id=parent_node_id,
                        )
                    )
                record.nodes.extend(executor_nodes)
                self.store.save(record)
                await asyncio.gather(
                    *(
                        self._execute_idea(record, idea, node, index)
                        for index, (idea, node) in enumerate(
                            zip(ideas, executor_nodes, strict=True)
                        )
                    )
                )
                best = min(record.candidates, key=lambda candidate: candidate.score)
                record.best_candidate_id = best.id
                self.store.save(record)
                if best.evaluation.solved or best.score <= record.request.options.solved_threshold:
                    break
            record.status = SearchStatus.COMPLETED
            record.completed_at = _now()
            self.store.save(record)
        except asyncio.CancelledError:
            record.status = SearchStatus.CANCELLED
            record.completed_at = _now()
            self.store.save(record)
        except Exception as error:
            record.status = SearchStatus.FAILED
            record.error = str(error)
            record.completed_at = _now()
            self.store.save(record)
        finally:
            self._tasks.pop(search_id, None)
            self._cancelled.discard(search_id)

    async def _brainstorm(self, record: SearchRecord) -> list[SearchIdea]:
        count = record.request.options.branches
        prompt = (
            f"Open problem:\n{record.request.problem}\n\nGoal:\n{record.request.goal}\n\n"
            f"Propose {count} substantially different solution strategies. Return only a JSON "
            "array. Each item must have name, niche, executor_prompt, and coordinates with numeric "
            "breadth, rigor, and novelty values between 0 and 1."
        )
        node = SearchNode(
            id=uuid4().hex,
            kind=SearchNodeKind.BRAINSTORM,
            iteration=-1,
            label="Generate strategy set",
        )
        record.nodes.append(node)
        self.store.save(record)
        run = await self._run(
            record,
            node,
            prompt,
            record.request.options.idea_tokens,
            False,
            17,
        )
        payload = _extract_json(run.output_text or "")
        if not isinstance(payload, list):
            return _fallback_seed_ideas(0, count)
        ideas: list[SearchIdea] = []
        for item in payload[:count]:
            if not isinstance(item, dict):
                continue
            coordinates = (
                item.get("coordinates") if isinstance(item.get("coordinates"), dict) else {}
            )
            ideas.append(
                SearchIdea(
                    id=uuid4().hex,
                    iteration=0,
                    name=str(item.get("name") or "Unnamed strategy"),
                    niche=str(item.get("niche") or f"strategy-{len(ideas)}"),
                    executor_prompt=str(item.get("executor_prompt") or item.get("name") or ""),
                    coordinates={
                        axis: _clip(float(coordinates.get(axis, 0.5)))
                        for axis in ResidualTrustRegionPolicy.axes
                    },
                )
            )
        return ideas or _fallback_seed_ideas(0, count)

    async def _execute_idea(
        self,
        record: SearchRecord,
        idea: SearchIdea,
        node: SearchNode,
        branch_index: int,
    ) -> CandidateRecord:
        request = record.request
        prompt = (
            f"Problem:\n{request.problem}\n\nGoal:\n{request.goal}\n\n"
            f"Strategy:\n{idea.executor_prompt}\n\n"
            "Develop and check a concrete candidate solution. "
            "Make useful inferential jumps, but identify each jump and test it against the goal."
        )
        run = await self._run(
            record,
            node,
            prompt,
            request.options.branch_tokens,
            True,
            request.options.seed + record.iteration * 100 + branch_index,
        )
        evaluation_node = SearchNode(
            id=uuid4().hex,
            kind=SearchNodeKind.EVALUATOR,
            iteration=record.iteration,
            label=f"Evaluate {idea.name}",
            idea_id=idea.id,
            parent_node_id=node.id,
        )
        record.nodes.append(evaluation_node)
        self.store.save(record)
        evaluation_run = await self._run(
            record,
            evaluation_node,
            _evaluation_prompt(request, run.output_text or ""),
            request.options.evaluator_tokens,
            False,
            request.options.seed + 10_000 + record.iteration * 100 + branch_index,
        )
        evaluation = _parse_evaluation(evaluation_run.output_text or "")
        candidate = CandidateRecord(
            id=uuid4().hex,
            idea=idea,
            run_id=run.id,
            evaluator_run_id=evaluation_run.id,
            output=run.output_text or "",
            evaluation=evaluation,
            score=evaluation.score,
        )
        record.candidates.append(candidate)
        self.store.save(record)
        return candidate

    async def _run(
        self,
        search: SearchRecord,
        node: SearchNode,
        prompt: str,
        max_tokens: int,
        thinking: bool,
        seed: int,
    ) -> RunRecord:
        run = await self.runs.submit(
            CreateRunRequest(
                prompt=prompt,
                system_prompt=(
                    "You are a node in a residual-guided nonlinear search graph. "
                    "Be explicit, falsifiable, and concise."
                ),
                generation=GenerationOptions(
                    max_new_tokens=max_tokens,
                    temperature=0.65 if thinking else 0.2,
                    top_p=0.95,
                    do_sample=True,
                    seed=seed,
                    enable_thinking=thinking,
                ),
                traces=TraceOptions(logprobs=False),
            )
        )
        node.run_id = run.id
        node.status = run.status
        self.store.save(search)
        while True:
            current = await asyncio.to_thread(self.runs.repository.get_run, run.id)
            if current is None:
                raise RuntimeError(f"Run {run.id} disappeared")
            if node.status is not current.status:
                node.status = current.status
                self.store.save(search)
            if current.status.terminal:
                node.output_excerpt = (current.output_text or "")[:1200] or None
                node.error = current.error
                self.store.save(search)
                if current.status is not RunStatus.COMPLETED:
                    raise RuntimeError(current.error or f"Run {run.id} did not complete")
                return current
            await asyncio.sleep(0.25)


def _evaluation_prompt(request: CreateSearchRequest, candidate: str) -> str:
    rubric = request.evaluator_rubric or (
        "Judge correctness, completeness, constraint satisfaction, and evidential support."
    )
    return (
        f"Problem:\n{request.problem}\n\nGoal:\n{request.goal}\n\nRubric:\n{rubric}\n\n"
        f"Candidate:\n{candidate}\n\nReturn only one JSON object with: residuals (an object "
        "containing correctness, completeness, constraints, and evidence numbers from 0 for no "
        "remaining error to 1 for total failure), feedback (specific next correction), and solved "
        "(boolean). Do not reward eloquence; estimate distance from the stated goal."
    )


def _parse_evaluation(text: str) -> ResidualEvaluation:
    payload = _extract_json(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("residuals"), dict):
        return ResidualEvaluation(
            residuals={"parse": 1.0, "correctness": 1.0},
            feedback="Evaluator did not return valid structured residuals.",
        )
    residuals: dict[str, float] = {}
    for name, value in payload["residuals"].items():
        try:
            residuals[str(name)] = _clip(float(value))
        except (TypeError, ValueError):
            residuals[str(name)] = 1.0
    return ResidualEvaluation(
        residuals=residuals or {"unknown": 1.0},
        feedback=str(payload.get("feedback") or ""),
        solved=bool(payload.get("solved", False)),
    )


def _extract_json(text: str) -> object | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _diverse_archive(candidates: list[CandidateRecord]) -> list[CandidateRecord]:
    niches: dict[str, CandidateRecord] = {}
    for candidate in candidates:
        current = niches.get(candidate.idea.niche)
        if current is None or candidate.score < current.score:
            niches[candidate.idea.niche] = candidate
    return sorted(niches.values(), key=lambda candidate: candidate.score)[:12]


def _fallback_seed_ideas(iteration: int, count: int) -> list[SearchIdea]:
    templates = (
        (
            "Decompose",
            "decomposition",
            "Break the problem into constraints and solve each dependency.",
        ),
        (
            "Construct",
            "first-principles",
            "Build a direct solution from first principles and verify it.",
        ),
        (
            "Challenge",
            "adversarial",
            "Seek counterexamples, hidden assumptions, and repair the best candidate.",
        ),
        ("Reframe", "analogy", "Change representation or abstraction and derive a new route."),
        (
            "Backward",
            "backward-chain",
            "Start at the goal and infer necessary preceding conditions.",
        ),
        ("Synthesize", "cross-domain", "Combine two distant methods and test the resulting jump."),
        (
            "Formalize",
            "formal",
            "Translate the problem into explicit variables, invariants, and checks.",
        ),
        (
            "Empirical",
            "simulation",
            "Design a small thought experiment or simulation to discriminate ideas.",
        ),
    )
    return [
        SearchIdea(
            id=uuid4().hex,
            iteration=iteration,
            name=name,
            niche=niche,
            executor_prompt=prompt,
            coordinates={
                "breadth": _clip(0.25 + 0.11 * index),
                "rigor": _clip(0.8 - 0.07 * index),
                "novelty": _clip(0.2 + 0.12 * index),
            },
        )
        for index, (name, niche, prompt) in enumerate(templates[:count])
    ]


def _mutation_prompt(parent: CandidateRecord, coordinates: dict[str, float]) -> str:
    feedback = parent.evaluation.feedback or "Reduce the remaining residuals."
    return (
        f"Improve this prior candidate:\n{parent.output}\n\nEvaluator feedback:\n{feedback}\n\n"
        f"Target controls: breadth={coordinates['breadth']:.2f}, "
        f"rigor={coordinates['rigor']:.2f}, novelty={coordinates['novelty']:.2f}. "
        "Preserve what worked, make a nontrivial corrective jump, and verify the changed claims."
    )


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))
