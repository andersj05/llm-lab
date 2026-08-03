"use client";

import type {
  CandidateRecord,
  CreateSearchRequest,
  HealthResponse,
  RunRecord,
  SearchNode,
  SearchRecord,
} from "@llm-lab/contracts";
import { FormEvent, useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";

interface SearchLabProps {
  health: HealthResponse | null;
  onInspectRun: (runId: string) => Promise<void>;
  onRunsChanged: () => Promise<RunRecord[]>;
}

const terminalSearchStatuses = new Set<SearchRecord["status"]>([
  "completed",
  "failed",
  "cancelled",
]);

const initialRequest: CreateSearchRequest = {
  problem: "",
  goal: "",
  evaluator_rubric: "",
  options: {
    iterations: 2,
    branches: 3,
    branch_tokens: 768,
    idea_tokens: 512,
    evaluator_tokens: 384,
    solved_threshold: 0.08,
    seed: 42,
  },
};

export function SearchLab({ health, onInspectRun, onRunsChanged }: SearchLabProps) {
  const [request, setRequest] = useState<CreateSearchRequest>(initialRequest);
  const [searches, setSearches] = useState<SearchRecord[]>([]);
  const [activeSearch, setActiveSearch] = useState<SearchRecord | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [followLive, setFollowLive] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const liveNodeId = useMemo(() => {
    if (!activeSearch?.nodes.length) return null;
    return (
      [...activeSearch.nodes]
        .reverse()
        .find((node) => ["running", "loading"].includes(node.status)) ??
      [...activeSearch.nodes].reverse().find((node) => node.status === "queued") ??
      activeSearch.nodes.at(-1)
    )?.id ?? null;
  }, [activeSearch]);
  const effectiveSelectedNodeId = followLive ? liveNodeId : selectedNodeId;
  const selectedNode = useMemo(
    () =>
      activeSearch?.nodes.find((node) => node.id === effectiveSelectedNodeId) ?? null,
    [activeSearch, effectiveSelectedNodeId],
  );
  const selectedCandidate = useMemo(
    () => candidateForNode(activeSearch, selectedNode),
    [activeSearch, selectedNode],
  );
  const bestCandidate = useMemo(
    () =>
      activeSearch?.candidates.find(
        (candidate) => candidate.id === activeSearch.best_candidate_id,
      ) ??
      [...(activeSearch?.candidates ?? [])].sort((left, right) => left.score - right.score)[0] ??
      null,
    [activeSearch],
  );
  const progress = activeSearch
    ? Math.min(
        1,
        (activeSearch.iteration + (terminalSearchStatuses.has(activeSearch.status) ? 1 : 0)) /
          activeSearch.request.options.iterations,
      )
    : 0;

  useEffect(() => {
    let disposed = false;
    api
      .searches()
      .then((records) => {
        if (disposed) return;
        setSearches(records);
        if (records[0]) {
          setActiveSearch(records[0]);
        }
      })
      .catch((reason: Error) => {
        if (!disposed) setError(reason.message);
      })
      .finally(() => {
        if (!disposed) setHistoryLoading(false);
      });
    return () => {
      disposed = true;
    };
  }, []);

  const activeSearchId = activeSearch?.id;
  const activeSearchStatus = activeSearch?.status;

  useEffect(() => {
    if (!activeSearchId || !activeSearchStatus) return;
    if (terminalSearchStatuses.has(activeSearchStatus)) return;
    const searchId = activeSearchId;
    let disposed = false;
    let timer: number | undefined;

    async function poll() {
      try {
        const updated = await api.search(searchId);
        if (disposed) return;
        setActiveSearch(updated);
        setSearches((current) => mergeSearch(current, updated));
        if (terminalSearchStatuses.has(updated.status)) {
          setStopping(false);
          await onRunsChanged();
          return;
        }
        timer = window.setTimeout(poll, 650);
      } catch (reason) {
        if (!disposed) {
          setError(reason instanceof Error ? reason.message : "Unable to refresh search");
        }
      }
    }

    timer = window.setTimeout(poll, 250);
    return () => {
      disposed = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [activeSearchId, activeSearchStatus, onRunsChanged]);

  async function startSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!request.problem.trim() || !request.goal.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    setFollowLive(true);
    setSelectedNodeId(null);
    try {
      const created = await api.createSearch({
        ...request,
        problem: request.problem.trim(),
        goal: request.goal.trim(),
        evaluator_rubric: request.evaluator_rubric?.trim() || null,
      });
      setActiveSearch(created);
      setSearches((current) => mergeSearch(current, created));
      await onRunsChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to start search");
    } finally {
      setSubmitting(false);
    }
  }

  async function stopSearch() {
    if (!activeSearch || terminalSearchStatuses.has(activeSearch.status)) return;
    setStopping(true);
    setError(null);
    try {
      const stopped = await api.cancelSearch(activeSearch.id);
      setActiveSearch(stopped);
      setSearches((current) => mergeSearch(current, stopped));
      await onRunsChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to stop search");
    } finally {
      setStopping(false);
    }
  }

  async function selectHistory(searchId: string) {
    if (!searchId) return;
    setError(null);
    setFollowLive(false);
    setSelectedNodeId(null);
    try {
      const record = await api.search(searchId);
      setActiveSearch(record);
      setSearches((current) => mergeSearch(current, record));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to load experiment");
    }
  }

  function selectNode(nodeId: string) {
    setFollowLive(false);
    setSelectedNodeId(nodeId);
  }

  return (
    <div className="search-lab">
      <section className="search-lab__intro">
        <div>
          <span>Residual-guided search laboratory</span>
          <h2>Watch strategies branch, fail, and converge.</h2>
          <p>
            Every box is a persisted model run. Follow queue pressure, inspect any reasoning trace,
            and compare evaluator residuals as the semantic trust region moves the next generation.
          </p>
        </div>
        <label className="search-history-picker">
          <span>Experiment history</span>
          <select
            disabled={historyLoading || !searches.length}
            onChange={(event) => void selectHistory(event.target.value)}
            value={activeSearch?.id ?? ""}
          >
            {!searches.length ? <option value="">No saved experiments</option> : null}
            {searches.map((search) => (
              <option key={search.id} value={search.id}>
                {historyLabel(search)}
              </option>
            ))}
          </select>
        </label>
      </section>

      <form className="panel search-composer" onSubmit={startSearch}>
        <div className="search-composer__prompts">
          <label>
            <span>Open problem</span>
            <textarea
              onChange={(event) =>
                setRequest((current) => ({ ...current, problem: event.target.value }))
              }
              placeholder="What should the search solve or discover?"
              required
              rows={5}
              value={request.problem}
            />
          </label>
          <label>
            <span>Success condition</span>
            <textarea
              onChange={(event) =>
                setRequest((current) => ({ ...current, goal: event.target.value }))
              }
              placeholder="Define an observable, falsifiable goal."
              required
              rows={5}
              value={request.goal}
            />
          </label>
        </div>
        <div className="search-controls">
          <NumberControl
            label="Iterations"
            max={12}
            min={1}
            onChange={(value) => updateOption(setRequest, "iterations", value)}
            value={request.options.iterations}
          />
          <NumberControl
            label="Branches"
            max={8}
            min={2}
            onChange={(value) => updateOption(setRequest, "branches", value)}
            value={request.options.branches}
          />
          <NumberControl
            label="Branch tokens"
            max={8192}
            min={64}
            onChange={(value) => updateOption(setRequest, "branch_tokens", value)}
            step={64}
            value={request.options.branch_tokens}
          />
          <label>
            <span>Solved residual</span>
            <input
              max={1}
              min={0}
              onChange={(event) =>
                updateOption(setRequest, "solved_threshold", Number(event.target.value))
              }
              step={0.01}
              type="number"
              value={request.options.solved_threshold}
            />
          </label>
          <label className="search-controls__rubric">
            <span>Evaluator rubric <i>optional</i></span>
            <input
              onChange={(event) =>
                setRequest((current) => ({
                  ...current,
                  evaluator_rubric: event.target.value,
                }))
              }
              placeholder="Hard constraints, tests, evidence requirements…"
              type="text"
              value={request.evaluator_rubric ?? ""}
            />
          </label>
          <div className="search-actions">
            {activeSearch && !terminalSearchStatuses.has(activeSearch.status) ? (
              <button
                className="secondary-button stop-button"
                disabled={stopping}
                onClick={() => void stopSearch()}
                type="button"
              >
                {stopping ? "Stopping…" : "Stop search"}
              </button>
            ) : null}
            <button
              className="primary-button"
              disabled={
                submitting ||
                !request.problem.trim() ||
                !request.goal.trim() ||
                !health?.model.available
              }
              type="submit"
            >
              {submitting ? "Starting…" : "Start experiment"}
            </button>
          </div>
        </div>
        {error ? <p className="message message--error">{error}</p> : null}
      </form>

      {activeSearch ? (
        <>
          <section className="search-overview" aria-label="Experiment status">
            <div className="search-overview__identity">
              <div>
                <span className={"search-status search-status--" + activeSearch.status}>
                  {activeSearch.status}
                </span>
                <strong>{activeSearch.request.problem}</strong>
              </div>
              <div className="search-progress" aria-label="Iteration progress">
                <i style={{ width: Math.round(progress * 100) + "%" }} />
              </div>
            </div>
            <Metric label="Iteration" value={iterationLabel(activeSearch)} />
            <Metric
              label="Graph nodes"
              value={
                activeSearch.nodes.filter((node) => node.status === "completed").length +
                " / " +
                activeSearch.nodes.length
              }
            />
            <Metric
              label="Best residual"
              value={bestCandidate ? bestCandidate.score.toFixed(3) : "—"}
            />
            <Metric label="Trust radius" value={activeSearch.trust_radius.toFixed(3)} />
          </section>

          <section className="panel search-graph-panel">
            <div className="section-heading search-section-heading">
              <div>
                <span>Live execution</span>
                <h2>Search graph</h2>
              </div>
              <div className="search-graph-legend">
                <span><i className="is-brainstorm" />strategy set</span>
                <span><i className="is-executor" />executor</span>
                <span><i className="is-evaluator" />evaluator</span>
                <button
                  className={followLive ? "secondary-button is-active" : "secondary-button"}
                  onClick={() => setFollowLive(true)}
                  type="button"
                >
                  {followLive ? "Following live" : "Resume live"}
                </button>
              </div>
            </div>
            <SearchGraph
              record={activeSearch}
              selectedNodeId={effectiveSelectedNodeId}
              onSelect={selectNode}
            />
          </section>

          <section className="search-analysis-grid">
            <article className="panel search-inspector">
              <div className="section-heading">
                <div>
                  <span>Selected node</span>
                  <h2>{selectedNode?.label ?? "Choose a graph node"}</h2>
                </div>
                {selectedNode ? (
                  <span className={"search-status search-status--" + selectedNode.status}>
                    {selectedNode.status}
                  </span>
                ) : null}
              </div>
              {selectedNode ? (
                <>
                  <dl className="search-node-facts">
                    <div><dt>Stage</dt><dd>{selectedNode.kind}</dd></div>
                    <div>
                      <dt>Iteration</dt>
                      <dd>{selectedNode.iteration < 0 ? "Seed" : selectedNode.iteration + 1}</dd>
                    </div>
                    <div>
                      <dt>Run</dt>
                      <dd>{selectedNode.run_id?.slice(0, 10) ?? "Waiting"}</dd>
                    </div>
                  </dl>
                  {selectedCandidate ? (
                    <ResidualList candidate={selectedCandidate} />
                  ) : null}
                  <div className="search-node-output">
                    {selectedNode.output_excerpt ? (
                      <pre>{selectedNode.output_excerpt}</pre>
                    ) : (
                      <p>
                        {selectedNode.status === "queued"
                          ? "Waiting for the single GPU worker."
                          : "Output will appear when this run reaches a terminal state."}
                      </p>
                    )}
                  </div>
                  {selectedNode.error ? (
                    <p className="message message--error">{selectedNode.error}</p>
                  ) : null}
                  {selectedNode.run_id ? (
                    <button
                      className="secondary-button"
                      onClick={() => void onInspectRun(selectedNode.run_id!)}
                      type="button"
                    >
                      Open full diagnostics
                    </button>
                  ) : null}
                </>
              ) : (
                <p className="empty-copy">Select a node to inspect its output and residuals.</p>
              )}
            </article>

            <article className="panel search-convergence">
              <div className="section-heading">
                <div>
                  <span>Benchmark signal</span>
                  <h2>Best residual by iteration</h2>
                </div>
                <small>lower is better</small>
              </div>
              <ConvergenceChart candidates={activeSearch.candidates} />
              {bestCandidate ? (
                <div className="search-best">
                  <span>Current best · {bestCandidate.idea.niche}</span>
                  <strong>{bestCandidate.idea.name}</strong>
                  <p>{bestCandidate.evaluation.feedback}</p>
                  <button
                    className="secondary-button"
                    onClick={() => void onInspectRun(bestCandidate.run_id)}
                    type="button"
                  >
                    Inspect candidate
                  </button>
                </div>
              ) : (
                <p className="empty-copy">Scores appear as evaluators complete.</p>
              )}
            </article>
          </section>

          <section className="panel search-leaderboard">
            <div className="section-heading">
              <div>
                <span>Candidate archive</span>
                <h2>Residual leaderboard</h2>
              </div>
              <small>{activeSearch.candidates.length} evaluated candidates</small>
            </div>
            <div className="search-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Iteration</th>
                    <th>Strategy</th>
                    <th>Niche</th>
                    <th>Residual</th>
                    <th>Components</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {[...activeSearch.candidates]
                    .sort((left, right) => left.score - right.score)
                    .map((candidate, index) => (
                      <tr key={candidate.id}>
                        <td>{index + 1}</td>
                        <td>{candidate.idea.iteration + 1}</td>
                        <td><strong>{candidate.idea.name}</strong></td>
                        <td><span className="search-niche">{candidate.idea.niche}</span></td>
                        <td><code>{candidate.score.toFixed(3)}</code></td>
                        <td>
                          <div className="search-residual-inline">
                            {Object.entries(candidate.evaluation.residuals).map(([name, value]) => (
                              <span key={name}>{name} {value.toFixed(2)}</span>
                            ))}
                          </div>
                        </td>
                        <td>
                          <button
                            onClick={() => void onInspectRun(candidate.run_id)}
                            type="button"
                          >
                            Inspect
                          </button>
                        </td>
                      </tr>
                    ))}
                  {!activeSearch.candidates.length ? (
                    <tr><td colSpan={7}>No evaluated candidates yet.</td></tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : (
        <section className="search-empty">
          <span>←</span>
          <h2>Configure the first benchmark above.</h2>
          <p>The live graph will appear as soon as strategy generation is queued.</p>
        </section>
      )}
    </div>
  );
}

function SearchGraph({
  record,
  selectedNodeId,
  onSelect,
}: {
  record: SearchRecord;
  selectedNodeId: string | null;
  onSelect: (nodeId: string) => void;
}) {
  if (!record.nodes.length) {
    return (
      <div className="search-graph-empty">
        <i />
        <span>Preparing strategy generation…</span>
      </div>
    );
  }

  const layout = graphLayout(record.nodes);
  return (
    <div className="search-graph-viewport">
      <svg
        aria-label="Live directed graph of brainstorm, executor, and evaluator model runs"
        className="search-graph-svg"
        role="img"
        viewBox={"0 0 " + layout.width + " " + layout.height}
      >
        <title>Search execution graph</title>
        <desc>Nodes show persisted model runs and edges show strategy and evaluator dependencies.</desc>
        <g className="search-graph-edges">
          {record.nodes.map((node) => {
            const parent = node.parent_node_id ? layout.positions.get(node.parent_node_id) : null;
            const current = layout.positions.get(node.id);
            if (!parent || !current) return null;
            const startX = parent.x + 176;
            const startY = parent.y + 35;
            const endX = current.x;
            const endY = current.y + 35;
            const bend = Math.max(28, (endX - startX) * 0.5);
            return (
              <path
                className={"search-edge search-edge--" + node.status}
                d={
                  "M " + startX + " " + startY +
                  " C " + (startX + bend) + " " + startY +
                  ", " + (endX - bend) + " " + endY +
                  ", " + endX + " " + endY
                }
                key={"edge-" + node.id}
              />
            );
          })}
        </g>
        {record.nodes.map((node) => {
          const position = layout.positions.get(node.id);
          if (!position) return null;
          const candidate = candidateForNode(record, node);
          const selected = selectedNodeId === node.id;
          return (
            <g
              aria-label={
                node.label + ", " + node.kind + ", " + node.status +
                (candidate ? ", residual " + candidate.score.toFixed(3) : "")
              }
              className={
                "search-graph-node search-graph-node--" + node.kind +
                " search-graph-node--" + node.status +
                (selected ? " is-selected" : "")
              }
              key={node.id}
              onClick={() => onSelect(node.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onSelect(node.id);
              }}
              role="button"
              tabIndex={0}
              transform={"translate(" + position.x + " " + position.y + ")"}
            >
              <rect height="70" rx="9" width="176" />
              <circle cx="15" cy="17" r="4" />
              <text className="search-graph-node__kind" x="27" y="20">
                {node.kind}{node.iteration >= 0 ? " · I" + (node.iteration + 1) : ""}
              </text>
              <text className="search-graph-node__label" x="14" y="42">
                {truncate(node.label, 24)}
              </text>
              <text className="search-graph-node__status" x="14" y="59">
                {node.status}
              </text>
              {candidate ? (
                <text className="search-graph-node__score" textAnchor="end" x="162" y="59">
                  r {candidate.score.toFixed(3)}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ConvergenceChart({ candidates }: { candidates: CandidateRecord[] }) {
  const points = bestByIteration(candidates);
  if (!points.length) {
    return <div className="search-chart-empty">Waiting for residual measurements…</div>;
  }
  const width = 520;
  const height = 180;
  const left = 38;
  const right = 16;
  const top = 18;
  const bottom = 32;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const maxIteration = Math.max(1, ...points.map((point) => point.iteration));
  const coordinates = points.map((point) => ({
    ...point,
    x: left + (point.iteration / maxIteration) * plotWidth,
    y: top + point.score * plotHeight,
  }));
  const path = coordinates
    .map((point, index) => (index ? "L " : "M ") + point.x + " " + point.y)
    .join(" ");

  return (
    <svg
      aria-label="Line chart of the lowest residual found in each completed iteration"
      className="search-convergence-chart"
      role="img"
      viewBox={"0 0 " + width + " " + height}
    >
      <title>Best residual by iteration</title>
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
        const y = top + tick * plotHeight;
        return (
          <g key={tick}>
            <line x1={left} x2={width - right} y1={y} y2={y} />
            <text textAnchor="end" x={left - 8} y={y + 3}>{tick.toFixed(2)}</text>
          </g>
        );
      })}
      <path className="search-convergence-chart__line" d={path} />
      {coordinates.map((point) => (
        <g key={point.iteration}>
          <circle cx={point.x} cy={point.y} r="5" />
          <text textAnchor="middle" x={point.x} y={height - 10}>
            I{point.iteration + 1}
          </text>
          <text className="search-convergence-chart__value" textAnchor="middle" x={point.x} y={point.y - 10}>
            {point.score.toFixed(3)}
          </text>
        </g>
      ))}
    </svg>
  );
}

function ResidualList({ candidate }: { candidate: CandidateRecord }) {
  return (
    <div className="search-residual-list">
      <div>
        <span>RMS residual</span>
        <strong>{candidate.score.toFixed(3)}</strong>
      </div>
      {Object.entries(candidate.evaluation.residuals).map(([name, value]) => (
        <label key={name}>
          <span>{name}</span>
          <i><b style={{ width: Math.min(100, value * 100) + "%" }} /></i>
          <code>{value.toFixed(2)}</code>
        </label>
      ))}
      {candidate.evaluation.feedback ? <p>{candidate.evaluation.feedback}</p> : null}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="search-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function NumberControl({
  label,
  max,
  min,
  onChange,
  step = 1,
  value,
}: {
  label: string;
  max: number;
  min: number;
  onChange: (value: number) => void;
  step?: number;
  value: number;
}) {
  return (
    <label>
      <span>{label}</span>
      <input
        max={max}
        min={min}
        onChange={(event) => onChange(Number(event.target.value))}
        step={step}
        type="number"
        value={value}
      />
    </label>
  );
}

function updateOption(
  setRequest: React.Dispatch<React.SetStateAction<CreateSearchRequest>>,
  key: keyof CreateSearchRequest["options"],
  value: number,
) {
  setRequest((current) => ({
    ...current,
    options: { ...current.options, [key]: value },
  }));
}

function candidateForNode(
  record: SearchRecord | null,
  node: SearchNode | null,
): CandidateRecord | null {
  if (!record || !node?.idea_id) return null;
  return record.candidates.find((candidate) => candidate.idea.id === node.idea_id) ?? null;
}

function bestByIteration(candidates: CandidateRecord[]) {
  const scores = new Map<number, number>();
  for (const candidate of candidates) {
    const current = scores.get(candidate.idea.iteration);
    if (current === undefined || candidate.score < current) {
      scores.set(candidate.idea.iteration, candidate.score);
    }
  }
  return [...scores.entries()]
    .sort(([left], [right]) => left - right)
    .map(([iteration, score]) => ({ iteration, score }));
}

function graphLayout(nodes: SearchNode[]) {
  const positions = new Map<string, { x: number; y: number }>();
  const iterations = Math.max(0, ...nodes.map((node) => node.iteration));
  const maxBranches = Math.max(
    1,
    ...Array.from({ length: iterations + 1 }, (_, iteration) =>
      nodes.filter(
        (node) => node.iteration === iteration && node.kind === "executor",
      ).length,
    ),
  );

  for (const node of nodes) {
    if (node.kind === "brainstorm") {
      positions.set(node.id, { x: 24, y: Math.max(38, ((maxBranches - 1) * 112) / 2 + 38) });
      continue;
    }
    const peers = nodes.filter(
      (peer) => peer.iteration === node.iteration && peer.kind === node.kind,
    );
    let row = peers.findIndex((peer) => peer.id === node.id);
    if (node.kind === "evaluator" && node.idea_id) {
      const executors = nodes.filter(
        (peer) => peer.iteration === node.iteration && peer.kind === "executor",
      );
      const executorRow = executors.findIndex((peer) => peer.idea_id === node.idea_id);
      if (executorRow >= 0) row = executorRow;
    }
    const baseX = 230 + node.iteration * 430;
    positions.set(node.id, {
      x: node.kind === "executor" ? baseX : baseX + 220,
      y: 38 + Math.max(0, row) * 112,
    });
  }

  return {
    height: Math.max(250, maxBranches * 112 + 50),
    positions,
    width: Math.max(720, 230 + (iterations + 1) * 430),
  };
}

function mergeSearch(current: SearchRecord[], updated: SearchRecord) {
  return [updated, ...current.filter((record) => record.id !== updated.id)].sort(
    (left, right) =>
      new Date(right.created_at).getTime() - new Date(left.created_at).getTime(),
  );
}

function iterationLabel(search: SearchRecord) {
  return (
    Math.min(search.iteration + 1, search.request.options.iterations) +
    " / " +
    search.request.options.iterations
  );
}

function historyLabel(search: SearchRecord) {
  const problem = truncate(search.request.problem, 42);
  return problem + " · " + search.status;
}

function truncate(value: string, length: number) {
  return value.length > length ? value.slice(0, length - 1) + "…" : value;
}

