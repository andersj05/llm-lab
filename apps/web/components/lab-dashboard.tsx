"use client";

import type {
  CreateRunRequest,
  HealthResponse,
  RunEvent,
  RunRecord,
  RunResult,
} from "@llm-lab/contracts";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { JSpaceViewer } from "@/components/jspace-viewer";
import { SearchLab } from "@/components/search-lab";
import { TraceMatrix } from "@/components/trace-matrix";
import { api, subscribeToRun } from "@/lib/api";
import { formatBytes, formatTimestamp } from "@/lib/format";

const terminalStatuses = new Set<RunRecord["status"]>([
  "completed",
  "failed",
  "cancelled",
]);

type DiagnosticState =
  | "no-run"
  | "not-requested"
  | "waiting"
  | "loading"
  | "ready"
  | "unavailable"
  | "error";

const initialRequest: CreateRunRequest = {
  prompt: "",
  system_prompt: null,
  generation: {
    max_new_tokens: 1024,
    temperature: 0.7,
    top_p: 0.9,
    do_sample: true,
    seed: 42,
    enable_thinking: true,
  },
  traces: {
    logprobs: true,
    hidden_states: true,
    attentions: false,
    jspace: false,
    persist_raw: false,
  },
};

export function LabDashboard() {
  const [activeTab, setActiveTab] = useState<"diagnostics" | "reasoning">("diagnostics");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [activeRun, setActiveRun] = useState<RunRecord | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [request, setRequest] = useState<CreateRunRequest>(initialRequest);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [streamedText, setStreamedText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [clearingHistory, setClearingHistory] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultStatus, setResultStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle",
  );
  const [resultError, setResultError] = useState<string | null>(null);
  const lastSequence = useRef(0);
  const resultRequest = useRef(0);

  const refreshRuns = useCallback(async () => {
    const nextRuns = await api.runs();
    setRuns(nextRuns);
    return nextRuns;
  }, []);

  const fetchResult = useCallback(async (runId: string) => {
    const requestId = resultRequest.current + 1;
    resultRequest.current = requestId;
    setResultStatus("loading");
    setResultError(null);
    try {
      const nextResult = await api.result(runId);
      if (resultRequest.current !== requestId) return;
      setResult(nextResult);
      setResultStatus("ready");
    } catch (reason) {
      if (resultRequest.current !== requestId) return;
      setResult(null);
      setResultStatus("error");
      setResultError(reason instanceof Error ? reason.message : "Unable to load diagnostics");
    }
  }, []);

  const resetSelection = useCallback(() => {
    lastSequence.current = 0;
    setActiveRun(null);
    setResult(null);
    setResultStatus("idle");
    setResultError(null);
    resultRequest.current += 1;
    setEvents([]);
    setStreamedText("");
    setError(null);
  }, []);

  const loadRun = useCallback(async (run: RunRecord) => {
    lastSequence.current = 0;
    setEvents([]);
    setActiveRun(run);
    setStreamedText(run.output_text ?? "");
    setResult(null);
    setResultStatus("idle");
    setResultError(null);
    resultRequest.current += 1;
    setError(null);
    try {
      const history = await api.events(run.id);
      lastSequence.current = history.at(-1)?.sequence ?? 0;
      setEvents(history);
      if (run.status === "completed" || run.status === "cancelled") {
        await fetchResult(run.id);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to load run");
    }
  }, [fetchResult]);

  useEffect(() => {
    Promise.all([api.health(), api.runs()])
      .then(([nextHealth, nextRuns]) => {
        setHealth(nextHealth);
        setRuns(nextRuns);
        setRequest((current) => ({
          ...current,
          traces: { ...current.traces, jspace: nextHealth.model.jspace_available },
        }));
        if (nextRuns[0]) void loadRun(nextRuns[0]);
      })
      .catch((reason: Error) => setError(reason.message));
  }, [loadRun]);

  useEffect(() => {
    if (!activeRun || terminalStatuses.has(activeRun.status)) return;
    return subscribeToRun(
      activeRun.id,
      lastSequence.current,
      (event) => {
        lastSequence.current = Math.max(lastSequence.current, event.sequence);
        setEvents((current) =>
          current.some((item) => item.sequence === event.sequence)
            ? current
            : [...current, event],
        );
        if (event.type === "token.generated" && typeof event.payload.text === "string") {
          setStreamedText((current) => current + event.payload.text);
        }
        if (["run.completed", "run.failed", "run.cancelled"].includes(event.type)) {
          setStopping(false);
          Promise.all([api.run(activeRun.id), api.runs()])
            .then(async ([updated, nextRuns]) => {
              setActiveRun(updated);
              setRuns(nextRuns);
              if (updated.status === "completed" || updated.status === "cancelled") {
                await fetchResult(updated.id);
              }
            })
            .catch((reason: Error) => setError(reason.message));
        }
      },
      () => setError("The live event stream disconnected."),
    );
  }, [activeRun, fetchResult]);

  async function submitRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setResult(null);
    setResultStatus("idle");
    setResultError(null);
    resultRequest.current += 1;
    setEvents([]);
    lastSequence.current = 0;
    setStreamedText("");
    try {
      const created = await api.createRun(request);
      setActiveRun(created);
      await refreshRuns();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to create run");
    } finally {
      setSubmitting(false);
    }
  }

  async function stopRun() {
    if (!activeRun || terminalStatuses.has(activeRun.status)) return;
    setStopping(true);
    setError(null);
    try {
      await api.cancelRun(activeRun.id);
    } catch (reason) {
      setStopping(false);
      setError(reason instanceof Error ? reason.message : "Unable to stop run");
    }
  }

  async function deleteRun(run: RunRecord) {
    if (!terminalStatuses.has(run.status)) return;
    if (!window.confirm("Delete this run and all of its stored artifacts?")) return;
    setDeletingId(run.id);
    setError(null);
    try {
      await api.deleteRun(run.id);
      const nextRuns = await refreshRuns();
      if (activeRun?.id === run.id) {
        const nextRun = nextRuns[0];
        if (nextRun) await loadRun(nextRun);
        else resetSelection();
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to delete run");
    } finally {
      setDeletingId(null);
    }
  }

  async function clearHistory() {
    const deletableRuns = runs.filter((run) => terminalStatuses.has(run.status));
    if (!deletableRuns.length) return;
    if (
      !window.confirm(
        `Delete ${deletableRuns.length} finished run${deletableRuns.length === 1 ? "" : "s"} and their artifacts?`,
      )
    ) {
      return;
    }
    setClearingHistory(true);
    setError(null);
    try {
      for (const run of deletableRuns) await api.deleteRun(run.id);
      const nextRuns = await refreshRuns();
      const currentStillExists = nextRuns.some((run) => run.id === activeRun?.id);
      if (!currentStillExists) {
        const nextRun = nextRuns[0];
        if (nextRun) await loadRun(nextRun);
        else resetSelection();
      }
    } catch (reason) {
      await refreshRuns();
      setError(reason instanceof Error ? reason.message : "Unable to clear run history");
    } finally {
      setClearingHistory(false);
    }
  }

  const inspectReasoningRun = useCallback(
    async (runId: string) => {
      const run = await api.run(runId);
      setActiveTab("diagnostics");
      await loadRun(run);
    },
    [loadRun],
  );

  const latestPhase = events.at(-1)?.type ?? activeRun?.status ?? "ready";
  const tokenTraces = useMemo(() => result?.tokens ?? [], [result]);
  const confidence = useMemo(() => {
    const logprobs = tokenTraces.flatMap((token) =>
      token.logprob === null ? [] : [Math.exp(token.logprob)],
    );
    if (!logprobs.length) return null;
    return logprobs.reduce((sum, value) => sum + value, 0) / logprobs.length;
  }, [tokenTraces]);
  const isActive = Boolean(activeRun && !terminalStatuses.has(activeRun.status));
  const outputText = streamedText || activeRun?.output_text || "";
  const maxTokenLimit = Math.min(32_768, health?.model.max_context_tokens ?? 32_768);
  const finishedRunCount = runs.filter((run) => terminalStatuses.has(run.status)).length;
  const layerDiagnostic = diagnosticLabel(
    activeRun,
    activeRun?.request.traces.hidden_states ?? false,
    Boolean(result?.traces.residual_update_norms?.length),
    resultStatus,
  );
  const likelihoodDiagnostic = diagnosticLabel(
    activeRun,
    activeRun?.request.traces.logprobs ?? false,
    tokenTraces.some((token) => token.logprob !== null),
    resultStatus,
  );
  const jspaceDiagnostic = diagnosticLabel(
    activeRun,
    activeRun?.request.traces.jspace ?? false,
    Boolean(result?.traces.jspace),
    resultStatus,
  );

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="app-title">
          <h1>LLM Lab</h1>
          <span>Local model workspace</span>
        </div>
        <nav className="app-tabs" aria-label="Workspace" role="tablist">
          <button
            aria-selected={activeTab === "diagnostics"}
            className={activeTab === "diagnostics" ? "is-active" : ""}
            onClick={() => setActiveTab("diagnostics")}
            role="tab"
            type="button"
          >
            Diagnostics
          </button>
          <button
            aria-selected={activeTab === "reasoning"}
            className={activeTab === "reasoning" ? "is-active" : ""}
            onClick={() => setActiveTab("reasoning")}
            role="tab"
            type="button"
          >
            Search Lab
          </button>
        </nav>
        <div className="model-status">
          <StatusDot active={health?.model.available ?? false} />
          <span>{health?.model.architecture ?? "API unavailable"}</span>
          <span className="model-status__detail">
            {health?.model.device ?? (health?.model.loaded ? "loaded" : "not loaded")}
          </span>
        </div>
      </header>

      <div className="app-body" hidden={activeTab !== "diagnostics"}>
        <aside className="history-sidebar">
          <div className="history-header">
            <div>
              <h2>History</h2>
              <span>{runs.length} runs</span>
            </div>
            <button className="secondary-button" onClick={resetSelection} type="button">
              New run
            </button>
          </div>

          <div className="history-list">
            {runs.length ? (
              runs.map((run) => (
                <div
                  className={`history-item ${activeRun?.id === run.id ? "is-selected" : ""}`}
                  key={run.id}
                >
                  <button className="history-select" onClick={() => void loadRun(run)} type="button">
                    <span className={`run-state run-state--${run.status}`} />
                    <span className="history-copy">
                      <strong>{run.prompt}</strong>
                      <small>
                        {formatTimestamp(run.created_at)} · {run.status} · {run.tokens_generated} tokens
                      </small>
                    </span>
                  </button>
                  <button
                    aria-label={`Delete run ${run.id}`}
                    className="delete-button"
                    disabled={!terminalStatuses.has(run.status) || deletingId === run.id}
                    onClick={() => void deleteRun(run)}
                    title={terminalStatuses.has(run.status) ? "Delete run" : "Stop this run first"}
                    type="button"
                  >
                    {deletingId === run.id ? "Deleting" : "Delete"}
                  </button>
                </div>
              ))
            ) : (
              <div className="empty-history">
                <p>No runs yet</p>
                <span>Your local experiments will appear here.</span>
              </div>
            )}
          </div>

          <div className="history-footer">
            <button
              className="text-button danger-text"
              disabled={!finishedRunCount || clearingHistory}
              onClick={() => void clearHistory()}
              type="button"
            >
              {clearingHistory ? "Clearing history…" : `Clear finished (${finishedRunCount})`}
            </button>
            <div className="model-summary">
              <span>{formatBytes(health?.model.weight_bytes ?? null)}</span>
              <span>{health?.model.layers ?? "—"} layers</span>
              <span>{health?.model.max_context_tokens?.toLocaleString() ?? "—"} context</span>
            </div>
          </div>
        </aside>

        <section className="workspace-main">
          <form className="composer panel" onSubmit={submitRun}>
            <div className="section-heading">
              <div>
                <h2>Prompt</h2>
                <p>Run Qwen locally and capture bounded observability data.</p>
              </div>
              <span className={`status-badge status-badge--${activeRun?.status ?? "ready"}`}>
                {latestPhase.replace(".", " ")}
              </span>
            </div>

            <label className="prompt-input">
              <span className="sr-only">Prompt</span>
              <textarea
                onChange={(event) => setRequest({ ...request, prompt: event.target.value })}
                placeholder="Ask the model something…"
                required
                rows={6}
                value={request.prompt}
              />
            </label>

            <div className="settings-row">
              <label className="number-field">
                <span>Max output tokens</span>
                <input
                  min={1}
                  max={maxTokenLimit}
                  onChange={(event) =>
                    setRequest({
                      ...request,
                      generation: {
                        ...request.generation,
                        max_new_tokens: Number(event.target.value),
                      },
                    })
                  }
                  type="number"
                  value={request.generation.max_new_tokens}
                />
                <small>Up to {maxTokenLimit.toLocaleString()}</small>
              </label>
              <label className="number-field">
                <span>Temperature</span>
                <input
                  min={0}
                  max={2}
                  step={0.1}
                  onChange={(event) =>
                    setRequest({
                      ...request,
                      generation: {
                        ...request.generation,
                        temperature: Number(event.target.value),
                      },
                    })
                  }
                  type="number"
                  value={request.generation.temperature}
                />
              </label>
              <div className="toggle-list" aria-label="Run options">
                <OptionToggle
                  checked={request.generation.enable_thinking}
                  label="Thinking"
                  onChange={(checked) =>
                    setRequest({
                      ...request,
                      generation: { ...request.generation, enable_thinking: checked },
                    })
                  }
                />
                <OptionToggle
                  checked={request.traces.logprobs}
                  label="Logprobs"
                  onChange={(checked) =>
                    setRequest({ ...request, traces: { ...request.traces, logprobs: checked } })
                  }
                />
                <OptionToggle
                  checked={request.traces.hidden_states}
                  label="Activations"
                  onChange={(checked) =>
                    setRequest({
                      ...request,
                      traces: { ...request.traces, hidden_states: checked },
                    })
                  }
                />
                <OptionToggle
                  checked={request.traces.attentions}
                  label="Attention"
                  onChange={(checked) =>
                    setRequest({ ...request, traces: { ...request.traces, attentions: checked } })
                  }
                />
                <OptionToggle
                  checked={request.traces.jspace}
                  disabled={!health?.model.jspace_available}
                  label="J-space"
                  onChange={(checked) =>
                    setRequest({ ...request, traces: { ...request.traces, jspace: checked } })
                  }
                />
              </div>
            </div>

            <div className="composer-actions">
              <p>
                Thinking changes the generated response; it does not enable diagnostics. Logprobs,
                Activations, Attention, and J-space control capture independently.
              </p>
              <div className="button-group">
                {isActive ? (
                  <button
                    className="secondary-button stop-button"
                    disabled={stopping}
                    onClick={() => void stopRun()}
                    type="button"
                  >
                    {stopping ? "Stopping…" : "Stop"}
                  </button>
                ) : null}
                <button
                  className="primary-button"
                  disabled={submitting || !health?.model.available || !request.prompt.trim()}
                  type="submit"
                >
                  {submitting ? "Queuing…" : "Run model"}
                </button>
              </div>
            </div>
            {error ? <p className="message message--error">{error}</p> : null}
          </form>

          <section className="panel output-panel">
            <div className="section-heading section-heading--compact">
              <div>
                <h2>Output</h2>
                <p>{activeRun ? `Run ${activeRun.id.slice(0, 8)}` : "No run selected"}</p>
              </div>
              <div className="output-metrics">
                <span>{activeRun?.tokens_generated ?? result?.tokens.length ?? 0} tokens</span>
                <span>{confidence === null ? "No likelihood" : `${(confidence * 100).toFixed(1)}% mean p`}</span>
              </div>
            </div>
            {activeRun?.error ? <p className="message message--error">{activeRun.error}</p> : null}
            <div className="output-copy" aria-live="polite">
              {outputText ? (
                <p>{outputText}</p>
              ) : (
                <p className="placeholder-copy">Generated text will stream here.</p>
              )}
            </div>
            {result?.generation ? (
              <div className="run-metadata">
                <span>{result.generation.input_tokens} input tokens</span>
                <span>{result.generation.context_limit.toLocaleString()} context window</span>
                <span>
                  {result.generation.logprobs_captured
                    ? "Logprobs captured"
                    : `Logprobs skipped above ${result.generation.logprob_token_limit.toLocaleString()} tokens`}
                </span>
              </div>
            ) : null}
          </section>

          <section className="panel diagnostics-overview">
            <div className="diagnostics-overview__copy">
              <span>Diagnostics</span>
              <strong>
                {activeRun ? `Run ${activeRun.id.slice(0, 8)}` : "Select or start a run"}
              </strong>
              <p>
                Results load after generation finishes. Each badge distinguishes loading, missing,
                and options that were not requested.
              </p>
            </div>
            <div className="diagnostics-badges">
              <DiagnosticBadge label="Layer activity" state={layerDiagnostic} />
              <DiagnosticBadge label="Token likelihood" state={likelihoodDiagnostic} />
              <DiagnosticBadge label="J-space" state={jspaceDiagnostic} />
            </div>
            {resultError && activeRun ? (
              <div className="diagnostics-error">
                <span>{resultError}</span>
                <button onClick={() => void fetchResult(activeRun.id)} type="button">
                  Retry diagnostics
                </button>
              </div>
            ) : null}
          </section>

          <section className="panel trace-panel">
            <div className="section-heading section-heading--compact">
              <div>
                <h2>Layer activity</h2>
                <p>
                  Transformer contribution by layer and token
                  {result?.traces.capture?.truncated ? " · final trace window" : ""}
                </p>
              </div>
              <div className="heat-legend"><span>Low</span><i /><span>High</span></div>
            </div>
            <TraceMatrix
              tokens={result?.traces.tokens}
              residualUpdates={result?.traces.residual_update_norms}
              residualNorms={result?.traces.hidden_state_norms}
              emptyMessage={diagnosticEmptyMessage(
                layerDiagnostic,
                "Enable Activations before starting the run.",
              )}
              emptyTitle={diagnosticEmptyTitle(layerDiagnostic, "Layer activity")}
            />
          </section>

          <div className="detail-grid">
            <section className="panel token-panel">
              <div className="section-heading section-heading--compact">
                <div>
                  <h2>Token likelihood</h2>
                  <p>Generated-token confidence</p>
                </div>
              </div>
              <div className="token-table" role="table" aria-label="Generated token probabilities">
                {tokenTraces.some((token) => token.logprob !== null) ? (
                  tokenTraces.slice(0, 128).map((token) => (
                    <div className="token-row" role="row" key={`${token.index}-${token.token_id}`}>
                      <span>{token.index + 1}</span>
                      <code>{token.text.replaceAll(" ", "·") || "∅"}</code>
                      <span>{token.logprob === null ? "—" : token.logprob.toFixed(3)}</span>
                      <span className="probability-track">
                        <i
                          style={{
                            width: `${token.logprob === null ? 0 : Math.exp(token.logprob) * 100}%`,
                          }}
                        />
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="empty-copy">
                    {diagnosticEmptyMessage(
                      likelihoodDiagnostic,
                      result?.generation && !result.generation.logprobs_captured
                        ? `The ${result.generation.effective_max_new_tokens.toLocaleString()}-token budget exceeded the ${result.generation.logprob_token_limit.toLocaleString()}-token likelihood capture limit.`
                        : "Enable Logprobs before starting the run.",
                    )}
                  </p>
                )}
              </div>
            </section>

          </div>

          <JSpaceViewer
            available={health?.model.jspace_available ?? false}
            loading={resultStatus === "loading"}
            reason={health?.model.jspace_reason}
            requested={activeRun?.request.traces.jspace ?? false}
            resultError={resultError}
            summary={result?.traces.jspace}
          />
        </section>
      </div>
      <div className="reasoning-tab" hidden={activeTab !== "reasoning"}>
        <SearchLab
          health={health}
          onInspectRun={inspectReasoningRun}
          onRunsChanged={refreshRuns}
        />
      </div>
    </main>
  );
}

function StatusDot({ active }: { active: boolean }) {
  return <span className={`status-dot ${active ? "is-active" : ""}`} aria-hidden="true" />;
}

function OptionToggle({
  checked,
  disabled = false,
  label,
  onChange,
}: {
  checked: boolean;
  disabled?: boolean;
  label: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className={`option-toggle ${disabled ? "is-disabled" : ""}`}>
      <input
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        type="checkbox"
      />
      <span>{label}</span>
    </label>
  );
}

function DiagnosticBadge({ label, state }: { label: string; state: DiagnosticState }) {
  return (
    <div className={`diagnostic-badge diagnostic-badge--${state}`}>
      <span>{label}</span>
      <strong>{diagnosticStateCopy(state)}</strong>
    </div>
  );
}

function diagnosticLabel(
  run: RunRecord | null,
  requested: boolean,
  ready: boolean,
  resultStatus: "idle" | "loading" | "ready" | "error",
): DiagnosticState {
  if (!run) return "no-run";
  if (ready) return "ready";
  if (!requested) return "not-requested";
  if (resultStatus === "loading") return "loading";
  if (resultStatus === "error") return "error";
  if (!terminalStatuses.has(run.status)) return "waiting";
  return "unavailable";
}

function diagnosticStateCopy(state: DiagnosticState): string {
  return {
    "no-run": "No run",
    "not-requested": "Not requested",
    waiting: "Waiting",
    loading: "Loading",
    ready: "Ready",
    unavailable: "No data",
    error: "Load error",
  }[state];
}

function diagnosticEmptyTitle(state: DiagnosticState, label: string): string {
  if (state === "loading") return `Loading ${label.toLowerCase()}`;
  if (state === "waiting") return "Waiting for the run to finish";
  if (state === "error") return "Diagnostics could not be loaded";
  if (state === "not-requested") return `${label} was not requested`;
  if (state === "unavailable") return `${label} contains no usable data`;
  return `No ${label.toLowerCase()} data`;
}

function diagnosticEmptyMessage(state: DiagnosticState, fallback: string): string {
  if (state === "loading") return "Reading and validating the diagnostics artifact…";
  if (state === "waiting") return "Diagnostics appear after generation and trace capture finish.";
  if (state === "error") return "Use Retry diagnostics above to load the result again.";
  if (state === "no-run") return "Select a completed run or start a new one.";
  return fallback;
}
