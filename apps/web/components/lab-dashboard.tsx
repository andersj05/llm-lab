"use client";

import type {
  CreateRunRequest,
  HealthResponse,
  RunEvent,
  RunRecord,
  RunResult,
} from "@llm-lab/contracts";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { TraceMatrix } from "@/components/trace-matrix";
import { api, subscribeToRun } from "@/lib/api";
import { formatBytes, formatTimestamp } from "@/lib/format";

const terminalStatuses = new Set<RunRecord["status"]>([
  "completed",
  "failed",
  "cancelled",
]);

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
  const lastSequence = useRef(0);

  const refreshRuns = useCallback(async () => {
    const nextRuns = await api.runs();
    setRuns(nextRuns);
    return nextRuns;
  }, []);

  const resetSelection = useCallback(() => {
    lastSequence.current = 0;
    setActiveRun(null);
    setResult(null);
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
    setError(null);
    try {
      const history = await api.events(run.id);
      lastSequence.current = history.at(-1)?.sequence ?? 0;
      setEvents(history);
      if (run.status === "completed" || run.status === "cancelled") {
        setResult(await api.result(run.id).catch(() => null));
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to load run");
    }
  }, []);

  useEffect(() => {
    Promise.all([api.health(), api.runs()])
      .then(([nextHealth, nextRuns]) => {
        setHealth(nextHealth);
        setRuns(nextRuns);
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
                setResult(await api.result(updated.id).catch(() => null));
              }
            })
            .catch((reason: Error) => setError(reason.message));
        }
      },
      () => setError("The live event stream disconnected."),
    );
  }, [activeRun]);

  async function submitRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setResult(null);
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

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="app-title">
          <h1>LLM Lab</h1>
          <span>Local model workspace</span>
        </div>
        <div className="model-status">
          <StatusDot active={health?.model.available ?? false} />
          <span>{health?.model.architecture ?? "API unavailable"}</span>
          <span className="model-status__detail">
            {health?.model.device ?? (health?.model.loaded ? "loaded" : "not loaded")}
          </span>
        </div>
      </header>

      <div className="app-body">
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
                Activation and attention capture use only the final 256-token window. Long generation
                is no longer blocked by that limit.
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

          <section className="panel trace-panel">
            <div className="section-heading section-heading--compact">
              <div>
                <h2>Activation magnitude</h2>
                <p>
                  Residual-stream norm by layer and token
                  {result?.traces.capture?.truncated ? " · final trace window" : ""}
                </p>
              </div>
              <div className="heat-legend"><span>Low</span><i /><span>High</span></div>
            </div>
            <TraceMatrix
              tokens={result?.traces.tokens}
              values={result?.traces.hidden_state_norms}
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
                    No logprobs were captured. Budgets above 1,024 tokens skip full-vocabulary
                    scores to keep GPU memory bounded.
                  </p>
                )}
              </div>
            </section>

            <section className="panel lens-panel">
              <div className="section-heading section-heading--compact">
                <div>
                  <h2>J-space lens</h2>
                  <p>Optional concept readout</p>
                </div>
                <StatusDot active={health?.model.jspace_available ?? false} />
              </div>
              <div className="lens-copy">
                <strong>
                  {health?.model.jspace_available ? "Lens ready" : "Not configured"}
                </strong>
                <p>
                  {health?.model.jspace_reason ??
                    "A compatible fitted lens is available for concept readout."}
                </p>
                <p className="evidence-note">
                  These measurements are model telemetry, not a hidden chain-of-thought transcript.
                </p>
              </div>
            </section>
          </div>
        </section>
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
