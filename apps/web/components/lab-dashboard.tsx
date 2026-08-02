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

const terminalStatuses = new Set(["completed", "failed", "cancelled"]);

const initialRequest: CreateRunRequest = {
  prompt: "Explain how a Jacobian lens differs from a logit lens in two sentences.",
  system_prompt: null,
  generation: {
    max_new_tokens: 128,
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
  const [error, setError] = useState<string | null>(null);
  const lastSequence = useRef(0);

  const refreshRuns = useCallback(async () => {
    const nextRuns = await api.runs();
    setRuns(nextRuns);
  }, []);

  useEffect(() => {
    Promise.all([api.health(), api.runs()])
      .then(([nextHealth, nextRuns]) => {
        setHealth(nextHealth);
        setRuns(nextRuns);
        setActiveRun(nextRuns[0] ?? null);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const loadRun = useCallback(async (run: RunRecord) => {
    lastSequence.current = 0;
    setEvents([]);
    setActiveRun(run);
    setStreamedText(run.output_text ?? "");
    setResult(null);
    setError(null);
    const history = await api.events(run.id);
    lastSequence.current = history.at(-1)?.sequence ?? 0;
    setEvents(history);
    if (run.status === "completed") {
      setResult(await api.result(run.id));
    }
  }, []);

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
          Promise.all([api.run(activeRun.id), api.runs()])
            .then(async ([updated, nextRuns]) => {
              setActiveRun(updated);
              setRuns(nextRuns);
              if (updated.status === "completed") {
                setResult(await api.result(updated.id));
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

  const latestPhase = events.at(-1)?.type ?? activeRun?.status ?? "idle";
  const tokenTraces = useMemo(() => result?.tokens ?? [], [result]);
  const confidence = useMemo(() => {
    const logprobs = tokenTraces.flatMap((token) =>
      token.logprob === null ? [] : [Math.exp(token.logprob)],
    );
    if (!logprobs.length) return null;
    return logprobs.reduce((sum, value) => sum + value, 0) / logprobs.length;
  }, [tokenTraces]);

  return (
    <main className="lab-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand__glyph" aria-hidden="true">
            λ
          </span>
          <div>
            <p>LOCAL INTERPRETABILITY WORKBENCH</p>
            <h1>LLM Lab</h1>
          </div>
        </div>
        <div className="runtime-strip">
          <StatusDot active={health?.model.available ?? false} />
          <span>{health?.model.architecture ?? "API offline"}</span>
          <span className="runtime-strip__divider" />
          <span>{health?.model.device ?? (health?.model.loaded ? "loaded" : "cold")}</span>
          <span className="runtime-strip__divider" />
          <span>{formatBytes(health?.model.weight_bytes ?? null)}</span>
        </div>
      </header>

      <div className="workspace">
        <aside className="run-rail">
          <div className="rail-heading">
            <span>RUN INDEX</span>
            <span>{runs.length.toString().padStart(2, "0")}</span>
          </div>
          <div className="run-list">
            {runs.length ? (
              runs.map((run) => (
                <button
                  className={`run-item ${activeRun?.id === run.id ? "is-active" : ""}`}
                  key={run.id}
                  onClick={() => void loadRun(run)}
                  type="button"
                >
                  <span className={`run-state run-state--${run.status}`} />
                  <span className="run-item__copy">
                    <strong>{run.prompt}</strong>
                    <small>
                      {formatTimestamp(run.created_at)} · {run.tokens_generated} tok
                    </small>
                  </span>
                  <span className="run-item__id">{run.id.slice(0, 5)}</span>
                </button>
              ))
            ) : (
              <p className="rail-empty">Runs will appear here after the first experiment.</p>
            )}
          </div>
          <div className="rail-model">
            <span>MODEL SOURCE</span>
            <code title={health?.model.path}>{health?.model.path ?? "Connecting…"}</code>
            <dl>
              <div>
                <dt>LAYERS</dt>
                <dd>{health?.model.layers ?? "—"}</dd>
              </div>
              <div>
                <dt>WIDTH</dt>
                <dd>{health?.model.hidden_size ?? "—"}</dd>
              </div>
              <div>
                <dt>VOCAB</dt>
                <dd>{health?.model.vocabulary_size?.toLocaleString() ?? "—"}</dd>
              </div>
            </dl>
          </div>
        </aside>

        <section className="main-stage">
          <form className="prompt-panel" onSubmit={submitRun}>
            <div className="panel-label">
              <span>01 / INPUT PROBE</span>
              <span className="phase-pill">{latestPhase.replace(".", " / ")}</span>
            </div>
            <label className="prompt-field">
              <span className="sr-only">Prompt</span>
              <textarea
                onChange={(event) => setRequest({ ...request, prompt: event.target.value })}
                placeholder="Enter a prompt to inspect…"
                required
                rows={4}
                value={request.prompt}
              />
              <span className="prompt-counter">{request.prompt.length} chars</span>
            </label>
            <div className="control-grid">
              <label>
                <span>MAX TOKENS</span>
                <input
                  min={1}
                  max={1024}
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
              </label>
              <label>
                <span>TEMPERATURE</span>
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
              <TraceToggle
                checked={request.generation.enable_thinking}
                label="THINKING"
                onChange={(checked) =>
                  setRequest({
                    ...request,
                    generation: { ...request.generation, enable_thinking: checked },
                  })
                }
              />
              <TraceToggle
                checked={request.traces.logprobs}
                label="LOGPROBS"
                onChange={(checked) =>
                  setRequest({ ...request, traces: { ...request.traces, logprobs: checked } })
                }
              />
              <TraceToggle
                checked={request.traces.hidden_states}
                label="ACTIVATIONS"
                onChange={(checked) =>
                  setRequest({
                    ...request,
                    traces: { ...request.traces, hidden_states: checked },
                  })
                }
              />
              <TraceToggle
                checked={request.traces.attentions}
                label="ATTENTION"
                onChange={(checked) =>
                  setRequest({ ...request, traces: { ...request.traces, attentions: checked } })
                }
              />
              <TraceToggle
                checked={request.traces.jspace}
                disabled={!health?.model.jspace_available}
                label="J-SPACE"
                onChange={(checked) =>
                  setRequest({ ...request, traces: { ...request.traces, jspace: checked } })
                }
              />
              <button
                className="run-button"
                disabled={submitting || !health?.model.available}
                type="submit"
              >
                <span>{submitting ? "QUEUING" : "RUN PROBE"}</span>
                <span aria-hidden="true">→</span>
              </button>
            </div>
            {error ? <p className="error-banner">{error}</p> : null}
          </form>

          <section className="output-panel">
            <div className="panel-label">
              <span>02 / MODEL OUTPUT</span>
              <div className="metrics-inline">
                <span>{activeRun?.tokens_generated ?? 0} TOKENS</span>
                <span>{confidence === null ? "—" : `${(confidence * 100).toFixed(1)}%`} MEAN P</span>
              </div>
            </div>
            <div className="output-copy">
              {streamedText || activeRun?.output_text ? (
                <p>{streamedText || activeRun?.output_text}</p>
              ) : (
                <p className="output-placeholder">
                  The generated response will stream here. Trace evidence remains separate below.
                </p>
              )}
            </div>
          </section>

          <section className="trace-panel">
            <div className="panel-label">
              <span>03 / RESIDUAL STREAM MAGNITUDE</span>
              <div className="legend">
                <span>LOW</span>
                <span className="legend__bar" />
                <span>HIGH</span>
              </div>
            </div>
            <TraceMatrix
              tokens={result?.traces.tokens}
              values={result?.traces.hidden_state_norms}
            />
          </section>

          <div className="lower-grid">
            <section className="token-panel">
              <div className="panel-label">
                <span>04 / TOKEN LIKELIHOOD</span>
                <span>TOP-1 TRACE</span>
              </div>
              <div className="token-table" role="table" aria-label="Generated token probabilities">
                {tokenTraces.length ? (
                  tokenTraces.slice(0, 24).map((token) => (
                    <div className="token-row" role="row" key={`${token.index}-${token.token_id}`}>
                      <span>{token.index.toString().padStart(3, "0")}</span>
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
                  <p className="table-empty">Enable logprobs and complete a run to inspect token confidence.</p>
                )}
              </div>
            </section>

            <section className="jspace-panel">
              <div className="panel-label">
                <span>05 / J-SPACE LENS</span>
                <StatusDot active={health?.model.jspace_available ?? false} />
              </div>
              <div className="jspace-state">
                <div className="orbital" aria-hidden="true">
                  <span />
                  <span />
                  <i>J</i>
                </div>
                <div>
                  <strong>
                    {health?.model.jspace_available ? "Lens ready" : "Lens checkpoint required"}
                  </strong>
                  <p>
                    {health?.model.jspace_reason ??
                      "A compatible fitted lens is available for concept readout."}
                  </p>
                </div>
              </div>
              <p className="evidence-note">
                J-space concepts are approximate activation readouts—not a complete reasoning transcript.
              </p>
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

function TraceToggle({
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
    <label className={`trace-toggle ${disabled ? "is-disabled" : ""}`}>
      <input
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        type="checkbox"
      />
      <span className="trace-toggle__box" aria-hidden="true" />
      <span>{label}</span>
    </label>
  );
}
