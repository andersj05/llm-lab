"use client";

import type { CreateRunRequest, HealthResponse, RunRecord } from "@llm-lab/contracts";
import { FormEvent, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";

interface ReasoningWorkspaceProps {
  health: HealthResponse | null;
  onInspectRun: (runId: string) => Promise<void>;
  onRunsChanged: () => Promise<RunRecord[]>;
}

interface BranchNode {
  key: string;
  label: string;
  description: string;
  status: RunRecord["status"] | "draft";
  runId?: string;
  output?: string;
  error?: string | null;
}

const branchRoles = [
  {
    key: "decompose",
    label: "Decompose",
    description: "Split the problem into assumptions, constraints, and smaller decisions.",
    instruction:
      "Decompose the problem. Identify assumptions, constraints, unknowns, and a compact solution plan.",
  },
  {
    key: "construct",
    label: "Construct",
    description: "Build a direct candidate solution from first principles.",
    instruction:
      "Construct the strongest direct solution you can from first principles. State the candidate and its support.",
  },
  {
    key: "challenge",
    label: "Challenge",
    description: "Search for counterexamples, failure modes, and hidden premises.",
    instruction:
      "Act as an adversarial reviewer. Find counterexamples, hidden premises, and ways a plausible answer could fail.",
  },
  {
    key: "reframe",
    label: "Reframe",
    description: "Try an analogy or alternate representation that changes the search space.",
    instruction:
      "Reframe the problem using an analogy, alternate representation, or different level of abstraction. Derive a candidate insight.",
  },
] as const;

const terminalStatuses = new Set<RunRecord["status"]>([
  "completed",
  "failed",
  "cancelled",
]);

export function ReasoningWorkspace({
  health,
  onInspectRun,
  onRunsChanged,
}: ReasoningWorkspaceProps) {
  const [problem, setProblem] = useState("");
  const [branchCount, setBranchCount] = useState(3);
  const [branchTokens, setBranchTokens] = useState(256);
  const [captureDiagnostics, setCaptureDiagnostics] = useState(false);
  const [branches, setBranches] = useState<BranchNode[]>([]);
  const [synthesis, setSynthesis] = useState<RunRecord | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const experimentId = useRef(0);

  const completedBranches = useMemo(
    () => branches.filter((branch) => branch.status === "completed").length,
    [branches],
  );

  function updateBranch(key: string, patch: Partial<BranchNode>) {
    setBranches((current) =>
      current.map((branch) => (branch.key === key ? { ...branch, ...patch } : branch)),
    );
  }

  async function startExperiment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedProblem = problem.trim();
    if (!trimmedProblem || busy) return;

    const currentExperiment = experimentId.current + 1;
    experimentId.current = currentExperiment;
    const roles = branchRoles.slice(0, branchCount);
    setBusy(true);
    setError(null);
    setSynthesis(null);
    setBranches(
      roles.map((role) => ({
        key: role.key,
        label: role.label,
        description: role.description,
        status: "draft",
      })),
    );

    const outcomes = await Promise.allSettled(
      roles.map(async (role, index) => {
        try {
          const created = await api.createRun(
            branchRequest(
              trimmedProblem,
              role.instruction,
              branchTokens,
              100 + index,
              captureDiagnostics,
              health?.model.jspace_available ?? false,
            ),
          );
          updateBranch(role.key, { runId: created.id, status: created.status });
          const completed = await waitForTerminal(created.id, (record) => {
            if (experimentId.current !== currentExperiment) return;
            updateBranch(role.key, {
              error: record.error,
              output: record.output_text ?? undefined,
              status: record.status,
            });
          });
          if (completed.status !== "completed" || !completed.output_text) {
            throw new Error(completed.error || `${role.label} branch did not complete`);
          }
          return { role, record: completed };
        } catch (reason) {
          const message = reason instanceof Error ? reason.message : "Branch failed";
          updateBranch(role.key, { error: message, status: "failed" });
          throw reason;
        }
      }),
    );

    await onRunsChanged();
    if (experimentId.current !== currentExperiment) return;
    const completed = outcomes.flatMap((outcome) =>
      outcome.status === "fulfilled" ? [outcome.value] : [],
    );
    if (!completed.length) {
      setError("No branch completed, so there was nothing to synthesize.");
      setBusy(false);
      return;
    }

    try {
      const created = await api.createRun(
        synthesisRequest(
          trimmedProblem,
          completed.map(({ role, record }) => ({
            label: role.label,
            output: record.output_text ?? "",
          })),
          Math.max(384, branchTokens),
          captureDiagnostics,
          health?.model.jspace_available ?? false,
        ),
      );
      setSynthesis(created);
      const completedSynthesis = await waitForTerminal(created.id, (record) => {
        if (experimentId.current === currentExperiment) setSynthesis(record);
      });
      if (completedSynthesis.status !== "completed") {
        throw new Error(completedSynthesis.error || "Synthesis did not complete");
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Synthesis failed");
    } finally {
      if (experimentId.current === currentExperiment) setBusy(false);
      await onRunsChanged();
    }
  }

  async function stopExperiment() {
    experimentId.current += 1;
    setBusy(false);
    const runIds = [
      ...branches
        .filter((branch) => branch.runId && !terminalStatuses.has(branch.status as RunRecord["status"]))
        .map((branch) => branch.runId as string),
      ...(synthesis && !terminalStatuses.has(synthesis.status) ? [synthesis.id] : []),
    ];
    await Promise.allSettled(runIds.map((runId) => api.cancelRun(runId)));
    await onRunsChanged();
  }

  return (
    <div className="reasoning-workspace">
      <section className="reasoning-intro">
        <span>Experimental workspace</span>
        <h2>Reason across a graph, not one uninterrupted trace.</h2>
        <p>
          LLM Lab fans one problem into independent searches, then merges their outputs through a
          separate synthesis run. Each model call is still next-token autoregression; the nonlinear
          part is the computation graph around those calls.
        </p>
      </section>

      <form className="panel reasoning-composer" onSubmit={startExperiment}>
        <label>
          <span>Problem or research question</span>
          <textarea
            onChange={(event) => setProblem(event.target.value)}
            placeholder="Describe a problem where competing hypotheses or representations would help…"
            required
            rows={5}
            value={problem}
          />
        </label>
        <div className="reasoning-controls">
          <label>
            <span>Parallel branches</span>
            <select
              onChange={(event) => setBranchCount(Number(event.target.value))}
              value={branchCount}
            >
              <option value={2}>2 branches</option>
              <option value={3}>3 branches</option>
              <option value={4}>4 branches</option>
            </select>
          </label>
          <label>
            <span>Tokens per branch</span>
            <input
              max={1024}
              min={64}
              onChange={(event) => setBranchTokens(Number(event.target.value))}
              step={64}
              type="number"
              value={branchTokens}
            />
          </label>
          <label className="reasoning-check">
            <input
              checked={captureDiagnostics}
              onChange={(event) => setCaptureDiagnostics(event.target.checked)}
              type="checkbox"
            />
            <span>Capture diagnostics for every node</span>
          </label>
          <div className="reasoning-actions">
            {busy ? (
              <button className="secondary-button stop-button" onClick={stopExperiment} type="button">
                Stop experiment
              </button>
            ) : null}
            <button
              className="primary-button"
              disabled={busy || !problem.trim() || !health?.model.available}
              type="submit"
            >
              {busy ? "Graph running…" : "Run reasoning graph"}
            </button>
          </div>
        </div>
        {error ? <p className="message message--error">{error}</p> : null}
      </form>

      <section className="reasoning-graph" aria-label="Reasoning graph">
        <article className="reasoning-node reasoning-node--root">
          <span>Input</span>
          <strong>{problem.trim() || "Your problem begins here"}</strong>
        </article>

        <div className="reasoning-branches">
          {(branches.length
            ? branches
            : branchRoles.slice(0, branchCount).map((role) => ({ ...role, status: "draft" as const }))
          ).map((branch) => (
            <article className={`reasoning-node reasoning-node--${branch.status}`} key={branch.key}>
              <div className="reasoning-node__heading">
                <span>{branch.label}</span>
                <i>{branch.status}</i>
              </div>
              <p>{branch.description}</p>
              {"output" in branch && branch.output ? (
                <div className="reasoning-node__output">{branch.output}</div>
              ) : (
                <div className="reasoning-node__placeholder">
                  {branch.status === "draft" ? "Independent search node" : "Working…"}
                </div>
              )}
              {"error" in branch && branch.error ? (
                <small className="reasoning-node__error">{branch.error}</small>
              ) : null}
              {"runId" in branch && branch.runId ? (
                <button onClick={() => void onInspectRun(branch.runId as string)} type="button">
                  Open diagnostics
                </button>
              ) : null}
            </article>
          ))}
        </div>

        <article className={`reasoning-node reasoning-node--synthesis ${synthesis ? `reasoning-node--${synthesis.status}` : ""}`}>
          <div className="reasoning-node__heading">
            <span>Synthesis</span>
            <i>{synthesis?.status ?? (completedBranches ? "waiting" : "merge node")}</i>
          </div>
          {synthesis?.output_text ? (
            <div className="reasoning-node__output reasoning-node__output--final">
              {synthesis.output_text}
            </div>
          ) : (
            <p>Compare the branches, resolve contradictions, and produce one calibrated answer.</p>
          )}
          {synthesis ? (
            <button onClick={() => void onInspectRun(synthesis.id)} type="button">
              Open synthesis diagnostics
            </button>
          ) : null}
        </article>
      </section>
    </div>
  );
}

function branchRequest(
  problem: string,
  instruction: string,
  maxTokens: number,
  seed: number,
  captureDiagnostics: boolean,
  jspaceAvailable: boolean,
): CreateRunRequest {
  return {
    prompt: `Problem:\n${problem}\n\nBranch role:\n${instruction}\n\nReturn a concise structured result with a candidate conclusion, supporting evidence, and uncertainty.`,
    system_prompt:
      "You are one independent node in a nonlinear reasoning graph. Explore your assigned route without assuming other branches agree with you.",
    generation: {
      max_new_tokens: maxTokens,
      temperature: 0.65,
      top_p: 0.9,
      do_sample: true,
      seed,
      enable_thinking: false,
    },
    traces: {
      logprobs: captureDiagnostics,
      hidden_states: captureDiagnostics,
      attentions: false,
      jspace: captureDiagnostics && jspaceAvailable,
      persist_raw: false,
    },
  };
}

function synthesisRequest(
  problem: string,
  branches: Array<{ label: string; output: string }>,
  maxTokens: number,
  captureDiagnostics: boolean,
  jspaceAvailable: boolean,
): CreateRunRequest {
  const evidence = branches
    .map(({ label, output }) => `## ${label} branch\n${output}`)
    .join("\n\n");
  return {
    prompt: `Original problem:\n${problem}\n\nIndependent branch results:\n${evidence}\n\nSynthesize a final answer. Compare disagreements, preserve useful minority evidence, state uncertainty, and do not merely choose the most repeated claim.`,
    system_prompt:
      "You are the merge node in a reasoning graph. Treat every branch as fallible evidence and produce a concise, checked synthesis.",
    generation: {
      max_new_tokens: maxTokens,
      temperature: 0,
      top_p: 1,
      do_sample: false,
      seed: 42,
      enable_thinking: false,
    },
    traces: {
      logprobs: captureDiagnostics,
      hidden_states: captureDiagnostics,
      attentions: false,
      jspace: captureDiagnostics && jspaceAvailable,
      persist_raw: false,
    },
  };
}

async function waitForTerminal(
  runId: string,
  onProgress: (record: RunRecord) => void,
): Promise<RunRecord> {
  while (true) {
    const record = await api.run(runId);
    onProgress(record);
    if (terminalStatuses.has(record.status)) return record;
    await new Promise((resolve) => window.setTimeout(resolve, 750));
  }
}
